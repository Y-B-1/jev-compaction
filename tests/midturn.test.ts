import { describe, expect, it } from 'vitest';
import { register } from '../hooks/fast-jev.js';
import { type MidTurn, recordMidTurn, restoreMidTurn } from '../src/midturn.js';
import type { SessionMessage } from 'claude-code';

const LEAD = 'The user sent a new message while you were working:';
const attachment = (said: string) =>
  `${LEAD}\n${said}\n\nThis is how Claude Code surfaces messages the user sends mid-turn — within the running turn, often alongside the next tool result, rather than as a separate conversation turn. Address the message above as you continue this turn.`;

const prompt = (text: string): SessionMessage => ({ role: 'user', text, toolUses: [], handle: `u-${text.slice(0, 8)}` });
const call = (id: string): SessionMessage => ({
  role: 'assistant',
  text: '',
  toolUses: [{ tool_use_id: id, tool: 'Read', input: { file_path: `/w/${id}.txt` } } as never],
  handle: `a-${id}`,
});
const result = (id: string, text = 'file body'): SessionMessage => ({
  role: 'user',
  text: '',
  toolUses: [],
  toolResults: [{ tool_use_id: id, text, isError: false } as never],
  handle: `r-${id}`,
});

describe('recordMidTurn', () => {
  it('records the message with the tool results it arrived after, once', () => {
    const pending: MidTurn[] = [];
    const tail = [prompt('Read the files'), call('t1'), result('t1'), call('t2'), result('t2')];
    recordMidTurn(pending, attachment('the code word is HERON'), tail);
    recordMidTurn(pending, attachment('the code word is HERON'), tail);
    expect(pending).toEqual([{ text: attachment('the code word is HERON'), after: ['t2'] }]);
  });

  it('ignores a queued command that is not a mid-turn message', () => {
    const pending: MidTurn[] = [];
    recordMidTurn(pending, '<task-notification>done</task-notification>', [result('t1')]);
    expect(pending).toEqual([]);
  });
});

describe('restoreMidTurn', () => {
  it('puts a missing message back on the result row it arrived after, without that row’s handle', () => {
    const messages = [prompt('Read the files'), call('t1'), result('t1'), call('t2'), result('t2'), call('t3'), result('t3')];
    const pending = [{ text: attachment('the code word is HERON'), after: ['t2'] }];
    const { messages: out, restored } = restoreMidTurn(messages, pending);
    expect(restored).toBe(1);
    expect(out[4]).toEqual({ role: 'user', text: attachment('the code word is HERON'), toolUses: [], toolResults: messages[4]!.toolResults });
    expect(out.filter((m, i) => m !== messages[i])).toHaveLength(1);
  });

  it('leaves a message the history already holds', () => {
    const messages = [prompt('Read the files'), call('t1'), { ...result('t1'), text: attachment('the code word is HERON') }];
    const { messages: out, restored } = restoreMidTurn(messages, [{ text: attachment('the code word is HERON'), after: ['t1'] }]);
    expect(restored).toBe(0);
    expect(out).toEqual(messages);
  });

  it('restores a short message even when a reply contains the same word', () => {
    const messages = [prompt('go'), { role: 'assistant', text: 'ok, looking', toolUses: [], handle: 'a-x' } as SessionMessage, call('t1'), result('t1')];
    const { messages: out, restored } = restoreMidTurn(messages, [{ text: attachment('ok'), after: ['t1'] }]);
    expect(restored).toBe(1);
    expect(out[3]!.text).toBe(attachment('ok'));
  });

  it('uses the last result row of a parallel group', () => {
    const messages = [prompt('go'), call('p1'), call('p2'), result('p1'), result('p2')];
    const { messages: out } = restoreMidTurn(messages, [{ text: attachment('stop after this'), after: ['p1'] }]);
    expect(out[3]!.text).toBe('');
    expect(out[4]!.text).toContain('stop after this');
  });

  it('falls back to the newest result row, never a prompt the user typed', () => {
    const messages = [prompt('go'), call('t1'), result('t1'), prompt('next')];
    const { messages: out, restored } = restoreMidTurn(messages, [{ text: attachment('use port 8080'), after: ['gone'] }]);
    expect(restored).toBe(1);
    expect(out[2]!.text).toContain('use port 8080');
    expect(out[3]).toBe(messages[3]);
    expect(restoreMidTurn([prompt('go')], [{ text: attachment('x'), after: [] }]).restored).toBe(0);
  });
});

describe('the hook keeps a mid-turn message through a Jev compaction', () => {
  it('records it from the attachment and hands it back inside the history', async () => {
    const hooks: { event: string; matcher?: { type?: string }; hook: (...args: any[]) => Promise<any> }[] = [];
    (register as any)(
      (event: string, a: unknown, b?: unknown) =>
        hooks.push(typeof b === 'function' ? { event, matcher: a as { type?: string }, hook: b as any } : { event, hook: a as any }),
      { apiKey: 'k', preserveRecentMessages: 2, minReductionRatio: 0 },
    );
    const big = 'alpha beta gamma\n'.repeat(400);
    const history = [prompt('Read the four files'), call('t1'), result('t1', big), call('t2'), result('t2', big), call('t3'), result('t3', big), call('t4'), result('t4', big), { role: 'assistant', text: 'DONE', toolUses: [], handle: 'a-done' } as SessionMessage];
    const files = new Map<string, string>();
    const $ = {
      clock: { sleep: () => new Promise(() => {}), now: async () => Date.UTC(2026, 9, 26) },
      http: {
        fetch: async (_url: string, init: { body: string }) => {
          const { questions } = JSON.parse(init.body) as { questions: Record<string, unknown> };
          const answers = Object.fromEntries(Object.keys(questions).map((k) => [k, { type: 'noul', noul: k.startsWith('result_') ? 0.01 : 0.9 }]));
          return { status: 200, ok: true, text: JSON.stringify({ answers }) };
        },
      },
      ui: { log: () => {}, toast: () => {} },
      fs: {
        exists: async (p: string) => files.has(p),
        write: async (p: string, t: string) => void files.set(p, t),
        read: async (p: string) => {
          if (!files.has(p)) throw new Error('missing');
          return files.get(p)!;
        },
      },
      session: {
        cwd: async () => '/w',
        root: async () => '/w',
        // The API form is not needed here; the session read at the attachment is.
        messages: async (args?: { as?: string }) => {
          if (args?.as === 'api') throw new Error('not in this test');
          return history.slice(0, 7);
        },
      },
      process: { run: async () => ({ exitCode: 1, stdout: '' }) },
    };
    const next = Object.assign(async (e: unknown) => e, { signal: new AbortController().signal, budget: { ms: 1e4, remainingMs: 1e4 } });
    const onAttachment = hooks.find((h) => h.event === 'prompt.attachment' && h.matcher?.type === 'queued_command')!.hook;
    const event = { type: 'queued_command', text: attachment('the code word is HERON-7342'), origin: { kind: 'engine' } };
    expect(await onAttachment($, event, next)).toBe(event);
    const compact = hooks.find((h) => h.event === 'session.compact')!.hook;
    const out = await compact($, { trigger: 'manual', messages: history }, next);
    const holder = (out.messages as SessionMessage[]).find((m) => m.text.includes('HERON-7342'));
    expect(holder?.toolResults?.[0]?.tool_use_id).toBe('t3');
    expect(holder?.handle).toBeUndefined();
    // The record serves one compaction: a later one does not bring the message back again.
    const again = await compact($, { trigger: 'manual', messages: history }, next);
    expect((again.messages as SessionMessage[]).some((m) => m.text.includes('HERON-7342'))).toBe(false);
  });
});
