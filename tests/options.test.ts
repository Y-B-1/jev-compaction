import { describe, expect, it } from 'vitest';
import { parseToolThresholds } from '../hooks/fast-jev.js';
import { compact } from '../src/compact.js';
import { STATE_CONTEXT, STATE_CONTEXT_ARCHIVE } from '../src/state.js';
import type { JevAsker, Message } from '../src/types.js';

const msg = (role: Message['role'], text: string, extra: Partial<Message> = {}): Message => ({ role, text, toolUses: [], ...extra });
const big = 'detail line of output\n'.repeat(200);
const session = (tool = 'Bash'): Message[] => [
  msg('user', 'Fix the build.'),
  msg('assistant', '', { toolUses: [{ tool_use_id: 'a', tool, input: { command: 'make' }, text: big }] }),
  msg('user', '', { toolResults: [{ tool_use_id: 'a', text: big }] }),
  msg('assistant', 'Fixed.'),
  msg('user', 'ok'),
  msg('assistant', 'done'),
];
const recording = (answer: (key: string) => number, seen: { state: any; keys: string[] }[] = []): JevAsker => ({
  async ask(state, questions) {
    seen.push({ state, keys: Object.keys(questions) });
    return { answers: Object.fromEntries(Object.keys(questions).map((k) => [k, { type: 'noul', noul: answer(k) }])) } as never;
  },
});

describe('stateWording', () => {
  it('says dropped output is archived only when an archive is given', async () => {
    const seen: { state: any; keys: string[] }[] = [];
    await compact(session(), recording(() => 0.9, seen), { stateWording: 'archive', archive: (id) => `arch/${id}`, preserveRecentMessages: 3 });
    await compact(session(), recording(() => 0.9, seen), { stateWording: 'archive', preserveRecentMessages: 3 });
    await compact(session(), recording(() => 0.9, seen), { preserveRecentMessages: 3 });
    expect(seen.map((s) => s.state.context)).toEqual([STATE_CONTEXT_ARCHIVE, STATE_CONTEXT, STATE_CONTEXT]);
  });
});

describe('resultChecks', () => {
  const run = (answer: (key: string) => number) => {
    const seen: { state: any; keys: string[] }[] = [];
    return compact(session(), recording(answer, seen), { resultChecks: true, preserveRecentMessages: 3 }).then((out) => ({ out, seen }));
  };

  it('asks four checks about the result instead of one keep question', async () => {
    const { seen } = await run(() => 0.9);
    expect(seen[0]!.keys.sort()).toEqual(['call_t1', 'con_t1', 'err_t1', 'rerun_t1', 'use_t1']);
  });

  it('keeps a result any of error, constraint or in-use says to keep', async () => {
    const { out } = await run((k) => (k.startsWith('con_') ? 0.6 : k.startsWith('call_') ? 0.9 : 0));
    expect(out.decisions[0]!.action).toBe('keep');
  });

  it('drops an error a rerun would reproduce', async () => {
    const { out } = await run((k) => (k.startsWith('err_') || k.startsWith('rerun_') || k.startsWith('call_') ? 0.9 : 0));
    expect(out.decisions[0]!.action).toBe('drop_result');
  });
});

describe('keepThresholdByTool', () => {
  it('applies a tool’s own threshold and the global one to other tools', async () => {
    const answer = (k: string) => (k.startsWith('result_') ? 0.3 : 0.9);
    const bash = await compact(session('Bash'), recording(answer), { keepThresholdByTool: { Bash: 0.5 }, preserveRecentMessages: 3 });
    const read = await compact(session('Read'), recording(answer), { keepThresholdByTool: { Bash: 0.5 }, preserveRecentMessages: 3 });
    expect(bash.decisions[0]!.action).toBe('drop_result');
    expect(read.decisions[0]!.action).toBe('keep');
  });

  it('parses the plugin option and skips invalid entries', () => {
    expect(parseToolThresholds('Bash=0.2, Read=0.1, Grep=2, junk')).toEqual({ Bash: 0.2, Read: 0.1 });
    expect(parseToolThresholds('  ')).toBeUndefined();
    expect(parseToolThresholds(3)).toBeUndefined();
  });
});

describe('hybridOriginals', () => {
  const turn = (n: number) => [
    { role: 'user', text: `Step ${n}: check module ${n}.`, toolUses: [], handle: `u${n}` },
    { role: 'assistant', text: '', toolUses: [{ tool_use_id: `c${n}`, tool: 'Bash', input: { command: `test ${n}` }, text: big }], handle: `a${n}` },
    { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: `c${n}`, text: big, isError: false }], handle: `r${n}` },
    { role: 'assistant', text: `Done with step ${n}.`, toolUses: [], handle: `d${n}` },
  ];
  const transcript = () => [1, 2, 3, 4, 5, 6, 7, 8].flatMap(turn);
  const run = async (hybridOriginals: boolean) => {
    const hooks: Record<string, (...args: any[]) => Promise<any>> = {};
    const { register } = await import('../hooks/fast-jev.js');
    (register as any)(
      (event: string, a: unknown, b?: any) => {
        hooks[event] = typeof b === 'function' ? b : (a as any);
      },
      // The two oldest results go (Jev answers 0.01), a cut under the 50% minimum, so the hybrid runs.
      { apiKey: 'k', preserveRecentMessages: 2, archiveResults: false, minReductionRatio: 0.5, hybridOriginals, proseTrim: false },
    );
    const summarized: unknown[][] = [];
    const $ = {
      clock: { sleep: () => new Promise(() => {}), now: async () => 0 },
      http: {
        fetch: async (_url: string, init: { body: string }) => {
          const { questions } = JSON.parse(init.body) as { questions: Record<string, unknown> };
          const answers = Object.fromEntries(Object.keys(questions).map((k) => [k, { type: 'noul', noul: k === 'result_t1' || k === 'result_t2' ? 0.01 : 0.9 }]));
          return { status: 200, ok: true, text: JSON.stringify({ answers }) };
        },
      },
      ui: { log: () => {}, toast: () => {} },
    };
    const next = Object.assign(
      async (e: { messages: unknown[] }) => {
        summarized.push(e.messages);
        return { messages: [{ role: 'user', text: 'Summary.', toolUses: [] }] };
      },
      { signal: new AbortController().signal, budget: { ms: 1e4, remainingMs: 1e4 } },
    );
    const out = await hooks['session.compact']!($, { trigger: 'auto', messages: transcript() }, next);
    return { out, older: summarized[0] as { text: string }[] };
  };

  it('hands the summarizer the original older rows instead of their truncated form', async () => {
    const trimmed = await run(false);
    expect(JSON.stringify(trimmed.older)).toContain('[jev-compaction truncated');
    const original = await run(true);
    expect(original.older).toEqual(transcript().slice(0, original.older.length));
    expect(original.older.length).toBe(trimmed.older.length);
    // The recent rows after the summary are the trimmed ones either way.
    expect(original.out.messages.slice(1)).toEqual(trimmed.out.messages.slice(1));
  });
});
