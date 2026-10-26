import { describe, expect, it } from 'vitest';
import { compactSession, resolveHookConfig, resumeSafeMessages } from '../hooks/fast-jev.ts';
import {
  COMPACTED_MARKER,
  compact,
  dedupeBlocks,
  type JevAsker,
  type JevQuestions,
  type Message,
} from '../src/index.js';

type SessionMessage = Message & { handle?: string };

const BIG = 'line of tool output\n'.repeat(120); // 2400 chars

function asker(answer: (name: string) => number, asked: string[] = []): JevAsker {
  return {
    async ask(_state, questions: JevQuestions) {
      asked.push(...Object.keys(questions));
      return {
        answers: Object.fromEntries(
          Object.keys(questions).map((key) => [key, { type: 'noul' as const, noul: answer(key) }]),
        ),
      };
    },
  };
}

function jevFetch(answer: (name: string) => number) {
  return async (_url: string, init?: { body?: string }) => {
    const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
    const answers = Object.fromEntries(
      Object.keys(questions).map((key) => [key, { type: 'noul', noul: answer(key) }]),
    );
    return { status: 200, ok: true, text: JSON.stringify({ answers }) };
  };
}

/** One call and its result, as two messages. */
function call(id: string, text: string, input: Record<string, unknown> = { file_path: `${id}.ts` }): Message[] {
  return [
    { role: 'assistant', text: '', toolUses: [{ tool_use_id: id, tool: 'Read', input, text }] },
    { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: id, text }] },
  ];
}

const tail: Message[] = [
  { role: 'user', text: 'next step', toolUses: [] },
  { role: 'assistant', text: 'done', toolUses: [] },
];

describe('resume-safe handles (#89, #137)', () => {
  function parallel(): SessionMessage[] {
    const rows = ['a', 'b', 'c', 'd'];
    return [
      { role: 'user', text: 'read the docs', toolUses: [], handle: 'u0' },
      { role: 'assistant', text: 'Reading them.', toolUses: [], handle: 'm1-text' },
      ...rows.map((id): SessionMessage => ({
        role: 'assistant',
        text: '',
        toolUses: [{ tool_use_id: id, tool: 'Read', input: { file_path: `${id}.md` }, text: BIG }],
        handle: `m1-${id}`,
      })),
      ...rows.map((id): SessionMessage => ({
        role: 'user',
        text: '',
        toolUses: [],
        toolResults: [{ tool_use_id: id, text: BIG }],
        handle: `r-${id}`,
      })),
      { role: 'user', text: 'thanks', toolUses: [], handle: 'u2' },
      { role: 'assistant', text: 'ok', toolUses: [], handle: 'm3' },
    ];
  }

  it('merges an older parallel response and its results into single handle-less messages', () => {
    const out = resumeSafeMessages(parallel());
    expect(out).toHaveLength(5);
    expect(out[0]?.handle).toBe('u0');
    expect(out[1]).toMatchObject({ role: 'assistant', text: 'Reading them.' });
    expect(out[1]?.handle).toBeUndefined();
    expect(out[1]?.toolUses.map((t) => t.tool_use_id)).toEqual(['a', 'b', 'c', 'd']);
    expect(out[2]?.handle).toBeUndefined();
    expect(out[2]?.toolResults?.map((r) => r.tool_use_id)).toEqual(['a', 'b', 'c', 'd']);
    expect(out[3]?.handle).toBe('u2');
    // the newest assistant run keeps its handle, and with it its reasoning
    expect(out[4]?.handle).toBe('m3');
  });

  it('keeps the newest assistant run whole, with handles, and strips its results', () => {
    const messages: SessionMessage[] = [
      { role: 'user', text: 'go', toolUses: [], handle: 'u0' },
      { role: 'assistant', text: '', toolUses: [{ tool_use_id: 'x', tool: 'Bash', input: {} }], handle: 'm-x' },
      { role: 'assistant', text: '', toolUses: [{ tool_use_id: 'y', tool: 'Bash', input: {} }], handle: 'm-y' },
      { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'x', text: '1' }], handle: 'r-x' },
    ];
    const out = resumeSafeMessages(messages);
    expect(out.map((m) => m.handle)).toEqual(['u0', 'm-x', 'm-y', undefined]);
    expect(out[1]).toBe(messages[1]);
  });

  it('keeps one parallel group together when compaction truncates only one of its results', async () => {
    const config = {
      ...resolveHookConfig({ preserveRecentMessages: 2, archiveResults: false }),
      apiKey: 'k',
    };
    // c's result is dropped; the rest are kept
    const { messages } = await compactSession(
      parallel(),
      config,
      jevFetch((key) => (key === 'result_t3' ? 0 : 0.9)),
    );
    const assistants = messages.filter((m) => m.role === 'assistant' && m.toolUses.length > 0);
    expect(assistants).toHaveLength(1);
    expect(assistants[0]?.toolUses.map((t) => t.tool_use_id)).toEqual(['a', 'b', 'c', 'd']);
    const results = messages.filter((m) => (m.toolResults?.length ?? 0) > 0);
    expect(results).toHaveLength(1);
    expect(results[0]?.toolResults?.[2]?.text).toContain('[jev-compaction truncated');
    expect(messages.filter((m) => m.toolUses.length + (m.toolResults?.length ?? 0) > 0).every((m) => m.handle === undefined)).toBe(true);
  });

  it('never merges a dropped call\'s note into the newest response (review finding 1)', async () => {
    const messages: SessionMessage[] = [
      { role: 'user', text: 'go', toolUses: [], handle: 'u0' },
      { role: 'assistant', text: '', toolUses: [{ tool_use_id: 'x', tool: 'Read', input: { file_path: 'x.md' }, text: BIG }], handle: 'm-x' },
      { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'x', text: BIG }], handle: 'r-x' },
      { role: 'assistant', text: '', toolUses: [{ tool_use_id: 'n1', tool: 'Bash', input: { command: 'a' } }], handle: 'm2-n1' },
      { role: 'assistant', text: '', toolUses: [{ tool_use_id: 'n2', tool: 'Bash', input: { command: 'b' } }], handle: 'm2-n2' },
      { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'n1', text: 'one' }], handle: 'r-n1' },
      { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'n2', text: 'two' }], handle: 'r-n2' },
    ];
    const config = {
      ...resolveHookConfig({ preserveRecentMessages: 2, archiveResults: false, minResultChars: 0, minKeepSignalCalls: 0 }),
      apiKey: 'k',
    };
    // x loses call and result; n1 and n2 are kept
    const { messages: out } = await compactSession(messages, config, jevFetch((key) => (key.endsWith('_t1') ? 0 : 0.9)));
    const note = out.find((m) => m.text.includes('[jev-compaction removed the Read call'));
    expect(note?.toolUses).toEqual([]);
    const newest = out.filter((m) => m.role === 'assistant' && m.toolUses.length > 0);
    expect(newest.map((m) => m.handle)).toEqual(['m2-n1', 'm2-n2']);
    expect(newest[0]).toBe(messages[3]);
  });

  it('merges the newest response whole when one of its results was truncated (re-review N1)', async () => {
    const big = 'x'.repeat(4000);
    const messages: SessionMessage[] = [
      { role: 'user', text: 'go', toolUses: [] },
      { role: 'assistant', text: '', toolUses: [{ tool_use_id: 'n1', tool: 'Read', input: { p: 1 }, text: big }], handle: 'm-n1' },
      { role: 'assistant', text: '', toolUses: [{ tool_use_id: 'n2', tool: 'Read', input: { p: 2 }, text: big }], handle: 'm-n2' },
      { role: 'assistant', text: '', toolUses: [{ tool_use_id: 'n3', tool: 'Read', input: { p: 3 }, text: big }], handle: 'm-n3' },
      { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'n1', text: big }], handle: 'r-n1' },
      { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'n2', text: big }], handle: 'r-n2' },
      { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'n3', text: big }], handle: 'r-n3' },
    ];
    const config = {
      ...resolveHookConfig({ preserveRecentMessages: 2, archiveResults: false, minResultChars: 0, minKeepSignalCalls: 0 }),
      apiKey: 'k',
    };
    const out = await compactSession(messages, config, jevFetch((name) => (name.startsWith('result_') ? 0 : 0.99)));
    const assistants = out.messages.filter((m) => m.role === 'assistant');
    expect(assistants).toHaveLength(1);
    expect(assistants[0]!.toolUses.map((t) => t.tool_use_id)).toEqual(['n1', 'n2', 'n3']);
    expect(assistants[0]!.handle).toBeUndefined();
  });

  it('merges the newest response whole when a call in the middle of it was dropped (re-review 2)', async () => {
    const big = 'x'.repeat(4000);
    const ids = ['n0', 'n1', 'n2', 'n3', 'n4'];
    const messages: SessionMessage[] = [
      { role: 'user', text: 'go', toolUses: [] },
      ...ids.map((id, i): SessionMessage => ({
        role: 'assistant',
        text: '',
        toolUses: [{ tool_use_id: id, tool: 'Read', input: { p: i }, text: big }],
        handle: `m-${id}`,
      })),
      ...ids.map((id): SessionMessage => ({
        role: 'user',
        text: '',
        toolUses: [],
        toolResults: [{ tool_use_id: id, text: big }],
        handle: `r-${id}`,
      })),
    ];
    const config = {
      ...resolveHookConfig({ preserveRecentMessages: 1, archiveResults: false, minResultChars: 0, minKeepSignalCalls: 0 }),
      apiKey: 'k',
    };
    // Drop the second call (t2) outright; keep the others whole.
    const out = await compactSession(messages, config, jevFetch((name) => (name.endsWith('_t2') ? 0 : 0.99)));
    expect(out.result.decisions.find((d) => d.id === 't2')?.action).toBe('drop_call');
    const assistants = out.messages.filter((m) => m.role === 'assistant');
    expect(assistants).toHaveLength(1);
    expect(assistants[0]!.toolUses.map((t) => t.tool_use_id)).toEqual(['n0', 'n2', 'n3', 'n4']);
    expect(assistants[0]!.handle).toBeUndefined();
  });

  it('keeps a result row with text of its own apart, so its results stay first', () => {
    const messages: SessionMessage[] = [
      { role: 'user', text: 'go', toolUses: [], handle: 'u0' },
      { role: 'assistant', text: '', toolUses: [{ tool_use_id: 'a', tool: 'Read', input: {} }], handle: 'm-a' },
      { role: 'assistant', text: '', toolUses: [{ tool_use_id: 'b', tool: 'Read', input: {} }], handle: 'm-b' },
      { role: 'user', text: '<system-reminder>x</system-reminder>', toolUses: [], toolResults: [{ tool_use_id: 'a', text: '1' }], handle: 'r-a' },
      { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'b', text: '2' }], handle: 'r-b' },
      { role: 'user', text: 'next', toolUses: [], handle: 'u2' },
      { role: 'assistant', text: 'ok', toolUses: [], handle: 'm3' },
    ];
    const out = resumeSafeMessages(messages);
    expect(out[1]?.toolUses.map((t) => t.tool_use_id)).toEqual(['a', 'b']);
    expect(out[2]).toMatchObject({ text: '<system-reminder>x</system-reminder>', toolResults: [{ tool_use_id: 'a' }] });
    expect(out[3]).toMatchObject({ text: '', toolResults: [{ tool_use_id: 'b' }] });
    expect(out.slice(1, 4).every((m) => m.handle === undefined)).toBe(true);
  });

  it('applies to the prose-trim rescue path too', async () => {
    const config = { ...resolveHookConfig({ preserveRecentMessages: 2, archiveResults: false }), apiKey: 'k' };
    expect(config.proseTrim).toBe(true);
    // everything kept: the tool cut is 0, so the prose rescue runs
    const out = await compactSession(parallel(), config, jevFetch(() => 0.9));
    expect(out.prose).toBeDefined();
    const withTools = out.messages.filter((m) => m.toolUses.length + (m.toolResults?.length ?? 0) > 0);
    expect(withTools.length).toBeGreaterThan(0);
    expect(withTools.every((m) => m.handle === undefined)).toBe(true);
  });

  it('leaves handles alone when stripHandles is off', async () => {
    const config = {
      ...resolveHookConfig({ preserveRecentMessages: 2, archiveResults: false, stripHandles: false }),
      apiKey: 'k',
    };
    const { messages } = await compactSession(parallel(), config, jevFetch(() => 0.9));
    expect(messages.map((m) => m.handle)).toEqual(parallel().map((m) => m.handle));
  });
});

describe('small-result floor', () => {
  it('keeps results under minResultChars without asking Jev', async () => {
    const asked: string[] = [];
    const messages = [{ role: 'user' as const, text: 'go', toolUses: [] }, ...call('s', 'short'), ...call('b', BIG), ...tail];
    const result = await compact(messages, asker(() => 0, asked), { preserveRecentMessages: 2, minKeepSignalCalls: 0 });
    expect(asked).toEqual(['call_t2', 'result_t2']);
    expect(result.decisions[0]).toMatchObject({ reason: 'small', action: 'keep' });
    expect(result.stats.small).toBe(1);
    expect(result.messages[1]).toBe(messages[1]);
  });

  it('asks about every result when the floor is 0', async () => {
    const asked: string[] = [];
    const messages = [{ role: 'user' as const, text: 'go', toolUses: [] }, ...call('s', 'short'), ...tail];
    await compact(messages, asker(() => 0.9, asked), { preserveRecentMessages: 2, minResultChars: 0 });
    expect(asked).toEqual(['call_t1', 'result_t1']);
  });
});

describe('already-compacted results', () => {
  it('never asks about or re-archives a result an earlier compaction cut', async () => {
    const cut = `${BIG.slice(0, 300)}\n${COMPACTED_MARKER}9000 chars of this tool result]\n${BIG}`;
    const asked: string[] = [];
    const archived: string[] = [];
    const messages = [{ role: 'user' as const, text: 'go', toolUses: [] }, ...call('old', cut), ...tail];
    const result = await compact(messages, asker(() => 0, asked), {
      preserveRecentMessages: 2,
      archive: (id) => (archived.push(id), `/a/${id}.txt`),
    });
    expect(asked).toEqual([]);
    expect(archived).toEqual([]);
    expect(result.decisions[0]).toMatchObject({ reason: 'compacted', action: 'keep' });
    expect(result.stats.compacted).toBe(1);
  });
});

describe('exact repeats', () => {
  it('truncates the earlier copy of an identical call without asking, and asks about the latest', async () => {
    const asked: string[] = [];
    const first = call('r1', BIG, { file_path: 'same.ts' });
    const second = call('r2', `${BIG}changed`, { file_path: 'same.ts' });
    const messages = [{ role: 'user' as const, text: 'go', toolUses: [] }, ...first, ...second, ...tail];
    const result = await compact(messages, asker(() => 0.9, asked), { preserveRecentMessages: 2 });
    expect(asked).toEqual(['call_t2', 'result_t2']);
    expect(result.decisions[0]).toMatchObject({ reason: 'duplicate', action: 'drop_result' });
    expect(result.decisions[1]).toMatchObject({ reason: 'kept' });
    expect(result.messages[2]?.toolResults?.[0]?.text).toContain('[jev-compaction truncated');
    expect(result.stats.duplicates).toBe(1);
  });

  it('leaves identical calls alone when dedupeRepeats is off', async () => {
    const asked: string[] = [];
    const messages = [
      { role: 'user' as const, text: 'go', toolUses: [] },
      ...call('r1', BIG, { file_path: 'same.ts' }),
      ...call('r2', BIG, { file_path: 'same.ts' }),
      ...tail,
    ];
    await compact(messages, asker(() => 0.9, asked), { preserveRecentMessages: 2, dedupeRepeats: false });
    expect(asked).toHaveLength(4);
  });

  const reminder = `<system-reminder>${'Remember the rules. '.repeat(20)}</system-reminder>`;

  it('replaces earlier copies of a repeated injected block and keeps the latest', () => {
    const messages: SessionMessage[] = [
      { role: 'user', text: 'start', toolUses: [], handle: 'u0' },
      { role: 'user', text: `first ${reminder}`, toolUses: [] },
      { role: 'assistant', text: 'ok', toolUses: [] },
      { role: 'user', text: `second ${reminder}`, toolUses: [], handle: 'u3' },
      { role: 'assistant', text: 'ok', toolUses: [] },
    ];
    const { messages: out, removed } = dedupeBlocks(messages, 2);
    expect(removed).toBe(1);
    expect(out[1]?.text).toMatch(/^first \[jev-compaction removed a repeated <system-reminder> block \(\d+ chars\); a later copy is kept\]$/);
    // a changed message carries no engine identity
    expect((out[1] as SessionMessage).handle).toBeUndefined();
    expect(out[3]).toBe(messages[3]);
    expect(out[2]).toBe(messages[2]);
  });

  it('replaces an earlier copy of a long repeated user message, such as a skill body', () => {
    const skill = `Base directory for this skill: /x\n${'Step. '.repeat(120)}`;
    const messages: Message[] = [
      { role: 'user', text: 'start', toolUses: [] },
      { role: 'user', text: skill, toolUses: [] },
      { role: 'assistant', text: 'ok', toolUses: [] },
      { role: 'user', text: skill, toolUses: [] },
      { role: 'assistant', text: 'ok', toolUses: [] },
      { role: 'user', text: 'more', toolUses: [] },
    ];
    const { messages: out, removed } = dedupeBlocks(messages, 0);
    expect(removed).toBe(1);
    expect(out[1]?.text).toBe(`[jev-compaction removed a repeated message (${skill.length} chars); a later copy is kept]`);
    expect(out[3]).toBe(messages[3]);
  });

  it('keeps the tool results of a rebuilt result row, and never rewrites an engine-held user prompt', () => {
    const messages: SessionMessage[] = [
      { role: 'user', text: 'start', toolUses: [], handle: 'u0' },
      { role: 'user', text: `prompt ${reminder}`, toolUses: [], handle: 'u1' },
      { role: 'user', text: reminder, toolUses: [], toolResults: [{ tool_use_id: 'a', text: 'r' }], handle: 'r-a' },
      { role: 'assistant', text: 'ok', toolUses: [] },
      { role: 'user', text: `again ${reminder}`, toolUses: [], handle: 'u3' },
      { role: 'assistant', text: 'ok', toolUses: [] },
    ];
    const { messages: out, removed } = dedupeBlocks(messages, 2);
    expect(removed).toBe(1);
    // the prompt may hold images the hook cannot see: left as is
    expect(out[1]).toBe(messages[1]);
    expect(out[2]?.text).toMatch(/^\[jev-compaction removed a repeated <system-reminder> block/);
    expect(out[2]?.toolResults).toEqual([{ tool_use_id: 'a', text: 'r' }]);
  });

  it('leaves short blocks, single copies and assistant text alone', () => {
    const short = '<system-reminder>be brief</system-reminder>';
    const messages: Message[] = [
      { role: 'user', text: 'start', toolUses: [] },
      { role: 'user', text: short, toolUses: [] },
      { role: 'assistant', text: reminder, toolUses: [] },
      { role: 'user', text: short, toolUses: [] },
      { role: 'assistant', text: reminder, toolUses: [] },
    ];
    const { messages: out, removed } = dedupeBlocks(messages, 0);
    expect(removed).toBe(0);
    out.forEach((message, index) => expect(message).toBe(messages[index]));
  });
});

describe('keep-signal guard', () => {
  const three = () => [
    { role: 'user' as const, text: 'go', toolUses: [] },
    ...call('a', BIG),
    ...call('b', `${BIG}b`),
    ...call('c', `${BIG}c`),
    ...tail,
  ];

  it('treats answers that drop every call as a failure', async () => {
    await expect(compact(three(), asker(() => 0), { preserveRecentMessages: 2 })).rejects.toThrow(
      /no keep signal: Jev dropped all 3 calls it answered/,
    );
  });

  it('accepts answers that keep at least the calls', async () => {
    const result = await compact(
      three(),
      asker((key) => (key.startsWith('call_') ? 0.9 : 0)),
      { preserveRecentMessages: 2 },
    );
    expect(result.stats.resultsDropped).toBe(3);
  });

  it('does not fire under minKeepSignalCalls answered calls, or when disabled', async () => {
    const two = [{ role: 'user' as const, text: 'go', toolUses: [] }, ...call('a', BIG), ...call('b', `${BIG}b`), ...tail];
    await expect(compact(two, asker(() => 0), { preserveRecentMessages: 2 })).resolves.toBeDefined();
    await expect(
      compact(three(), asker(() => 0), { preserveRecentMessages: 2, minKeepSignalCalls: 0 }),
    ).resolves.toBeDefined();
  });

  it('makes the hook fall back instead of installing an all-dropped history', async () => {
    const config = { ...resolveHookConfig({ preserveRecentMessages: 2, archiveResults: false }), apiKey: 'k' };
    await expect(compactSession(three(), config, jevFetch(() => 0))).rejects.toThrow(/no keep signal/);
  });
});
