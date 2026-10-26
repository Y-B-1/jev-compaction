import { describe, expect, it } from 'vitest';
import {
  compactSession,
  contextBreakdown,
  formatBreakdown,
  hybridSplit,
  messageTokens,
  register,
  resolveHookConfig,
} from '../hooks/fast-jev.ts';
import { CAPPED_MARKER } from '../src/compact.js';
import { COMPACTED_MARKER } from '../src/dedupe.js';
import { elideBody } from '../src/prose.js';
import { estimateTokens } from '../src/state.js';
import { compact, messageChars, resolveOptions, type JevAsker, type Message } from '../src/index.js';

type SessionMessage = Message & { handle?: string };

function message(role: Message['role'], text: string, extra: Partial<SessionMessage> = {}): SessionMessage {
  return { role, text, toolUses: [], ...extra };
}
function call(id: string, tool: string, input: Record<string, unknown>, text: string): SessionMessage {
  return message('assistant', '', { toolUses: [{ tool_use_id: id, tool, input, text }] });
}
function result(id: string, text: string): SessionMessage {
  return message('user', '', { toolResults: [{ tool_use_id: id, text }] });
}
function jev(answer: number, call = answer): JevAsker {
  return {
    async ask(_state, questions) {
      return {
        answers: Object.fromEntries(Object.keys(questions).map((k) => [k, { noul: k.startsWith('call_') ? call : answer }])),
      };
    },
  };
}
function jevFetch(answer: (name: string) => number) {
  return async (_url: string, init?: { body?: string }) => {
    const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
    const answers = Object.fromEntries(Object.keys(questions).map((k) => [k, { type: 'noul', noul: answer(k) }]));
    return { status: 200, ok: true, text: JSON.stringify({ answers }) };
  };
}
function archive() {
  const files = new Map<string, string>();
  return { files, cite: (id: string, text: string) => (files.set(`arch/${id}.txt`, text), `arch/${id}.txt`) };
}

const BIG = 'line of real output with detail\n'.repeat(400); // 12,800 chars
const tail = [message('assistant', 'Working.'), message('user', 'continue'), message('assistant', 'ok'), message('user', 'go')];

const NOTICE = /^(fallback to built-in summary|skipped \(|kept \d+\/\d+ messages|summarized \d+ older|built-in summary failed|kept the cut)/;
describe('defaults', () => {
  it('keeps a shorter head and salvage, and caps kept results at 4000 chars', () => {
    const o = resolveOptions();
    expect(o.truncateHeadChars).toBe(150);
    expect(o.salvageChars).toBe(150);
    expect(o.maxKeptResultChars).toBe(4000);
    expect(resolveOptions({ maxKeptResultChars: -5 }).maxKeptResultChars).toBe(0);
  });
});

describe('shorter truncation note', () => {
  it('cites the archive path without the extra wording', async () => {
    const a = archive();
    const out = await compact(
      [message('user', 'Find the bug.'), call('t1', 'Read', { file_path: 'a' }, BIG), result('t1', BIG), ...tail],
      jev(0.01, 0.9),
      { archive: a.cite, preserveRecentMessages: 4 },
    );
    const text = out.messages.flatMap((m) => m.toolResults ?? [])[0]!.text;
    expect(text).toContain(`${COMPACTED_MARKER}${BIG.length - 150} chars; full output: arch/t1.txt]`);
    expect(text).not.toContain('Read or grep');
  });
});

describe('cap on kept results', () => {
  const session = () => [message('user', 'Find the bug.'), call('t1', 'Read', { file_path: 'a' }, BIG), result('t1', BIG), ...tail];

  it('cuts a result Jev keeps whole to its head and tail within maxKeptResultChars and archives it', async () => {
    const a = archive();
    const out = await compact(session(), jev(0.9), { archive: a.cite, preserveRecentMessages: 4 });
    expect(out.decisions[0]!.action).toBe('keep');
    expect(out.stats.resultsCapped).toBe(1);
    const text = out.messages.flatMap((m) => m.toolResults ?? [])[0]!.text;
    expect(text.startsWith(BIG.slice(0, 3000))).toBe(true);
    expect(text.endsWith(BIG.slice(-1000))).toBe(true);
    expect(text).toContain(`${CAPPED_MARKER}${BIG.length} chars, ${BIG.length - 4000} omitted; full output: arch/t1.txt]`);
    expect(text.length).toBeLessThan(4400);
    expect(a.files.get('arch/t1.txt')).toBe(BIG);
  });

  it('leaves a capped result alone when kept again, and cites the full archive when it is dropped later', async () => {
    const a = archive();
    const first = await compact(session(), jev(0.9), { archive: a.cite, preserveRecentMessages: 4 });
    const capped = first.messages.flatMap((m) => m.toolResults ?? [])[0]!.text;
    const again = await compact(first.messages, jev(0.9), { archive: a.cite, preserveRecentMessages: 4 });
    expect(again.messages.flatMap((m) => m.toolResults ?? [])[0]!.text).toBe(capped);
    expect(again.stats.resultsCapped).toBe(0);
    const dropped = await compact(first.messages, jev(0.01, 0.9), { archive: a.cite, preserveRecentMessages: 4 });
    const text = dropped.messages.flatMap((m) => m.toolResults ?? [])[0]!.text;
    expect(text).toContain('full output: arch/t1.txt]');
    expect(text.length).toBeLessThan(capped.length);
    expect(a.files.get('arch/t1.txt')).toBe(BIG);
  });

  it('keeps the full archive when a capped result is later dropped with its call', async () => {
    const a = archive();
    const first = await compact(session(), jev(0.9), { archive: a.cite, preserveRecentMessages: 4 });
    const dropped = await compact(first.messages, jev(0.01, 0.01), { archive: a.cite, preserveRecentMessages: 4, minKeepSignalCalls: 0 });
    expect(dropped.decisions[0]!.action).toBe('drop_call');
    expect(dropped.messages.map((m) => m.text).join('\n')).toContain('full output: arch/t1.txt]');
    expect(a.files.get('arch/t1.txt')).toBe(BIG);
  });

  it('never caps a result the cap would not shorten', async () => {
    const near = 'y'.repeat(4300);
    const a = archive();
    const out = await compact(
      [message('user', 'Find the bug.'), call('t1', 'Read', { file_path: 'a' }, near), result('t1', near), ...tail],
      jev(0.9),
      { archive: a.cite, preserveRecentMessages: 4 },
    );
    expect(out.stats.resultsCapped).toBe(0);
    expect(out.messages.flatMap((m) => m.toolResults ?? [])[0]!.text).toBe(near);
    expect(a.files.size).toBe(0);
  });

  it('leaves kept results whole when the cap is 0, and never caps the newest messages', async () => {
    const off = await compact(session(), jev(0.9), { preserveRecentMessages: 4, maxKeptResultChars: 0 });
    expect(off.stats.resultsCapped).toBe(0);
    expect(off.messages.flatMap((m) => m.toolResults ?? [])[0]!.text).toBe(BIG);
    const pinned = await compact(session(), jev(0.9), { preserveRecentMessages: 6 });
    expect(pinned.messages.flatMap((m) => m.toolResults ?? [])[0]!.text).toBe(BIG);
  });
});

describe('elided bodies are never cut twice', () => {
  it('returns an already elided body unchanged', () => {
    const once = elideBody(BIG, 'Write.content for a.ts', 150, 150, 'arch/x.txt');
    expect(once).not.toBe(BIG);
    expect(elideBody(once, 'Write.content for a.ts', 150, 150, 'arch/x.txt')).toBe(once);
  });
});

describe('inputTrim', () => {
  const body = 'const value = compute();\n'.repeat(200);
  const session = (): SessionMessage[] => [
    message('user', 'Fix the stale cache.'),
    call('w1', 'Write', { file_path: 'src/app.ts', content: body }, 'ok'),
    result('w1', 'ok'),
    call('t1', 'Read', { file_path: 'big' }, BIG),
    result('t1', BIG),
    call('w2', 'Write', { file_path: 'src/new.ts', content: body }, 'ok'),
    result('w2', 'ok'),
    message('user', 'go'),
  ];
  const config = (options: Record<string, unknown> = {}) => ({
    ...resolveHookConfig({ preserveRecentMessages: 3, proseTrim: false, notificationTrim: false, ...options }),
    apiKey: 'k',
  });
  const content = (messages: SessionMessage[], id: string) =>
    messages.flatMap((m) => m.toolUses).find((t) => t.tool_use_id === id)?.input['content'] as string;

  it('cuts old code bodies even when tool compaction alone reaches the minimum', async () => {
    const out = await compactSession(session(), config(), jevFetch(() => 0.01));
    expect(content(out.messages, 'w1')).toContain('[jev-compaction elided ');
    expect(content(out.messages, 'w1').length).toBeLessThan(600);
    expect(content(out.messages, 'w2')).toBe(body);
    expect(out.result.stats.inputsElided).toBe(1);
    expect(out.prose).toBeUndefined();
  });

  it('leaves code bodies whole when inputTrim is off', async () => {
    const out = await compactSession(session(), config({ inputTrim: false }), jevFetch(() => 0.01));
    expect(content(out.messages, 'w1')).toBe(body);
    expect(out.result.stats.inputsElided).toBe(0);
  });

  it('counts the cut toward the reduction', async () => {
    const on = await compactSession(session(), config(), jevFetch(() => 0.01));
    const off = await compactSession(session(), config({ inputTrim: false }), jevFetch(() => 0.01));
    expect(on.result.stats.charsAfter).toBeLessThan(off.result.stats.charsAfter - 3000);
  });

  it('measures charsAfter on the trimmed messages, quotes and newlines included', async () => {
    const quoted = 'const s = "a\tb";\n'.repeat(300);
    const messages = session().map((m) =>
      m.toolUses.some((t) => t.tool_use_id === 'w1')
        ? call('w1', 'Write', { file_path: 'src/app.ts', content: quoted }, 'ok')
        : m,
    );
    const out = await compactSession(messages, config(), jevFetch(() => 0.01));
    expect(out.result.stats.charsAfter).toBe(out.result.messages.reduce((sum, m) => sum + messageChars(m), 0));
  });
});

describe('context breakdown', () => {
  it('estimates tokens per kind of content the model sees', () => {
    const api = [
      { role: 'user', content: [{ type: 'text', text: 'a'.repeat(400) }] },
      {
        role: 'assistant',
        content: [
          { type: 'thinking', thinking: 'b'.repeat(40) },
          { type: 'text', text: 'c'.repeat(80) },
          { type: 'tool_use', id: 'x', name: 'Read', input: { file_path: 'd'.repeat(100) } },
        ],
      },
      {
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 'x', content: 'e'.repeat(800) },
          { type: 'tool_result', tool_use_id: 'y', content: [{ type: 'text', text: 'f'.repeat(400) }] },
          { type: 'image', source: { type: 'base64', data: 'g'.repeat(10_000) } },
        ],
      },
    ];
    const b = contextBreakdown(api);
    expect(b).toEqual({
      userText: estimateTokens('a'.repeat(400)),
      assistantText: estimateTokens('c'.repeat(80)),
      thinking: estimateTokens('b'.repeat(40)),
      toolInputs: estimateTokens(JSON.stringify({ file_path: 'd'.repeat(100) })),
      toolResults: estimateTokens('e'.repeat(800)) + estimateTokens('f'.repeat(400)),
      images: 1,
    });
    expect(b.images).toBe(1);
    expect(formatBreakdown(b)).toMatch(/^user text ~\d+, assistant text ~\d+, thinking ~\d+, tool inputs ~\d+, tool results ~\d+, images 1$/);
  });
});

describe('hybrid rescue: built-in summary of the older part only', () => {
  const MID = 'kept output line\n'.repeat(200); // 3,400 chars: under the kept-result cap
  const turn = (n: number): SessionMessage[] => [
    message('user', `Step ${n}: keep going on the cache work.`),
    call(`r${n}`, 'Read', { file_path: `f${n}` }, MID),
    result(`r${n}`, MID),
    message('assistant', `Done with step ${n}.`),
  ];
  const transcript = (): SessionMessage[] => [1, 2, 3, 4, 5, 6, 7, 8].flatMap(turn);
  const SUMMARY = message('user', 'Summary of the earlier work.', { handle: 'summary' });
  const setup = (options: Record<string, unknown> = {}, trigger = 'auto') => {
    const hooks: Record<string, (...args: any[]) => Promise<any>> = {};
    (register as any)(
      (event: string, a: unknown, b?: any) => {
        hooks[event] = typeof b === 'function' ? b : (a as any);
      },
      { apiKey: 'k', preserveRecentMessages: 2, archiveResults: false, ...options },
    );
    const toasts: string[] = [];
    const notices: string[] = [];
    const calls: SessionMessage[][] = [];
    const $ = {
      // A sleep that only ends on abort: the request timeout never fires first.
      clock: {
        sleep: (_ms: number, o?: { signal?: AbortSignal }) =>
          new Promise<void>((_r, reject) => o?.signal?.addEventListener('abort', () => reject(o.signal?.reason), { once: true })),
      },
      http: { fetch: async (url: string, init?: { body?: string }) => jevFetch(() => 0.99)(url, init) },
      ui: { log: (t: string) => void (NOTICE.test(t) && notices.push(t)), toast: (t: string) => toasts.push(t) },
    };
    const next = Object.assign(
      async (e: { messages: SessionMessage[] }) => {
        calls.push(e.messages);
        return { messages: [SUMMARY] };
      },
      { signal: new AbortController().signal, budget: { ms: 10_000, remainingMs: 10_000 } },
    );
    return { run: () => hooks['session.compact']!($, { trigger, messages: transcript() }, next), toasts, notices, calls };
  };

  it('summarizes only the older turns and keeps the recent ones after the summary', async () => {
    const on = setup();
    const out = await on.run();
    expect(on.calls).toHaveLength(1);
    const older = on.calls[0]!;
    expect(older.length).toBeGreaterThan(0);
    expect(older.length).toBeLessThan(transcript().length);
    expect(out.messages[0]).toBe(SUMMARY);
    const recent = out.messages.slice(1) as SessionMessage[];
    expect(recent[0]!.role).toBe('user');
    expect(recent[0]!.text).toMatch(/^Step \d: /);
    expect(recent.at(-1)!.text).toBe('Done with step 8.');
    expect(on.notices[0]).toMatch(/^summarized 24 older messages, kept 8 recent \(trim alone below 25% minimum, prose trim tried: \d+% reduction; /);
  });

  it('splits only at a user turn, so no tool call is separated from its result', async () => {
    const on = setup();
    await on.run();
    const older = on.calls[0]!;
    const ids = (ms: SessionMessage[], f: (m: SessionMessage) => string[]) => new Set(ms.flatMap(f));
    const uses = ids(older, (m) => m.toolUses.map((t) => t.tool_use_id));
    const results = ids(older, (m) => (m.toolResults ?? []).map((r) => r.tool_use_id));
    expect([...uses].sort()).toEqual([...results].sort());
  });

  it('falls back to the full built-in summary when hybridSummary is off', async () => {
    const on = setup({ hybridSummary: false });
    await on.run();
    expect(on.calls[0]).toHaveLength(transcript().length);
    expect(on.notices[0]).toMatch(/^fallback to built-in summary \(below 25% minimum/);
  });

  it('never summarizes when the built-in fallback is not allowed for the trigger', async () => {
    const on = setup({ builtinFallback: 'never' });
    const out = await on.run();
    expect(on.calls).toHaveLength(0);
    expect(out.skip).toMatch(/builtin fallback disabled/);
  });
});

describe('hybridSplit', () => {
  const turn = (n: number, size: number): SessionMessage[] => [
    message('user', `turn ${n}`),
    call(`c${n}`, 'Read', { file_path: 'x' }, 'y'.repeat(size)),
    result(`c${n}`, 'y'.repeat(size)),
  ];
  it('returns the earliest user turn whose tail fits the share', () => {
    const ms = [...turn(1, 4000), ...turn(2, 4000), ...turn(3, 1000), ...turn(4, 1000)];
    expect(hybridSplit(ms, messageTokens(ms), 0.3)).toBe(6);
  });
  it('returns 0 when no user turn leaves a tail that fits', () => {
    const ms = [...turn(1, 100), ...turn(2, 9000)];
    expect(hybridSplit(ms, messageTokens(ms), 0.3)).toBe(0);
  });
  it('never splits at a host notice inside an open parallel tool run', () => {
    const big = 'x y z '.repeat(3000);
    const ms: SessionMessage[] = [
      message('user', 'start'),
      message('assistant', `big work ${big}`),
      message('user', 'go'),
      { ...call('a', 'Read', {}, ''), toolUses: [
        { tool_use_id: 'a', tool: 'Read', input: {} },
        { tool_use_id: 'b', tool: 'Read', input: {} },
      ] } as SessionMessage,
      result('a', big),
      message('user', '<system-reminder>file changed</system-reminder>'),
      result('b', 'rb'),
      message('assistant', 'done'),
    ];
    const split = hybridSplit(ms, messageTokens(ms), 0.3);
    expect(split).not.toBe(5);
    const tail = ms.slice(split);
    const orphans = tail
      .flatMap((m) => m.toolResults ?? [])
      .filter((r) => !tail.some((m) => m.toolUses.some((u) => u.tool_use_id === r.tool_use_id)));
    if (split > 0) expect(orphans).toEqual([]);
  });
  it('never returns 0 as a split of the first message', () => {
    const ms = turn(1, 100);
    expect(hybridSplit(ms, messageTokens(ms), 0.99)).toBe(0);
  });
});
