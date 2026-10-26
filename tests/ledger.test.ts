import { describe, expect, it } from 'vitest';
import { archiveFilesNamed, register } from '../hooks/fast-jev.js';

type Hook = (...args: any[]) => Promise<any>;

const setup = (options: Record<string, unknown> = {}) => {
  const hooks: { event: string; matcher?: { tool?: string }; hook: Hook }[] = [];
  (register as any)(
    (event: string, a: unknown, b?: Hook) =>
      hooks.push(typeof b === 'function' ? { event, matcher: a as { tool?: string }, hook: b } : { event, hook: a as Hook }),
    { apiKey: 'k', preserveRecentMessages: 2, minReductionRatio: 0, ...options },
  );
  const files = new Map<string, string>();
  const $ = {
    clock: { sleep: () => new Promise(() => {}), now: async () => Date.UTC(2026, 9, 21) },
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
    session: { cwd: async () => '/w', root: async () => '/w' },
    process: { run: async () => ({ exitCode: 1, stdout: '' }) },
  };
  const next = (result: unknown = undefined) =>
    Object.assign(async (e: unknown) => result ?? e, { signal: new AbortController().signal, budget: { ms: 1e4, remainingMs: 1e4 } });
  const hook = (event: string, tool?: string) => hooks.find((h) => h.event === event && h.matcher?.tool === tool)!.hook;
  return { $, files, next, hook };
};

const big = 'output line\n'.repeat(300);
const transcript = () => [
  { role: 'user', text: 'Fix it.', toolUses: [] },
  { role: 'assistant', text: '', toolUses: [{ tool_use_id: 'toolu_A', tool: 'Bash', input: { command: 'make' }, text: big }] },
  { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'toolu_A', text: big, isError: false }] },
  { role: 'assistant', text: 'ok', toolUses: [] },
  { role: 'user', text: 'go', toolUses: [] },
];
const lines = (files: Map<string, string>) =>
  (files.get('/w/.claude/jev-compaction/log.jsonl') ?? '').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));

describe('compaction ledger', () => {
  it('appends one line per compaction with each call’s tool, scores, decision and archive file', async () => {
    const { $, files, next, hook } = setup();
    await hook('session.compact')($, { trigger: 'manual', messages: transcript() }, next());
    const [entry] = lines(files);
    expect(entry).toMatchObject({ trigger: 'manual', subagent: false, model: 'jev-latest', messagesBefore: 5 });
    expect(entry.outcome).toMatch(/^Compacted with Jev/);
    expect(entry.calls).toEqual([['Bash', 'drop_result', 'result_dropped', 0.9, 0.01, 'toolu_A.txt']]);
    expect(files.has('/w/.claude/jev-compaction/toolu_A.txt')).toBe(true);
  });

  it('notes an archive file read back with Read, Grep or Bash, and nothing else', async () => {
    const { $, files, next, hook } = setup();
    await hook('tool.call', 'Read')($, { tool: 'Read', file_path: '/w/.claude/jev-compaction/toolu_A.txt' }, next({}));
    await hook('tool.call', 'Grep')($, { tool: 'Grep', pattern: 'error', path: '/w/.claude/jev-compaction' }, next({}));
    await hook('tool.call', 'Bash')($, { tool: 'Bash', command: 'cat .claude/jev-compaction/toolu_B.txt' }, next({}));
    await hook('tool.call', 'Read')($, { tool: 'Read', file_path: '/w/src/app.ts' }, next({}));
    expect(lines(files).map((l) => [l.via, l.readBack])).toEqual([
      ['Read', ['toolu_A.txt']],
      ['Grep', ['*']],
      ['Bash', ['toolu_B.txt']],
    ]);
  });

  it('writes nothing when compactionLog is off', async () => {
    const { $, files, next, hook } = setup({ compactionLog: false });
    await hook('session.compact')($, { trigger: 'manual', messages: transcript() }, next());
    await hook('tool.call', 'Read')($, { tool: 'Read', file_path: '/w/.claude/jev-compaction/toolu_A.txt' }, next({}));
    expect(files.has('/w/.claude/jev-compaction/log.jsonl')).toBe(false);
  });

  it('finds archive files named in a tool input', () => {
    expect(archiveFilesNamed({ command: 'grep -n x .claude/jev-compaction/a.txt .claude/jev-compaction/b.txt' })).toEqual(['a.txt', 'b.txt']);
    expect(archiveFilesNamed({ file_path: 'src/a.txt' })).toEqual([]);
  });
});

describe('what a compaction reports', () => {
  it('logs the marker rows left and each stage outcome', async () => {
    const { $, files, next, hook } = setup();
    await hook('session.compact')($, { trigger: 'manual', messages: transcript() }, next());
    const [entry] = lines(files);
    expect(entry.markersAfter).toBe(0);
    expect(entry.stages).toEqual(expect.objectContaining({ tool: expect.any(Number) }));
  });

  it('states the share of the whole context freed when the engine reports the context size', async () => {
    const { $, next, hook } = setup();
    const toasts: string[] = [];
    Object.assign($.ui, { toast: (t: string) => void toasts.push(t) });
    Object.assign($.session, { usage: async () => ({ context: { tokens: 100_000, breakdown: { categories: [{ kind: 'used', name: 'Messages', tokens: 50_000 }] } } }) });
    await hook('session.compact')($, { trigger: 'manual', messages: transcript() }, next());
    expect(toasts[0]).toMatch(/^Compacted with Jev: 5 → \d+ messages, frees \d+% of the context, no summary$/);
  });
});
