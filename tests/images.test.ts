import { describe, expect, it } from 'vitest';
import { originalIndex, register } from '../hooks/fast-jev.js';
import { applyDecisions } from '../src/compact.js';
import { collectToolCalls } from '../src/state.js';

type Hook = (...args: any[]) => Promise<any>;

const DIR = '/w/.claude/jev-compaction';

/** The engine as the compact hook reads it, with the conversation's Messages API form. */
const setup = (api: unknown[], options: Record<string, unknown> = {}) => {
  const hooks: { event: string; matcher?: { command?: string }; hook: Hook }[] = [];
  (register as any)(
    (event: string, a: unknown, b?: Hook) =>
      hooks.push(typeof b === 'function' ? { event, matcher: a as { command?: string }, hook: b } : { event, hook: a as Hook }),
    { apiKey: 'k', preserveRecentMessages: 2, minReductionRatio: 0, ...options },
  );
  const files = new Map<string, string>();
  const toasts: string[] = [];
  const $ = {
    clock: { sleep: () => new Promise(() => {}), now: async () => Date.UTC(2026, 9, 22) },
    http: {
      fetch: async (_url: string, init: { body: string }) => {
        const { questions } = JSON.parse(init.body) as { questions: Record<string, unknown> };
        const answers = Object.fromEntries(Object.keys(questions).map((k) => [k, { type: 'noul', noul: 0.9 }]));
        return { status: 200, ok: true, text: JSON.stringify({ answers }) };
      },
    },
    ui: { log: () => {}, toast: (text: string) => void toasts.push(text) },
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
      messages: async (query?: { as?: string }) => {
        if (query?.as !== 'api') throw new Error('api form only');
        return api;
      },
    },
    process: { run: async () => ({ exitCode: 1, stdout: '' }) },
    env: { get: async () => undefined },
  };
  const next = () =>
    Object.assign(async (e: unknown) => e, { signal: new AbortController().signal, budget: { ms: 1e4, remainingMs: 1e4 } });
  const hook = (event: string) => hooks.find((h) => h.event === event)!.hook;
  return { $, files, toasts, next, hook };
};

const png = (data: string) => ({ type: 'image', source: { type: 'base64', media_type: 'image/png', data } });
const pdf = (data: string) => ({ type: 'document', source: { type: 'base64', media_type: 'application/pdf', data } });

const transcript = () => [
  { role: 'user', text: 'Why does login fail?\n[Image: source: /tmp/s/images/1.png]', toolUses: [], handle: 'u0' },
  { role: 'assistant', text: 'It shows a login form with a red error banner. I will read the chart.', toolUses: [], handle: 'a1' },
  { role: 'assistant', text: '', toolUses: [{ tool_use_id: 'toolu_IMG', tool: 'Read', input: { file_path: '/w/pic.png' }, text: '' }], handle: 'a2' },
  { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'toolu_IMG', text: '', isError: false }], handle: 'r3' },
  { role: 'assistant', text: 'The chart shows sales rising in March. Now the spec.', toolUses: [], handle: 'a4' },
  { role: 'assistant', text: '', toolUses: [{ tool_use_id: 'toolu_DOC', tool: 'Read', input: { file_path: '/w/spec.pdf' }, text: '' }], handle: 'a5' },
  { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'toolu_DOC', text: '', isError: false }], handle: 'r6' },
  { role: 'assistant', text: 'The spec asks for two-factor login.', toolUses: [], handle: 'a7' },
  { role: 'user', text: 'And this one?\n[Image: source: /tmp/s/images/2.png]', toolUses: [], handle: 'u8' },
];

const api = () => [
  { role: 'user', content: [{ type: 'text', text: 'Why does login fail?' }, png('UFJPTVBU'), { type: 'text', text: '[Image: source: /tmp/s/images/1.png]' }] },
  { role: 'assistant', content: [{ type: 'text', text: 'It shows a login form with a red error banner. I will read the chart.' }, { type: 'tool_use', id: 'toolu_IMG', name: 'Read', input: { file_path: '/w/pic.png' } }] },
  { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'toolu_IMG', content: [png('Q0hBUlQ=')] }] },
  { role: 'assistant', content: [{ type: 'text', text: 'The chart shows sales rising in March. Now the spec.' }, { type: 'tool_use', id: 'toolu_DOC', name: 'Read', input: { file_path: '/w/spec.pdf' } }] },
  { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'toolu_DOC', content: [pdf('U1BFQw==')] }] },
  { role: 'assistant', content: [{ type: 'text', text: 'The spec asks for two-factor login.' }] },
  { role: 'user', content: [{ type: 'text', text: 'And this one?' }, png('Q1VSUkVOVA=='), { type: 'text', text: '[Image: source: /tmp/s/images/2.png]' }] },
];

type Row = { role: string; text: string; handle?: string; toolResults?: { tool_use_id: string; text: string }[] };
const resultText = (rows: Row[], id: string) =>
  rows.flatMap((r) => r.toolResults ?? []).find((r) => r.tool_use_id === id)?.text ?? '';
const archived = (files: Map<string, string>, data: string) =>
  [...files].find(([path, text]) => path.startsWith(`${DIR}/`) && text === data)?.[0];

describe('images and documents at compaction', () => {
  it('puts a placeholder with the path, the description and the archived copy in place of a tool-result image', async () => {
    const { $, files, next, hook } = setup(api());
    const out = (await hook('session.compact')($, { trigger: 'manual', messages: transcript() }, next())) as { messages: Row[] };
    const copy = archived(files, 'Q0hBUlQ=');
    expect(copy).toMatch(/\/img-[0-9a-f]+\.png\.b64$/);
    const text = resultText(out.messages, 'toolu_IMG');
    expect(text).toMatch(/^\[image removed by compaction/);
    expect(text).toContain('Read /w/pic.png');
    expect(text).toContain('The chart shows sales rising in March.');
    expect(text).toContain(copy);
    expect(text).toContain('archived copy (base64): ');
  });

  it('names a removed PDF as a document', async () => {
    const { $, files, next, hook } = setup(api());
    const out = (await hook('session.compact')($, { trigger: 'manual', messages: transcript() }, next())) as { messages: Row[] };
    const text = resultText(out.messages, 'toolu_DOC');
    expect(text).toMatch(/^\[document removed by compaction/);
    expect(text).toContain('Read /w/spec.pdf');
    expect(text).toContain('The spec asks for two-factor login.');
    expect(archived(files, 'U1BFQw==')).toMatch(/\.pdf\.b64$/);
  });

  it('removes an image pasted in an earlier prompt and keeps the one in the current prompt', async () => {
    const { $, files, next, hook } = setup(api());
    const out = (await hook('session.compact')($, { trigger: 'manual', messages: transcript() }, next())) as { messages: Row[] };
    const older = out.messages.find((r) => r.role === 'user' && r.text.startsWith('Why does login fail?'))!;
    expect(older.handle).toBeUndefined();
    expect(older.text).toContain('[image removed by compaction');
    expect(older.text).toContain('Read /tmp/s/images/1.png');
    expect(older.text).toContain('It shows a login form with a red error banner.');
    expect(archived(files, 'UFJPTVBU')).toBeDefined();
    const current = out.messages.at(-1)!;
    expect(current).toMatchObject({ handle: 'u8', text: 'And this one?\n[Image: source: /tmp/s/images/2.png]' });
    expect(archived(files, 'Q1VSUkVOVA==')).toBeUndefined();
  });

  it('gives a pasted image only to the prompt it came with, not to an earlier prompt with shorter text', async () => {
    const rows = [
      { role: 'user', text: 'ok', toolUses: [], handle: 'u0' },
      { role: 'assistant', text: 'Waiting.', toolUses: [], handle: 'a1' },
      { role: 'user', text: 'ok, see this', toolUses: [], handle: 'u2' },
      { role: 'assistant', text: 'A bar chart.', toolUses: [], handle: 'a3' },
      { role: 'user', text: 'thanks', toolUses: [], handle: 'u4' },
    ];
    const form = [
      { role: 'user', content: [{ type: 'text', text: 'ok' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'Waiting.' }] },
      { role: 'user', content: [{ type: 'text', text: 'ok, see this' }, png('QkFS')] },
      { role: 'assistant', content: [{ type: 'text', text: 'A bar chart.' }] },
      { role: 'user', content: [{ type: 'text', text: 'thanks' }] },
    ];
    const { $, next, hook } = setup(form);
    const out = (await hook('session.compact')($, { trigger: 'manual', messages: rows }, next())) as { messages: Row[] };
    expect(out.messages.find((r) => r.text === 'ok')).toMatchObject({ handle: 'u0' });
    const pasted = out.messages.find((r) => r.text.startsWith('ok, see this'))!;
    expect(pasted.handle).toBeUndefined();
    expect(pasted.text).toContain('described then as: "A bar chart."');
  });

  it('lists each archived copy in INDEX.md', async () => {
    const { $, files, next, hook } = setup(api());
    await hook('session.compact')($, { trigger: 'manual', messages: transcript() }, next());
    const index = files.get(`${DIR}/INDEX.md`) ?? '';
    const name = archived(files, 'Q0hBUlQ=')!.slice(DIR.length + 1);
    expect(index).toMatch(new RegExp(`- ${name.replace('.', '\\.')}: .*pic\\.png`));
    expect(index).toContain('The chart shows sales rising in March.');
  });

  it('counts the removed images and documents in the toast and in /jev', async () => {
    const { $, toasts, next, hook } = setup(api());
    await hook('session.compact')($, { trigger: 'manual', messages: transcript() }, next());
    expect(toasts.at(-1)).toMatch(/^Compacted with Jev: .*, 2 images and 1 document removed \(paths kept\)$/);
    const jev = (await hook('command.run')($, {}, next())) as { text: string };
    expect(jev.text).toContain('2 images and 1 document removed (paths kept)');
  });
});

describe('a placeholder survives a later cut of its result', () => {
  const note = `[image removed by compaction · shot.png (image/png) · described then as: "A login form with a red error banner." · view again: Read /very/long/project/path/${'d'.repeat(120)}/shot.png · archived copy: /w/.claude/jev-compaction/img-0123456789ab.png.b64]`;
  const tool = { tool_use_id: 'x', tool: 'Read', input: { file_path: '/w/shot.png' }, text: '' };
  const messages = [
    { role: 'user' as const, text: 's', toolUses: [] },
    { role: 'assistant' as const, text: '', toolUses: [tool] },
    { role: 'user' as const, text: '', toolUses: [], toolResults: [{ tool_use_id: 'x', text: `${'log line\n'.repeat(400)}${note}` }] },
    { role: 'user' as const, text: 'e', toolUses: [] },
  ];
  it.each(['drop_result', 'cap'] as const)('when Jev decides %s', (action) => {
    const decision =
      action === 'cap'
        ? { id: 't1', tool: 'Read', action: 'keep', reason: 'kept', keepCall: 1, keepResult: 1 }
        : { id: 't1', tool: 'Read', action: 'drop_result', reason: 'result_dropped', keepCall: 1, keepResult: 0 };
    const out = applyDecisions(messages, [decision] as never, collectToolCalls(messages, 0), 150, undefined, 0, 1000);
    const text = out.flatMap((m) => m.toolResults ?? []).find((r) => r.tool_use_id === 'x')!.text;
    expect(text.length).toBeLessThan(4000 + note.length);
    expect(text).toContain(note);
  });
});

describe('review notes on the placeholders', () => {
  it('gives a pasted image to the prompt it came with when an earlier prompt has the same text', async () => {
    const rows = [
      { role: 'user', text: 'ok', toolUses: [], handle: 'u0' },
      { role: 'assistant', text: 'Waiting.', toolUses: [], handle: 'a1' },
      { role: 'user', text: 'ok\n[Image: source: /p/1.png]', toolUses: [], handle: 'u2' },
      { role: 'assistant', text: 'A bar chart.', toolUses: [], handle: 'a3' },
      { role: 'user', text: 'thanks', toolUses: [], handle: 'u4' },
    ];
    const form = [
      { role: 'user', content: [{ type: 'text', text: 'ok' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'Waiting.' }] },
      { role: 'user', content: [{ type: 'text', text: 'ok' }, png('QkFS'), { type: 'text', text: '[Image: source: /p/1.png]' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'A bar chart.' }] },
      { role: 'user', content: [{ type: 'text', text: 'thanks' }] },
    ];
    const { $, next, hook } = setup(form);
    const out = (await hook('session.compact')($, { trigger: 'manual', messages: rows }, next())) as { messages: Row[] };
    expect(out.messages.find((r) => r.text === 'ok')).toMatchObject({ handle: 'u0' });
    const pasted = out.messages.find((r) => r.text.startsWith('ok\n[Image'))!;
    expect(pasted.handle).toBeUndefined();
    expect(pasted.text).toContain('described then as: "A bar chart."');
  });

  it('keeps the placeholder in the note of a dropped call', () => {
    const note = '[image removed by compaction · a.png (image/png) · described then as: "Red." · view again: Read /w/a.png]';
    const tool = { tool_use_id: 'x', tool: 'Read', input: { file_path: '/w/a.png' }, text: '' };
    const messages = [
      { role: 'user' as const, text: 's', toolUses: [] },
      { role: 'assistant' as const, text: '', toolUses: [tool] },
      { role: 'user' as const, text: '', toolUses: [], toolResults: [{ tool_use_id: 'x', text: note }] },
      { role: 'user' as const, text: 'e', toolUses: [] },
    ];
    const decision = { id: 't1', tool: 'Read', action: 'drop_call', reason: 'call_dropped', keepCall: 0, keepResult: 0 };
    const out = applyDecisions(messages, [decision] as never, collectToolCalls(messages, 0), 150);
    expect(out.map((m) => m.text).join('\n')).toContain(note);
  });

  it('names a tool-result image without a file path as a tool result, not a paste', async () => {
    const rows = transcript().map((r) => (r.handle === 'a2' ? { ...r, toolUses: [{ ...r.toolUses[0]!, input: {} }] } : r));
    const form = api().map((m) =>
      m.role === 'assistant' && m.content.some((b: any) => b.id === 'toolu_IMG')
        ? { ...m, content: m.content.map((b: any) => (b.id === 'toolu_IMG' ? { ...b, input: {} } : b)) }
        : m,
    );
    const { $, next, hook } = setup(form);
    const out = (await hook('session.compact')($, { trigger: 'manual', messages: rows }, next())) as { messages: Row[] };
    const text = out.messages.flatMap((r) => r.toolResults ?? []).find((r) => r.tool_use_id === 'toolu_IMG')!.text;
    expect(text).toContain('a tool result image');
    expect(text).not.toContain('a pasted image');
  });

  it('finds the original prompt behind a row that gained a placeholder', () => {
    const original = [
      { role: 'user', text: 'start', toolUses: [] },
      { role: 'user', text: 'yes', toolUses: [] },
      { role: 'user', text: 'k', toolUses: [] },
      { role: 'user', text: 'yes', toolUses: [] },
    ];
    const trimmed = [
      { role: 'user', text: 'start', toolUses: [] },
      { role: 'user', text: 'yes\n[image removed by compaction · a pasted image (image/png) · no description recorded · no copy kept]', toolUses: [] },
      { role: 'user', text: 'k', toolUses: [] },
      { role: 'user', text: 'yes', toolUses: [] },
    ];
    expect(originalIndex(original as never, trimmed as never, 3)).toBe(3);
  });
});
