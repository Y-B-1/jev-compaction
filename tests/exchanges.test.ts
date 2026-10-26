import { describe, expect, it } from 'vitest';
import { compactSession } from '../hooks/fast-jev.ts';
import { applyExchanges, findExchanges, renderExchange, scoreExchanges } from '../src/exchanges.js';
import { resolveOptions } from '../src/compact.js';
import type { JevAsker, Message, SessionMessage } from '../src/types.js';

const user = (text: string): Message => ({ role: 'user', text, toolUses: [] });
const reply = (text: string): Message => ({ role: 'assistant', text, toolUses: [] });
const call = (id: string, tool: string, input: Record<string, unknown>, text = ''): Message => ({
  role: 'assistant',
  text,
  toolUses: [{ tool_use_id: id, tool, input }],
});
const result = (id: string, text: string): Message => ({ role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: id, text, isError: false }] });

const BIG = 'line of old output that nobody needs any more\n'.repeat(80);

/** Five exchanges: a pinned first prompt, two old finished ones, then the three goal prompts. */
function session(): Message[] {
  return [
    user('start the session'),
    reply('ok'),
    user('look at the logging module'),
    call('t1', 'Read', { file_path: 'src/log.ts' }),
    result('t1', BIG),
    reply('The logging module writes to stderr; nothing to change.'),
    user('check the old cache layer'),
    call('t2', 'Grep', { pattern: 'cache' }),
    result('t2', BIG),
    call('t3', 'Read', { file_path: 'src/cache.ts' }),
    result('t3', BIG),
    reply('The cache layer is unused; leaving it.'),
    user('now fix the parser bug'),
    reply('Looking at the parser.'),
    user('use the second approach'),
    reply('Switching.'),
    user('run the tests'),
    reply('Running.'),
  ];
}

const options = resolveOptions({ goal: 'fix the parser', preserveRecentMessages: 2 });

function asker(noul: number | 'fail', seen: string[] = []): JevAsker {
  return {
    ask: async (_state, questions) => {
      if (noul === 'fail') throw new Error('down');
      seen.push(...Object.values(questions).map((q) => (q as { instructions: string }).instructions));
      return { answers: Object.fromEntries(Object.keys(questions).map((k) => [k, { noul }])) };
    },
  } as JevAsker;
}

describe('findExchanges', () => {
  it('offers only old finished exchanges, never a goal prompt, the first row or one with an unpaired call', () => {
    const messages = session();
    const found = findExchanges(messages, 2);
    expect(found.map((e) => [e.prompt, e.work])).toEqual([
      [2, [3, 6]],
      [6, [7, 12]],
    ]);
    expect(findExchanges([...messages.slice(0, 4), ...messages.slice(5)], 2).map((e) => e.prompt)).toEqual([5]);
  });

  it('never offers an exchange whose work rows hold text the user typed', () => {
    const typed = (text: string): Message => ({ role: 'user', text, toolUses: [] });
    const withBang = session();
    withBang.splice(4, 0, typed('<bash-input>git status</bash-input>'));
    expect(findExchanges(withBang, 2).map((e) => e.prompt)).toEqual([7]);
    const withArgs = session();
    withArgs.splice(4, 0, typed('<command-message>review</command-message>\n<command-name>/review</command-name>\n<command-args>focus on the parser</command-args>'));
    expect(findExchanges(withArgs, 2).map((e) => e.prompt)).toEqual([7]);
    const angled = session();
    angled.splice(4, 0, typed('<command-name>/review</command-name>\n<command-args><Button> props are wrong</command-args>'));
    expect(findExchanges(angled, 2).map((e) => e.prompt)).toEqual([7]);
    const emptyArgs = session();
    emptyArgs.splice(4, 0, typed('<command-name>/clear</command-name>\n<command-args> </command-args>'));
    expect(findExchanges(emptyArgs, 2).map((e) => e.prompt)).toEqual([2, 7]);
    const queued = '<system-reminder>\nThe user sent a new message while you were working:\nstop touching the cache\n</system-reminder>';
    const inRow = session();
    inRow.splice(4, 0, typed(queued));
    expect(findExchanges(inRow, 2).map((e) => e.prompt)).toEqual([7]);
    const inResult = session();
    inResult[4] = { ...inResult[4]!, toolResults: [{ tool_use_id: 't1', text: `${BIG}\n${queued}`, isError: false }] };
    expect(findExchanges(inResult, 2).map((e) => e.prompt)).toEqual([6]);
  });

  it('renders every row of the exchange into the archive text', () => {
    const messages = session();
    const [, cache] = findExchanges(messages, 2);
    const text = renderExchange(messages, cache!);
    expect(text).toContain('check the old cache layer');
    expect(text).toContain('"pattern": "cache"');
    expect(text.split(BIG).length - 1).toBe(2);
    expect(text).toContain('The cache layer is unused; leaving it.');
  });
});

describe('applyExchanges', () => {
  it('keeps the prompt verbatim, replaces the work with one cited row and leaves no orphan tool id', async () => {
    const messages = session();
    const scores = await scoreExchanges(messages, asker(0.01), options);
    const files = new Map<string, string>();
    const out = applyExchanges(scores, 0.05, (id, text) => (files.set(id, text), `arch/${id}.txt`));
    expect(out.stats.dropped).toBe(2);
    expect(out.messages.filter((m) => m.role === 'user' && m.text.length > 0).map((m) => m.text)).toEqual(
      messages.filter((m) => m.role === 'user' && m.text.length > 0).map((m) => m.text),
    );
    const marker = out.messages[3]!;
    expect(marker.text).toMatch(/^\[jev-compaction archived this exchange: 3 rows, 1 tool calls \(Read\), \d+ chars; full text: arch\/exchange-[0-9a-f]{12}\.txt/);
    expect(marker.text).toContain('Last reply: The logging module writes to stderr');
    expect(out.messages[5]!.text).toContain('2 tool calls (Grep, Read)');
    const ids = (ms: Message[]) => ms.flatMap((m) => [...m.toolUses.map((u) => u.tool_use_id), ...(m.toolResults ?? []).map((r) => r.tool_use_id)]);
    expect(ids(out.messages)).toEqual([]);
    expect(files.size).toBe(2);
    expect([...files.values()].every((t) => t.includes(BIG))).toBe(true);
    expect(out.messages.slice(6)).toEqual(messages.slice(12));
  });

  it('keeps an exchange Jev scores at or above the cut-off, one with no answer, and one with no archive path', async () => {
    const messages = session();
    const kept = applyExchanges(await scoreExchanges(messages, asker(0.5), options), 0.35, () => 'p');
    expect(kept.messages).toEqual(messages);
    expect(applyExchanges(await scoreExchanges(messages, asker(0.2), options), 0.35, () => 'p').stats.dropped).toBe(2);
    const failed = await scoreExchanges(messages, asker('fail'), options);
    expect(failed.stats.failed).toBe('down');
    expect(applyExchanges(failed, 0.35, () => 'p').messages).toEqual(messages);
    expect(applyExchanges(await scoreExchanges(messages, asker(0.01), options), 0.35, () => undefined).messages).toEqual(messages);
  });
});

describe('exchange drop through compactSession', () => {
  const sessionRows = (): SessionMessage[] => session().map((m, i) => ({ ...m, handle: `h-${i}` }) as SessionMessage);
  const isExchange = (q: unknown) => String((q as { instructions?: string }).instructions ?? '').includes('done for the request in entry');
  const fetchWith = (exchange: number, other = 0.9) => async (_url: string, init?: { body?: string }) => {
    const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
    const answers = Object.fromEntries(Object.entries(questions).map(([k, q]) => [k, { noul: isExchange(q) ? exchange : other }]));
    return { status: 200, ok: true, text: JSON.stringify({ answers }) };
  };
  const memory = () => {
    const files = new Map<string, string>();
    const pathOf = (id: string) => `arch/${id}.txt`;
    return { files, pathOf, cite: (id: string, text: string) => (files.set(id, text), pathOf(id)), flush: async () => {} };
  };
  const cfg = (extra: Record<string, unknown> = {}) => ({
    apiKey: 'k',
    preserveRecentMessages: 2,
    minReductionRatio: 0.5,
    archiveResults: true,
    proseTrim: true,
    ...extra,
  });

  it('reaches the minimum by dropping old exchanges after every other stage fell short', async () => {
    const a = memory();
    const out = await compactSession(sessionRows(), cfg() as any, fetchWith(0.01), a);
    expect(out.exchanges?.cutoff).toBe(0.05);
    expect(out.exchanges?.stats.dropped).toBe(2);
    expect(out.messages.some((m) => m.text.startsWith('[jev-compaction archived this exchange'))).toBe(true);
    expect(out.messages.map((m) => m.text)).toContain('check the old cache layer');
    expect([...a.files.values()].filter((t) => t.startsWith('request (entry'))).toHaveLength(2);
  });

  it('runs when exchangeDrop is the only stage on', async () => {
    const out = await compactSession(sessionRows(), cfg({ proseTrim: false, notificationTrim: false, injectTrim: false, replyTrim: false }) as any, fetchWith(0.01), memory());
    expect(out.exchanges?.stats.dropped).toBe(2);
  });

  it('changes nothing when exchangeDrop is off, without an archive, or when every exchange is still needed', async () => {
    const off = await compactSession(sessionRows(), cfg({ exchangeDrop: false }) as any, fetchWith(0.01), memory());
    expect(off.exchanges).toBeUndefined();
    const noArchive = await compactSession(sessionRows(), cfg({ archiveResults: false }) as any, fetchWith(0.01));
    expect(noArchive.exchanges).toBeUndefined();
    const needed = await compactSession(sessionRows(), cfg() as any, fetchWith(0.9), memory());
    expect(needed.exchanges?.cutoff).toBeUndefined();
    expect(needed.messages.some((m) => m.text.includes('archived this exchange'))).toBe(false);
  });
});
