import { describe, expect, it } from 'vitest';
import { compactSession, summarize } from '../hooks/fast-jev.ts';
import { reductionRatio, resolveOptions } from '../src/compact.js';
import { isHostOnly } from '../src/state.js';
import { findSteps, scoreSteps, stepLadder } from '../src/steps.js';
import type { CompactResult, JevAsker, Message, SessionMessage } from '../src/types.js';

const user = (text: string): Message => ({ role: 'user', text, toolUses: [] });
const reply = (text: string): Message => ({ role: 'assistant', text, toolUses: [] });
const call = (id: string, text = ''): Message => ({ role: 'assistant', text, toolUses: [{ tool_use_id: id, tool: 'Bash', input: { command: `run ${id}` } }] });
const result = (id: string, text: string): Message => ({ role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: id, text, isError: false }] });

const BIG = 'build output line that the task already acted on\n'.repeat(60); // ~3,000 chars
const REMINDER = `<system-reminder>\n${'The task list is unchanged. '.repeat(60)}\n</system-reminder>`;

/** Three prompts, all of them goal prompts: exchange drop has nothing to offer here. */
function session(): Message[] {
  return [
    user('start'),
    reply('ok'),
    user('build the parser'),
    call('a'), result('a', BIG),
    reply(`Built the parser. ${'Details of the first build. '.repeat(50)}`),
    user(REMINDER),
    user('now add the tests'),
    call('b'), result('b', BIG),
    call('c'), result('c', BIG),
    user('and run them'),
    call('d'), result('d', BIG),
    call('e'), result('e', BIG),
    call('f', 'Running the final check.'), result('f', 'pass'),
  ];
}

const options = resolveOptions({ goal: 'add tests', preserveRecentMessages: 2 });

function asker(score: (key: string) => number | 'fail'): JevAsker {
  return {
    ask: async (_state, questions) => {
      const answers: Record<string, { noul: number }> = {};
      for (const key of Object.keys(questions)) {
        const n = score(key);
        if (n === 'fail') throw new Error('down');
        answers[key] = { noul: n };
      }
      return { answers };
    },
  } as JevAsker;
}

const toolResult = (messages: Message[]): CompactResult => {
  const chars = messages.reduce((sum, m) => sum + m.text.length + (m.toolResults ?? []).reduce((a, r) => a + r.text.length, 0) + m.toolUses.length * 30, 0);
  return { messages, decisions: [], stats: { charsBefore: chars, charsAfter: chars } } as unknown as CompactResult;
};

describe('findSteps', () => {
  it('offers closed tool runs, replies and host-only rows, never a prompt, row 0, a pinned row or typed text', () => {
    const messages = session();
    const steps = findSteps(messages, 2);
    expect(steps.map((s) => [s.from, s.to])).toEqual([[3, 5], [5, 6], [6, 7], [8, 10], [10, 12], [13, 15], [15, 17]]);
    const prompts = new Set([0, 2, 7, 12]);
    for (const s of steps) for (let i = s.from; i < s.to; i++) expect(prompts.has(i)).toBe(false);
    expect(steps.filter((s) => s.current).map((s) => s.from)).toEqual([13, 15]);
  });

  it('never offers a user row holding typed words beside a host block, or a call without its result', () => {
    const mixed = session();
    mixed[6] = user(`${REMINDER}\nplease keep the old parser API`);
    expect(findSteps(mixed, 2).some((s) => s.from === 6)).toBe(false);
    expect(isHostOnly(REMINDER)).toBe(true);
    expect(isHostOnly(`${REMINDER} typed`)).toBe(false);
    const between = session();
    between.splice(9, 0, user(REMINDER)); // a host row inside the open run of call b
    expect(findSteps(between, 2).some((s) => s.from === 9)).toBe(false);
    const parallel: Message[] = [user('start'), user('go'), { ...call('a'), toolUses: [...call('a').toolUses, ...call('b').toolUses] }, result('a', BIG), user(REMINDER), result('b', BIG), reply('done'), user('next'), reply('ok')];
    expect(findSteps(parallel, 1).some((s) => s.from <= 4 && s.to > 4)).toBe(false);
    const unpaired = session();
    unpaired.splice(9, 1);
    expect(findSteps(unpaired, 2).some((s) => s.from === 8)).toBe(false);
  });
});

describe('stepLadder', () => {
  it('trims only before the last prompt when that reaches the minimum', async () => {
    const messages = session();
    const scores = await scoreSteps(messages, asker(() => 0.01), options);
    const out = stepLadder(scores, toolResult(messages), 0.2, (id) => `arch/${id}.txt`)!;
    expect(out.pass).toBe('trim');
    expect(out.applied.archived.every((a) => !a.text.includes('run d') && !a.text.includes('run e'))).toBe(true);
  });

  it('fills from the lowest score, the current task included, when trimming falls short', async () => {
    const messages = session();
    const scores = await scoreSteps(messages, asker((k) => (k === 's7' ? 0.02 : 0.9)), options);
    const out = stepLadder(scores, toolResult(messages), 0.75, (id) => `arch/${id}.txt`)!;
    expect(out.pass).toBe('fill');
    expect(out.applied.archived.some((a) => a.text.includes('run e'))).toBe(true);
    expect(reductionRatio(out.result)).toBeGreaterThanOrEqual(0.75);
    expect(stepLadder(scores, toolResult(messages), 0.99, (id) => id)).toBeUndefined();
  });
});

describe('step trim through compactSession', () => {
  const rows = (): SessionMessage[] => session().map((m, i) => ({ ...m, handle: `h-${i}` }) as SessionMessage);
  const fetchWith = (step: number | 'fail') => async (_url: string, init?: { body?: string }) => {
    const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
    const keys = Object.keys(questions);
    if (step === 'fail' && keys.some((k) => /^s\d+$/.test(k))) return { status: 500, ok: false, text: 'down' };
    const answers = Object.fromEntries(keys.map((k) => [k, { noul: /^s\d+$/.test(k) ? step : 0.95 }]));
    return { status: 200, ok: true, text: JSON.stringify({ answers }) };
  };
  const memory = () => {
    const files = new Map<string, string>();
    const pathOf = (id: string) => `arch/${id}.txt`;
    return { files, pathOf, cite: (id: string, text: string) => (files.set(pathOf(id), text), pathOf(id)), flush: async () => {} };
  };
  const cfg = (extra: Record<string, unknown> = {}) => ({ apiKey: 'k', preserveRecentMessages: 2, minReductionRatio: 0.6, archiveResults: true, proseTrim: true, ...extra }) as never;
  const ids = (ms: readonly Message[]) => ms.flatMap((m) => [...m.toolUses.map((u) => u.tool_use_id), ...(m.toolResults ?? []).map((r) => r.tool_use_id)]);

  it('reaches a minimum no earlier stage can, keeping every prompt verbatim, every call paired and every citation written', async () => {
    const a = memory();
    const out = await compactSession(rows(), cfg(), fetchWith(0.9), a);
    expect(reductionRatio(out.result)).toBeGreaterThanOrEqual(0.6);
    expect(out.steps?.pass).toBe('fill');
    const prompts = (ms: readonly Message[]) => ms.filter((m) => m.role === 'user' && m.text.length > 0 && !m.text.startsWith('<')).map((m) => m.text);
    expect(prompts(out.messages)).toEqual(prompts(session()));
    const left = ids(out.messages);
    expect(left.filter((id) => left.indexOf(id) !== left.lastIndexOf(id)).length).toBe(left.length);
    // A chain of step notes is folded into one note whose file holds them (src/chains.ts); follow it.
    const shown = out.messages.map((m) => m.text).join('\n');
    // Folded notes (src/stepnotes.ts, src/chains.ts) name a file holding the originals; follow them, nested too.
    const folded: string[] = [];
    const queue = [shown];
    while (queue.length > 0) {
      const text = queue.pop()!;
      for (const m of text.matchAll(/(?:each naming its file|each step's note and file): (arch\/(?:notes|stepnotes)-[0-9a-f]{12}\.txt)/g)) {
        expect(a.files.has(m[1]!)).toBe(true);
        folded.push(a.files.get(m[1]!)!);
        queue.push(a.files.get(m[1]!)!);
      }
    }
    const cited = [...[shown, ...folded].join('\n').matchAll(/full text: (arch\/step-[0-9a-f]{12}\.txt)/g)].map((m) => m[1]);
    expect(cited.length).toBeGreaterThan(0);
    for (const path of cited) expect(a.files.has(path!)).toBe(true);
    expect(summarize(out.result, out.prose, out.notifications, out.injected, out.exchanges, out.steps)).toMatch(/; step trim \(lowest-scored first\): \d+ archived, \d+ chars$/);
  });

  it('falls short, changing nothing beyond earlier stages, when the step request fails or stepTrim is off', async () => {
    for (const [extra, fetch] of [[{}, fetchWith('fail')], [{ stepTrim: false }, fetchWith(0.01)]] as const) {
      const a = memory();
      const out = await compactSession(rows(), cfg(extra), fetch, a);
      expect(reductionRatio(out.result)).toBeLessThan(0.6);
      expect([...a.files.keys()].some((p) => p.includes('step-'))).toBe(false);
    }
  });
});
