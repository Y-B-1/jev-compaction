import { describe, expect, test } from 'vitest';

import { compact, resolveOptions } from '../src/compact.js';
import { trimRescue } from '../src/trim.js';
import type { JevAsker, JevQuestions, Message } from '../src/types.js';

const pathOf = (id: string): string => `.claude/jev-compaction/${id}.txt`;
const options = { ...resolveOptions({ preserveRecentMessages: 2 }), goal: 'Fix the stale cache.' };

const BODY =
  'The worker examined the module and found the cache stale. '.repeat(10) +
  '\nerror: build failed in src/cache/store.ts at 4f3a9c2 with exit code 2.\n' +
  'Trailing detail about the layout of the code. '.repeat(12);
const NOTIFICATION = [
  '<task-notification>',
  '<task-id>a1b2c3</task-id>',
  '<status>completed</status>',
  '<summary>Worker finished the audit</summary>',
  `<result>${BODY}</result>`,
  '<usage><total_tokens>81234</total_tokens></usage>',
  '</task-notification>',
].join('\n');
const FILLER = 'Let me have a good look at the layout of this module now. '.repeat(12);

const user = (text: string): Message => ({ role: 'user', text, toolUses: [] });
const assistant = (text: string): Message => ({ role: 'assistant', text, toolUses: [] });
const session = (): Message[] => [
  user('Fix the stale cache.'),
  assistant(`${FILLER} We settled on flamingo-7.`),
  user(NOTIFICATION),
  assistant('Working.'),
  user('ok'),
];

/** Answers every question with `noul`; a key matching `refuse` throws, as a failed request would. */
function asker(noul: number, refuse?: RegExp): JevAsker {
  return {
    async ask(_state, questions: JevQuestions) {
      if (refuse && Object.keys(questions).some((k) => refuse.test(k))) throw new Error('jev down');
      return { answers: Object.fromEntries(Object.keys(questions).map((k) => [k, { type: 'noul' as const, noul }])) };
    },
  };
}

async function run(minRatio: number, a: JevAsker, proseTrim: boolean, cited: string[] = []) {
  const messages = session();
  const result = await compact(messages, a, { preserveRecentMessages: 2 });
  return trimRescue({
    messages,
    result,
    asker: a,
    options,
    proseTrim,
    minRatio,
    archive: { cite: (id) => (cited.push(id), pathOf(id)) },
    pathOf,
  });
}

describe('trimRescue', () => {
  test('takes the first rung that reaches the minimum and cites only that rung', async () => {
    const cited: string[] = [];
    const r = await run(0.2, asker(0.01), true, cited);
    expect(r.rescue?.rung).toBe(0.05);
    expect(r.rescue?.applied.notified?.stats).toMatchObject({ folded: 1 });
    expect(r.rescue?.applied.prose?.stats.dropped).toBeGreaterThan(0);
    expect(cited.some((id) => /^notif-a1b2c3-[0-9a-f]{8}$/.test(id))).toBe(true);
    expect(r.rescue?.result.messages).toHaveLength(5);
    expect(r.scores?.entries).toHaveLength(1);
  });

  test('returns no rescue and keeps the scoring stats when no rung reaches the minimum', async () => {
    const r = await run(0.99, asker(0.01), true);
    expect(r.rescue).toBeUndefined();
    expect(r.out.notifications?.stats).toMatchObject({ eligible: 1, scored: 1 });
    expect(r.out.prose?.stats.requests).toBeGreaterThan(0);
  });

  test('with proseTrim off the prose is untouched and not scored', async () => {
    const r = await run(0.1, asker(0.01), false);
    expect(r.rescue?.applied.prose).toBeUndefined();
    expect(r.out.prose).toBeUndefined();
    expect(r.rescue?.applied.notified?.stats.folded).toBe(1);
  });

  test('a failed prose stage is skipped and the notification stage still runs', async () => {
    const r = await run(0.1, asker(0.01, /^c\d+$/), true);
    expect(r.out.prose?.stats.failed).toBeDefined();
    expect(r.rescue?.applied.prose).toBeUndefined();
    expect(r.rescue?.applied.notified?.stats.folded).toBe(1);
  });

  test('a failed notification stage is skipped and the prose stage still runs', async () => {
    const r = await run(0.1, asker(0.01, /^n/), true);
    expect(r.out.notifications?.stats.failed).toBeDefined();
    expect(r.rescue?.applied.notified).toBeUndefined();
    expect(r.rescue?.applied.prose?.stats.dropped).toBeGreaterThan(0);
  });

  test('throws when the archive names a different path than the ladder promised', async () => {
    const messages = session();
    const a = asker(0.01);
    const result = await compact(messages, a, { preserveRecentMessages: 2 });
    await expect(
      trimRescue({ messages, result, asker: a, options, proseTrim: false, minRatio: 0.1, archive: { cite: () => 'elsewhere.txt' }, pathOf }),
    ).rejects.toThrow('archive path for notif-a1b2c3 is not');
  });
});
