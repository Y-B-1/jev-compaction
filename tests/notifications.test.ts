import { describe, expect, test } from 'vitest';

import { messageChars, reductionRatio, resolveOptions } from '../src/compact.js';
import {
  NOTIFICATION_RUNGS,
  applyNotifications,
  ladderRescueWith,
  mergeReplaced,
  scoreNotifications,
} from '../src/notifications.js';
import type { CompactResult, JevAsker, JevQuestions, Message } from '../src/types.js';

const MARKER = '[jev-compaction ';
const options = { ...resolveOptions({ preserveRecentMessages: 2 }), goal: 'Fix the stale cache.' };

type Parts = {
  id?: string;
  status?: string;
  summary?: string;
  body?: string;
  usage?: string;
  diagnostics?: string;
};

const BODY =
  'The worker examined the module and found the cache stale. '.repeat(10) +
  '\nerror: build failed in src/cache/store.ts at 4f3a9c2 with exit code 2.\n' +
  'Trailing detail about the layout of the code. '.repeat(12);

function notification(p: Parts = {}): string {
  const id = p.id ?? 'a1b2c3';
  return [
    '<task-notification>',
    `<task-id>${id}</task-id>`,
    `<tool-use-id>toolu_${id}</tool-use-id>`,
    `<output-file>/tmp/tasks/${id}.output</output-file>`,
    `<status>${p.status ?? 'completed'}</status>`,
    `<summary>${p.summary ?? 'Worker finished the audit'}</summary>`,
    `<result>${p.body ?? BODY}</result>`,
    `<usage>${p.usage ?? '<total_tokens>81234</total_tokens><tool_uses>31</tool_uses><duration_ms>412345</duration_ms>'}</usage>`,
    `<diagnostics>${p.diagnostics ?? 'Output file is missing: /tmp/tasks/gone.output'}</diagnostics>`,
    '</task-notification>',
  ].join('\n');
}

const user = (text: string): Message => ({ role: 'user', text, toolUses: [] });
const assistant = (text: string): Message => ({ role: 'assistant', text, toolUses: [] });

/** Message 0, then a worker line and the given user texts, then `tail` pinned filler messages. */
function session(texts: string[], tail = 2): Message[] {
  return [
    user('Fix the stale cache.'),
    ...texts.flatMap((t) => [assistant('Dispatching a worker.'), user(t)]),
    ...Array.from({ length: tail }, (_, i) => (i % 2 === 0 ? assistant('Working.') : user('ok'))),
  ];
}

type Call = { state: unknown; questions: JevQuestions };
function fakeAsker(answer: (key: string) => number | undefined, calls: Call[] = []): JevAsker {
  return {
    async ask(state, questions) {
      calls.push({ state, questions });
      const answers: Record<string, { type: 'noul'; noul: number }> = {};
      for (const key of Object.keys(questions)) {
        const n = answer(key);
        if (n !== undefined) answers[key] = { type: 'noul', noul: n };
      }
      return { answers };
    },
  };
}

const archiveAt = (id: string): string => `.claude/jev-compaction/${id}.txt`;
const cite = (id: string): string => archiveAt(id);

/** `noul` null means Jev gives no answer. */
async function applied(texts: string[], rung: number | 'all', noul: number | null = 0.01, tail = 2) {
  const messages = session(texts, tail);
  const scores = await scoreNotifications(messages, fakeAsker(() => noul ?? undefined), options);
  return { messages, scores, out: applyNotifications(scores, rung, cite) };
}

describe('strip (plan item 1)', () => {
  test('removes usage and diagnostics, keeps the identity tags, leaves one marker with the archive path', async () => {
    const original = notification();
    const { out } = await applied([original], 0.05, 0.9);
    const text = out.messages[2]!.text;
    expect(text).not.toContain('<usage>');
    expect(text).not.toContain('<diagnostics>');
    expect(text).not.toContain('81234');
    expect(text).not.toContain('gone.output');
    for (const kept of [
      '<task-id>a1b2c3</task-id>',
      '<tool-use-id>toolu_a1b2c3</tool-use-id>',
      '<output-file>/tmp/tasks/a1b2c3.output</output-file>',
      '<status>completed</status>',
      '<summary>Worker finished the audit</summary>',
      `<result>${BODY}</result>`,
    ]) {
      expect(text).toContain(kept);
    }
    expect(text.split(MARKER)).toHaveLength(2);
    expect(text).toContain(
      `[jev-compaction removed usage and diagnostics; full notification: ${archiveAt('notif-a1b2c3')} (Read or grep it if needed)]`,
    );
    expect(out.stats).toMatchObject({ stripped: 1, folded: 0 });
  });

  test('leaves a notification alone when stripping would not make it shorter', async () => {
    const tiny = notification({ usage: '1', diagnostics: '2' });
    const { out, messages } = await applied([tiny], 0.05, 0.9);
    expect(out.messages[2]).toBe(messages[2]);
    expect(out.archived).toEqual([]);
  });
});

describe('fold (plan item 2)', () => {
  test('a noul below the rung folds the body to a truncateHeadChars head, a marker and a salvage block', async () => {
    const original = notification();
    const { out } = await applied([original], 0.15, 0.1);
    const text = out.messages[2]!.text;
    const marker = `[jev-compaction folded ${BODY.length - options.truncateHeadChars} chars of this task result and removed its usage and diagnostics; full notification: ${archiveAt('notif-a1b2c3')} (Read or grep it if needed)]`;
    const lines = text.split('\n');
    expect(lines).toContain(marker);
    expect(text).toContain(`<result>${BODY.slice(0, options.truncateHeadChars)}\n${marker}\n`);
    expect(text).toContain('errors: error: build failed in src/cache/store.ts at 4f3a9c2 with exit code 2.');
    expect(text).toContain('refs: src/cache/store.ts 4f3a9c2');
    expect(text).toContain('\n</result>');
    expect(text).not.toContain('<usage>');
    expect(text.split(MARKER)).toHaveLength(2);
    expect(text.length).toBeLessThan(original.length * 0.6);
    expect(out.stats).toMatchObject({ folded: 1, stripped: 0 });
  });

  test('a noul at the rung, or no answer, keeps the body byte-identical and still strips', async () => {
    for (const noul of [0.15, 0.9, null]) {
      const { out } = await applied([notification()], 0.15, noul);
      const text = out.messages[2]!.text;
      expect(text).toContain(`<result>${BODY}</result>`);
      expect(text).not.toContain('<usage>');
      expect(out.stats).toMatchObject({ folded: 0, stripped: 1 });
    }
  });

  test('one score per notification serves every rung', async () => {
    const calls: Call[] = [];
    const messages = session([notification()]);
    const scores = await scoreNotifications(messages, fakeAsker(() => 0.1, calls), options);
    const folded = (rung: number | 'all') =>
      applyNotifications(scores, rung, cite).stats.folded;
    expect([0.05, 0.15, 0.35, 'all'].map((r) => folded(r as number | 'all'))).toEqual([0, 1, 1, 1]);
    expect(calls).toHaveLength(1);
  });
});

describe('never touched (plan item 3)', () => {
  const cases: [string, (n: string) => Message[]][] = [
    ['message 0', (n) => [user(n), ...session([], 4).slice(1)]],
    ['the pinned tail', (n) => [user('Fix the stale cache.'), assistant('Dispatching.'), assistant('x'), user('y'), user(n)]],
    ['a status other than completed', (_n) => session([notification({ status: 'failed' })])],
    ['a notification with no result', (_n) => session([notification().replace(/<result>[\s\S]*<\/result>\n/, '')])],
    ['typed text that mentions the tag', (_n) => session([`please explain what a <task-notification> is. ${BODY}`])],
    ['an assistant message', (_n) => [user('Fix the stale cache.'), assistant(notification()), assistant('a'), user('b'), assistant('c')]],
    ['a user message with tool results', (n) => [user('go'), { role: 'user', text: n, toolUses: [], toolResults: [{ tool_use_id: 't', text: 'r' }] }, assistant('a'), user('b'), assistant('c')]],
  ];
  for (const [name, build] of cases) {
    test(name, async () => {
      const messages = build(notification());
      const scores = await scoreNotifications(messages, fakeAsker(() => 0.01), options);
      const out = applyNotifications(scores, 'all', cite);
      expect(out.messages).toEqual(messages);
      out.messages.forEach((m, i) => expect(m).toBe(messages[i]));
      expect(out.archived).toEqual([]);
    });
  }

  test('a second pass over a folded notification changes nothing and never re-cites', async () => {
    const first = await applied([notification()], 'all', 0.01);
    expect(first.out.archived).toHaveLength(1);
    const cited: string[] = [];
    const scores = await scoreNotifications(first.out.messages, fakeAsker(() => 0.01), options);
    const second = applyNotifications(scores, 'all', (id) => (cited.push(id), archiveAt(id)));
    expect(second.messages).toEqual(first.out.messages);
    expect(cited).toEqual([]);
    expect(second.archived).toEqual([]);
  });

  test('asks no question when nothing is eligible', async () => {
    const calls: Call[] = [];
    const scores = await scoreNotifications(session([]), fakeAsker(() => 0.1, calls), options);
    expect(calls).toHaveLength(0);
    expect(scores.stats).toMatchObject({ eligible: 0, requests: 0 });
  });
});

describe('extractive property (plan item 4)', () => {
  const salvageLine = /^(?:errors|refs): /;
  function check(original: string, text: string): void {
    for (const line of text.split(/\r?\n/)) {
      if (line.startsWith(MARKER)) continue;
      if (salvageLine.test(line)) {
        const items = line.replace(salvageLine, '').split(/ \| | /);
        for (const item of items) expect(original).toContain(item);
        continue;
      }
      expect(original).toContain(line);
    }
  }

  const bodies: [string, string][] = [
    ['plain prose', 'Plain sentence about nothing in particular. '.repeat(40)],
    ['errors and paths', `${'filler words here. '.repeat(30)}\nfatal: cannot open src/a/b.ts\nsee https://example.com/x and 4f3a9c2\n`],
    ['CRLF', `${'line one of the report\r\n'.repeat(30)}error: broken at lib/z.ts\r\n`],
    ['surrogate pair at the cut', `${'a'.repeat(299)}😀${'b'.repeat(400)} error: oops`],
    ['surrogate pair just before the cut', `${'a'.repeat(298)}😀${'b'.repeat(400)}`],
    ['escaped JSON record', `{"result":"${'text with \\"quotes\\" and \\n escapes '.repeat(20)}"}`],
    ['assignment value', `${'x '.repeat(400)}\nTOKEN_NAME=abc123 error: auth failed`],
  ];
  for (const [name, body] of bodies) {
    test(`every non-marker line is a slice of the input: ${name}`, async () => {
      const original = notification({ body });
      const { out } = await applied([original], 'all', 0.01);
      const text = out.messages[2]!.text;
      expect(text.length).toBeLessThan(original.length);
      if (name.startsWith('assignment')) {
        expect(text).not.toContain('abc123');
        return;
      }
      check(original, text);
      // A lone surrogate would make the transcript invalid.
      expect(text).not.toMatch(/[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]/);
    });
  }
});

describe('archive (plan item 5)', () => {
  test('the archive callback gets the whole original under a notif- id, and the output cites the path it returned', async () => {
    const original = notification();
    const seen: [string, string][] = [];
    const messages = session([original]);
    const scores = await scoreNotifications(messages, fakeAsker(() => 0.01), options);
    const out = applyNotifications(scores, 'all', (id, text) => (seen.push([id, text]), `elsewhere/${id}.md`));
    expect(seen).toEqual([['notif-a1b2c3', original]]);
    expect(out.archived).toEqual([{ id: 'notif-a1b2c3', text: original, path: 'elsewhere/notif-a1b2c3.md' }]);
    expect(out.messages[2]!.text).toContain('full notification: elsewhere/notif-a1b2c3.md (Read or grep it if needed)');
  });

  test('no archive callback, or one that returns nothing, changes nothing', async () => {
    const messages = session([notification()]);
    const scores = await scoreNotifications(messages, fakeAsker(() => 0.01), options);
    for (const c of [undefined, () => undefined]) {
      const out = applyNotifications(scores, 'all', c);
      expect(out.messages[2]).toBe(messages[2]);
      expect(out.archived).toEqual([]);
    }
  });

  test('two notifications with the same task id get distinct archive ids', async () => {
    const { out } = await applied([notification(), notification()], 'all', 0.01, 2);
    expect(out.archived.map((a) => a.id)).toEqual(['notif-a1b2c3', 'notif-a1b2c3-2']);
  });

  test('an unsafe task id is made path-safe', async () => {
    const { out } = await applied([notification({ id: '../x y' })], 'all', 0.01);
    expect(out.archived.map((a) => a.id)).toEqual(['notif--x-y']);
  });
});

describe('ladder (plan item 6)', () => {
  function resultOf(messages: Message[]): CompactResult {
    const charsBefore = messages.reduce((s, m) => s + messageChars(m), 0);
    return {
      messages,
      decisions: [],
      stats: {
        messagesBefore: messages.length, messagesAfter: messages.length, charsBefore, charsAfter: charsBefore,
        calls: 0, kept: 0, resultsDropped: 0, callsDropped: 0, pinned: 0, stateTokens: 0, stateStage: '',
        requests: 0, failedRequests: 0, ms: 0,
      },
    };
  }
  const walk = (messages: Message[], scores: Awaited<ReturnType<typeof scoreNotifications>>, min = 0.25) =>
    ladderRescueWith((rung) => applyNotifications(scores, rung, cite), resultOf(messages), min);

  test('the rungs are the prose ladder, then all', () => {
    expect(NOTIFICATION_RUNGS).toEqual([0.05, 0.15, 0.35, 'all']);
  });

  test('takes the first rung that reaches the minimum against the original characters', async () => {
    const messages = session([notification({ id: 'a' }), notification({ id: 'b' })], 2);
    // noul 0.1: nothing folds at 0.05 (strip alone cannot reach 25%), both fold at 0.15.
    const scores = await scoreNotifications(messages, fakeAsker(() => 0.1), options);
    const rescue = walk(messages, scores)!;
    expect(rescue.rung).toBe(0.15);
    expect(reductionRatio(rescue.result)).toBeGreaterThanOrEqual(0.25);
    expect(rescue.applied.stats.folded).toBe(2);
  });

  test('the all rung folds every eligible notification when no scored rung reaches the minimum', async () => {
    const messages = session([notification({ id: 'a' }), notification({ id: 'b' })], 2);
    const scores = await scoreNotifications(messages, fakeAsker(() => 0.9), options);
    const rescue = walk(messages, scores)!;
    expect(rescue.rung).toBe('all');
    expect(rescue.applied.stats.folded).toBe(2);
  });

  test('is undefined when even the all rung falls short', async () => {
    const messages = session([notification()], 2);
    const scores = await scoreNotifications(messages, fakeAsker(() => 0.9), options);
    expect(walk(messages, scores, 0.99)).toBeUndefined();
  });

  test('merges by object identity, so a message dropped elsewhere does not shift the replacement', async () => {
    const messages = session([notification({ id: 'a' })], 2);
    const scores = await scoreNotifications(messages, fakeAsker(() => 0.01), options);
    const out = applyNotifications(scores, 'all', cite);
    const shifted = messages.filter((m) => m.text !== 'Dispatching a worker.');
    const merged = mergeReplaced(shifted, out.replaced);
    expect(merged).toHaveLength(shifted.length);
    expect(merged.filter((m) => m.text.includes('folded'))).toHaveLength(1);
    expect(merged.filter((m) => m !== messages[0] && !m.text.includes('folded') && m.text.startsWith('<task'))).toHaveLength(0);
  });
});

describe('scoring', () => {
  test('one request when the questions fit one request, none of them answers the keep questions of tool calls', async () => {
    const calls: Call[] = [];
    const messages = session([notification({ id: 'a' }), notification({ id: 'b' }), notification({ id: 'c' })]);
    const scores = await scoreNotifications(messages, fakeAsker(() => 0.1, calls), options);
    expect(calls).toHaveLength(1);
    expect(Object.keys(calls[0]!.questions)).toEqual(['n1', 'n2', 'n3']);
    expect(scores.stats).toMatchObject({ eligible: 3, scored: 3, requests: 1, failedRequests: 0 });
  });

  test('splits into more requests when maxRequestTokens is small', async () => {
    const calls: Call[] = [];
    const messages = session([notification({ id: 'a' }), notification({ id: 'b' }), notification({ id: 'c' })]);
    const small = { ...options, maxStateTokens: 3000 };
    const probe = fakeAsker(() => 0.1, []);
    const first = await scoreNotifications(messages, probe, small);
    const tokens = first.stats.stateTokens;
    const tight = { ...small, maxRequestTokens: tokens + 20 + 400 };
    await scoreNotifications(messages, fakeAsker(() => 0.1, calls), tight);
    expect(calls.length).toBeGreaterThan(1);
  });

  test('the question quotes the summary and a bounded preview and judges stale or restated reports', async () => {
    const calls: Call[] = [];
    await scoreNotifications(session([notification()]), fakeAsker(() => 0.1, calls), options);
    const q = calls[0]!.questions['n1']!;
    expect(q.type).toBe('noul');
    expect(q.instructions).toContain('a1b2c3');
    expect(q.instructions).toContain('Worker finished the audit');
    expect(q.instructions.length).toBeLessThan(1500);
    expect(q).toMatchObject({ criteria: { false: expect.stringContaining('stale, already stated in the assistant text') } });
  });

  test('a failed request leaves the notification unscored and sets failed when all requests fail', async () => {
    const down: JevAsker = { ask: async () => { throw new Error('down'); } };
    const scores = await scoreNotifications(session([notification()]), down, options);
    expect(scores.stats.failed).toBe('down');
    const out = applyNotifications(scores, 'all', cite);
    expect(out.archived).toEqual([]);
  });

  test('a missing answer keeps the body', async () => {
    const { out } = await applied([notification()], 0.35, null);
    expect(out.stats.folded).toBe(0);
  });
});

describe('repair T22R', () => {
  const filler = 'The worker looked around the module and kept notes about it. '.repeat(8);

  test('error clauses past the cut survive in a single-line (escaped JSON) body, within the salvage budget', async () => {
    const body =
      `{"result":"${filler}Error: boom at /src/foo.ts:3\\nMore narration follows here. ${'padding words '.repeat(30)}` +
      `Then TypeError: bad thing in lib/z.ts happened. ${'closing words '.repeat(30)}"}`;
    expect(body.indexOf('Error: boom')).toBeGreaterThan(300);
    expect(body.includes('\n')).toBe(false);
    const original = notification({ body });
    const { out } = await applied([original], 'all', 0.01);
    const text = out.messages[2]!.text;
    const errors = text.split('\n').find((l) => l.startsWith('errors: '));
    expect(errors).toBeDefined();
    expect(errors).toContain('Error: boom at /src/foo.ts:3');
    expect(errors).toContain('TypeError: bad thing in lib/z.ts happened.');
    expect(errors!.length).toBeLessThanOrEqual(options.salvageChars + 'errors: '.length);
    const items = errors!.replace(/^errors: /, '').split(' | ');
    for (const item of items) expect(original).toContain(item);
  });

  test('the status is the envelope own: a failed notification whose body says completed is never folded', async () => {
    const body = `${filler.repeat(3)}<status>completed</status> ${filler.repeat(3)}`;
    const original = notification({ status: 'failed', body });
    const messages = session([original]);
    const scores = await scoreNotifications(messages, fakeAsker(() => 0.01), options);
    expect(scores.stats.eligible).toBe(0);
    expect(applyNotifications(scores, 'all', cite).messages[2]).toBe(messages[2]);
  });

  test('a message holding two task-notifications is never folded', async () => {
    const original = `${notification({ id: 'aa' })}\n${notification({ id: 'bb' })}`;
    const messages = session([original]);
    const scores = await scoreNotifications(messages, fakeAsker(() => 0.01), options);
    expect(scores.stats.eligible).toBe(0);
    expect(applyNotifications(scores, 'all', cite).messages[2]).toBe(messages[2]);
  });
});
