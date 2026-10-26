import { describe, expect, it, vi } from 'vitest';
import {
  applyDecisions,
  batchCalls,
  buildJevRequest,
  collectToolCalls,
  compact as compactLib,
  compactMessages,
  decideCall,
  estimateTokens,
  fitState,
  goalFromMessages,
  isMachineText,
  JevClient,
  parseJevResponse,
  questionsFor,
  reductionRatio,
  resolveOptions,
  truncate,
  type HistoryToolCall,
  type JevAsker,
  type JevQuestions,
  type Message,
  type ToolCall,
} from '../src/index.js';

// These fixtures use short results; they predate the small-result floor,
// duplicate removal and the keep-signal guard, which have their own tests.
const LEGACY = { minResultChars: 0, dedupeRepeats: false, minKeepSignalCalls: 0 };
const compact: typeof compactLib = (messages, asker, options = {}) =>
  compactLib(messages, asker, { ...LEGACY, ...options });

function message(role: Message['role'], text: string, extra: Partial<Message> = {}): Message {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): Message {
  return message('assistant', '', { toolUses: [{ tool_use_id: id, tool, input, text }] });
}

function result(id: string, text: string, isError = false): Message {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }] });
}

const fileA = 'export const a = 1;\n'.repeat(50);
const fileB = 'export const b = 2;\n'.repeat(50);

function transcript(): Message[] {
  return [
    message('user', 'Never edit anything under src/generated. Fix the failing test.'),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    message('assistant', 'a.ts looks fine; checking b.ts'),
    call('tool-2', 'Read', { file_path: 'src/b.ts' }, fileB),
    result('tool-2', fileB),
    call('tool-3', 'Bash', { command: 'npm test' }, 'FAIL b.test.ts'),
    result('tool-3', 'FAIL b.test.ts: expected 2 to be 3', true),
    message('assistant', 'The failure is in b.test.ts; fixing now.'),
    message('user', 'go ahead'),
  ];
}

type Seen = { state: unknown; questions: string[] };

function fakeJev(answer: (name: string) => number, seen: Seen[] = []): JevAsker {
  return {
    async ask(state, questions: JevQuestions) {
      seen.push({ state, questions: Object.keys(questions) });
      return {
        answers: Object.fromEntries(
          Object.keys(questions).map((key) => [key, { type: 'noul' as const, noul: answer(key) }]),
        ),
      };
    },
  };
}

const fit = {
  maxStateTokens: 25_000,
  preserveRecentMessages: 0,
  goal: 'fix the test',
};

describe('options', () => {
  it('fills in defaults and ignores non-finite values', () => {
    expect(resolveOptions()).toMatchObject({
      keepThreshold: 0.15,
      preserveRecentMessages: 6,
      maxStateTokens: 25_000,
      maxRequestTokens: 60_000,
      truncateHeadChars: 150,
      resultPreviewChars: 300,
      salvageChars: 150,
      maxKeptResultChars: 4000,
    });
    expect(resolveOptions({
      keepThreshold: Number.NaN,
      preserveRecentMessages: 2.7,
      truncateHeadChars: -1.2,
    })).toMatchObject({
      keepThreshold: 0.15,
      preserveRecentMessages: 2,
      truncateHeadChars: 0,
    });
  });
});

describe('token estimate', () => {
  it('charges words, digits and symbols separately and never undercounts JSON badly', () => {
    expect(estimateTokens('')).toBe(0);
    expect(estimateTokens('hello world')).toBe(2);
    expect(estimateTokens('internationalization')).toBe(4);
    expect(estimateTokens('12345678')).toBe(4);
    const json = JSON.stringify({ file_path: '/Users/x/src/a.ts', old_string: 'a = 1;', n: 42 });
    expect(estimateTokens(json)).toBeGreaterThanOrEqual(Math.ceil(json.length / 3));
  });

  it('charges dense alphanumeric runs (hex, UUIDs, base64, hashes) at their denser rate', () => {
    // A mixed letters+digits run (a commit hash) tokenizes far denser than a
    // six-letters-per-token word.
    const hash = 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0';
    expect(estimateTokens(hash)).toBeGreaterThanOrEqual(Math.ceil(hash.length / 4));
    // A long, low-vowel, all-letters run (e.g. a base64/opaque token) is
    // charged the same way even without digits.
    const opaque = 'xkcqzvfmtnbgwplshjdrqzxvbnmkcplsq';
    expect(estimateTokens(opaque)).toBeGreaterThanOrEqual(Math.ceil(opaque.length / 4));
    // An ordinary long English word is unaffected: still one token per six
    // letters, not per four characters.
    expect(estimateTokens('internationalization')).toBe(4);
    // A UUID with hyphens: each hyphen-separated segment is still dense.
    const uuid = 'f47ac10b-58cc-4372-a567-0e02b2c3d479';
    const alnumChars = uuid.replace(/-/g, '').length;
    expect(estimateTokens(uuid)).toBeGreaterThanOrEqual(Math.ceil(alnumChars / 4));
  });
});

describe('truncate', () => {
  it('never cuts a surrogate pair in half', () => {
    const emoji = '\u{1F600}';
    const text = `${'x'.repeat(9)}${emoji}${'z'.repeat(5)}`;
    // limit 11 would cut between the emoji's two code units; back off by one.
    expect(truncate(text, 11)).toBe(`${'x'.repeat(9)}…`);
    // limit 12 lands cleanly after the whole emoji.
    expect(truncate(text, 12)).toBe(`${'x'.repeat(9)}${emoji}…`);
  });
});

describe('tool call collection', () => {
  it('pairs each tool call with its result and pins recent ones', () => {
    const calls = collectToolCalls(transcript(), 3);
    expect(calls.map((c) => [c.id, c.tool, c.callIndex, c.resultIndex, c.pinned])).toEqual([
      ['t1', 'Read', 1, 2, false],
      ['t2', 'Read', 4, 5, false],
      ['t3', 'Bash', 6, 7, true],
    ]);
    expect(calls[2]?.isError).toBe(true);
    expect(calls[0]?.resultChars).toBe(fileA.length);
  });

  it('ignores calls without a result', () => {
    expect(collectToolCalls([message('user', 'hi'), call('x', 'Read', {}, '')], 0)).toHaveLength(0);
  });
});

describe('state fitting', () => {
  it('sends the whole history with tool results replaced by a note', () => {
    const messages = transcript();
    const { state, stage } = fitState(messages, collectToolCalls(messages, 0), fit);
    expect(stage).toBe('full');
    const json = JSON.stringify(state);
    expect(json).not.toContain('export const a = 1;');
    expect(json).toContain('Never edit anything under src/generated');
    expect(json).toContain('go ahead');
    expect(state.history.map((entry) => entry.i)).toEqual([0, 1, 3, 4, 6, 8, 9]);
    expect(state.history[1]?.tool_calls?.[0]).toMatchObject({
      id: 't1',
      tool: 'Read',
      result: `ok, ${fileA.length} chars (omitted)`,
    });
    expect((state.history[4]?.tool_calls?.[0] as HistoryToolCall).result).toMatch(/^error, /);
  });

  it('redacts a credential-shaped tool input field by name, not just by value shape', () => {
    const messages = [
      message('user', 'start'),
      call('cred', 'Bash', { command: 'run-thing', password: 'hunter2', note: 'ok' }, 'ok'),
      result('cred', 'ok'),
    ];
    const { state } = fitState(messages, collectToolCalls(messages, 0), fit);
    const input = (state.history[1]?.tool_calls?.[0] as HistoryToolCall).input;
    expect(input).not.toContain('hunter2');
    expect(input).toContain('[REDACTED]');
    expect(input).toContain('"note":"ok"');
  });

  it('defaults the goal to the latest user prompts', () => {
    const { state } = fitState(transcript(), [], { ...fit, goal: '' });
    expect(state.goal).toContain('Fix the failing test');
    expect(state.goal).toContain('go ahead');
  });

  it('truncates tool inputs before touching message text', () => {
    const messages = [
      message('user', 'start'),
      call('w', 'Write', { file_path: 'x.ts', content: 'lorem ipsum '.repeat(400) }, 'ok'),
      result('w', 'ok'),
      message('assistant', 'written'),
    ];
    const { state, stage, tokens } = fitState(messages, collectToolCalls(messages, 0), {
      ...fit,
      maxStateTokens: 300,
    });
    expect(stage).toBe('inputs<=200');
    expect(tokens).toBeLessThanOrEqual(300);
    expect(state.history[0]?.text).toBe('start');
    expect((state.history[1]?.tool_calls?.[0] as HistoryToolCall).input.length).toBeLessThanOrEqual(200);
  });

  it('shrinks old tool calls to one line each when nothing else is left to cut', () => {
    const messages = [message('user', 'start')];
    for (let i = 0; i < 40; i += 1) {
      messages.push(call(`c${i}`, 'Read', { file_path: `/repo/src/module-${i}.ts` }, 'x'), result(`c${i}`, 'x'));
    }
    messages.push(message('assistant', 'done'));
    const calls = collectToolCalls(messages, 1);
    const full = fitState(messages, calls, { ...fit, preserveRecentMessages: 1 });
    const compacted = fitState(messages, calls, {
      ...fit,
      preserveRecentMessages: 1,
      maxStateTokens: Math.floor(full.tokens * 0.8),
    });
    expect(compacted.stage).toBe('old calls compacted');
    expect(compacted.tokens).toBeLessThanOrEqual(Math.floor(full.tokens * 0.8));
    expect(compacted.tokens).toBeGreaterThanOrEqual(estimateTokens(JSON.stringify(compacted.state)));
    expect(compacted.state.history[1]?.tool_calls?.[0]).toBe(
      't1 Read file_path=/repo/src/module-0.ts → ok 1ch',
    );
    expect(compacted.state.history.at(-1)?.text).toBe('done');

    const merged = fitState(messages, calls, {
      ...fit,
      preserveRecentMessages: 1,
      maxStateTokens: Math.floor(full.tokens * 0.45),
    });
    expect(merged.stage).toBe('old calls merged');
    expect(merged.tokens).toBeLessThanOrEqual(Math.floor(full.tokens * 0.45));
    expect(merged.state.history).toHaveLength(3);
    expect(merged.state.history[1]?.tool_calls).toHaveLength(40);
    expect(merged.state.history[1]?.tool_calls?.[39]).toMatch(/^t40 Read /);
    expect(merged.state.history[0]?.text).toBe('start');
    expect(merged.state.history[2]?.text).toBe('done');
  });

  it('abridges long texts oldest-first and collapses old messages last', () => {
    const long = (n: number) => `${n} ` + 'lorem ipsum '.repeat(300);
    const messages = [
      message('user', long(0)),
      message('assistant', long(1)),
      message('user', long(2)),
      message('assistant', long(3)),
      message('user', 'latest'),
    ];
    const abridged = fitState(messages, [], { ...fit, maxStateTokens: 1800, preserveRecentMessages: 1 });
    expect(abridged.stage).toBe('texts abridged');
    expect(abridged.tokens).toBeLessThanOrEqual(1800);
    expect(abridged.state.history[1]?.text).toContain('chars omitted');
    expect(abridged.state.history[0]?.text).toBe(long(0));
    expect(abridged.state.history[4]?.text).toBe('latest');

    const collapsed = fitState(messages, [], { ...fit, maxStateTokens: 420, preserveRecentMessages: 1 });
    expect(collapsed.stage).toBe('old messages collapsed');
    expect(collapsed.tokens).toBeLessThanOrEqual(420);
    expect(collapsed.state.history[1]?.text).toMatch(/^\[… \d+ chars omitted …\]$/);
    expect(collapsed.state.history[0]?.text).toContain('lorem');
    expect(collapsed.state.history[4]?.text).toBe('latest');
  });

  it('throws when the history cannot be fitted', () => {
    const messages = [message('user', 'a'.repeat(2000)), message('assistant', 'b')];
    expect(() => fitState(messages, [], { ...fit, maxStateTokens: 50 })).toThrow(/too large/);
  });
});

describe('question batching', () => {
  const calls: ToolCall[] = Array.from({ length: 10 }, (_, i) => ({
    id: `t${i + 1}`,
    tool_use_id: `tool-${i + 1}`,
    tool: 'Read',
    input: {},
    callIndex: i * 2 + 1,
    resultIndex: i * 2 + 2,
    resultChars: 100,
    isError: false,
    pinned: false,
  }));
  const options = { maxRequestTokens: 30_000 };

  it('puts everything in one request when it fits', () => {
    expect(batchCalls(calls, 1000, options)).toHaveLength(1);
  });

  it('splits questions across requests when the state leaves little room', () => {
    const batches = batchCalls(calls, 29_600, options);
    expect(batches.length).toBeGreaterThan(1);
    expect(batches.flat().map((c) => c.id)).toEqual(calls.map((c) => c.id));
  });

  it('throws when a single question does not fit', () => {
    expect(() => batchCalls(calls, 29_990, options)).toThrow(/no room/);
  });

  it('caps each request at 128 questions even when the token budget allows more', () => {
    const many = Array.from({ length: 200 }, (_, i) => ({ ...calls[0]!, id: `t${i + 1}`, tool_use_id: `tool-${i + 1}` }));
    const batches = batchCalls(many, 1000, { maxRequestTokens: 60_000 });
    expect(batches.length).toBeGreaterThan(1);
    for (const batch of batches) {
      const questions = batch.reduce((n, call) => n + Object.keys(questionsFor(call)).length, 0);
      expect(questions).toBeLessThanOrEqual(128);
    }
    expect(batches.flat()).toHaveLength(200);
  });
});

describe('decisions', () => {
  const options = { keepThreshold: 0.5 };
  const unpinned = { id: 't1', tool: 'Read', pinned: false };

  it('keeps, drops the result, or drops the call based on the keep probabilities', () => {
    expect(decideCall(unpinned, { keepCall: 0.9, keepResult: 0.7 }, options).action).toBe('keep');
    expect(decideCall(unpinned, { keepCall: 0.9, keepResult: 0.2 }, options).action).toBe('drop_result');
    expect(decideCall(unpinned, { keepCall: 0.01, keepResult: 0.2 }, options).action).toBe('drop_call');
    expect(decideCall({ ...unpinned, pinned: true }, { keepCall: 0, keepResult: 0 }, options)).toMatchObject({
      action: 'keep',
      reason: 'pinned',
    });
  });

  it('gates the call on its own, much lower threshold than the result', () => {
    const gates = { keepThreshold: 0.5, keepCallThreshold: 0.2 };
    // keepCall clears the call gate (0.2) but not the (unused-here) result gate:
    // the call survives with its result truncated, not removed outright.
    expect(decideCall(unpinned, { keepCall: 0.3, keepResult: 0.1 }, gates).action).toBe('drop_result');
    // Below the call gate too: the call goes.
    expect(decideCall(unpinned, { keepCall: 0.1, keepResult: 0.1 }, gates).action).toBe('drop_call');
    // Omitting keepCallThreshold falls back to its own default (0.05), not keepThreshold.
    expect(decideCall(unpinned, { keepCall: 0.1, keepResult: 0.1 }, { keepThreshold: 0.5 }).action).toBe(
      'drop_result',
    );
  });

  it('removes dropped calls and truncates dropped results', () => {
    const messages = transcript();
    messages[4]!.toolUses[0]!.text = 'x'.repeat(2000);
    messages[5]!.toolResults![0]!.text = 'x'.repeat(2000);
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.01, keepResult: 0.1 }, options),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.1 }, options),
      decideCall(calls[2]!, { keepCall: 0.9, keepResult: 0.9 }, options),
    ];
    const kept = applyDecisions(messages, decisions, calls, 300);

    expect(kept.map((m) => m.text || m.toolUses[0]?.tool_use_id || m.toolResults?.[0]?.tool_use_id)).toEqual([
      'Never edit anything under src/generated. Fix the failing test.',
      '[jev-compaction removed the Read call {"file_path":"src/a.ts"}]',
      'a.ts looks fine; checking b.ts',
      'tool-2',
      'tool-2',
      'tool-3',
      'tool-3',
      'The failure is in b.test.ts; fixing now.',
      'go ahead',
    ]);
    expect(kept[0]).toBe(messages[0]);
    expect(kept[3]).not.toBe(messages[4]);
    expect(kept[3]?.toolUses[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[jev-compaction truncated 1700 chars`),
    );
    expect(kept[4]?.toolResults?.[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[jev-compaction truncated 1700 chars`),
    );
    expect(kept[4]).not.toBe(messages[5]);
    expect(kept[5]).toBe(messages[6]);
    expect(kept[6]?.toolResults?.[0]?.text).toContain('expected 2 to be 3');

    const shortMessages = transcript();
    shortMessages[4]!.toolUses[0]!.text = 'y'.repeat(100);
    shortMessages[5]!.toolResults![0]!.text = 'y'.repeat(100);
    const shortKept = applyDecisions(shortMessages, decisions, calls, 300);
    expect(shortKept[3]).toBe(shortMessages[4]);
    expect(shortKept[4]).toBe(shortMessages[5]);
  });

  it('drops an already-empty assistant message but keeps thinking-only text and empty user turns', () => {
    const empty = message('assistant', '');
    const thinking = message('assistant', 'thinking through the next step');
    const emptyUser = message('user', '');
    const messages = [message('user', 'start'), empty, thinking, emptyUser, message('user', 'done')];

    expect(applyDecisions(messages, [], [], 300)).toEqual([
      messages[0],
      thinking,
      emptyUser,
      messages[4],
    ]);
  });

  it('honours truncateHeadChars, including a zero head', () => {
    const messages = transcript();
    const calls = collectToolCalls(messages, 0);
    const decisions = [decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 })];
    const original = messages[2]!.toolResults![0]!.text;
    const total = original.length;

    const kept = applyDecisions(messages, decisions, calls, 50);
    expect(kept[2]?.toolResults?.[0]?.text).toBe(
      `${original.slice(0, 50)}\n[jev-compaction truncated ${total - 50} chars; re-run the tool if needed]`,
    );
    expect(kept[1]?.toolUses[0]?.text).toBe(kept[2]?.toolResults?.[0]?.text);

    const noHead = applyDecisions(messages, decisions, calls, 0);
    expect(noHead[2]?.toolResults?.[0]?.text).toBe(
      `[jev-compaction truncated ${total} chars; re-run the tool if needed]`,
    );
  });

  it('appends salvaged identifiers and errors from the dropped tail when salvageChars is set', () => {
    const messages = [
      message('user', 'run the build'),
      call('a', 'Bash', { command: 'npm run build' }, ''),
      result('a', `${'filler line\n'.repeat(200)}Error: build failed\nsrc/deep/Thing.tsx\n`),
    ];
    const calls = collectToolCalls(messages, 0);
    const decisions = [decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 })];

    const withoutSalvage = applyDecisions(messages, decisions, calls, 50, undefined, 0);
    expect(withoutSalvage[2]?.toolResults?.[0]?.text).not.toContain('src/deep/Thing.tsx');

    const withSalvage = applyDecisions(messages, decisions, calls, 50, undefined, 600);
    const text = withSalvage[2]?.toolResults?.[0]?.text ?? '';
    expect(text).toContain('src/deep/Thing.tsx');
    expect(text).toContain('Error: build failed');
  });

  it('never cuts the head of a dropped result inside a surrogate pair', () => {
    const emoji = '\u{1F600}';
    const note = (omitted: number): string =>
      `[jev-compaction truncated ${omitted} chars; re-run the tool if needed]`;
    const truncated = (text: string): string | undefined => {
      const messages = transcript();
      messages[1]!.toolUses[0]!.text = text;
      messages[2]!.toolResults![0]!.text = text;
      const calls = collectToolCalls(messages, 0);
      const decisions = [decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 })];
      const kept = applyDecisions(messages, decisions, calls, 300);
      expect(kept[1]?.toolUses[0]?.text).toBe(kept[2]?.toolResults?.[0]?.text);
      return kept[2]?.toolResults?.[0]?.text;
    };

    const straddling = `${'x'.repeat(299)}${emoji.repeat(200)}`;
    expect(truncated(straddling)).toBe(`${'x'.repeat(299)}\n${note(straddling.length - 299)}`);

    const aligned = `${'x'.repeat(298)}${emoji.repeat(200)}`;
    expect(truncated(aligned)).toBe(`${'x'.repeat(298)}${emoji}\n${note(aligned.length - 300)}`);
  });
});

describe('compact', () => {
  it('resends the full state with every batch and merges the answers', async () => {
    const seen: Seen[] = [];
    const messages = transcript();
    const calls = collectToolCalls(messages, 1);
    const stateTokens = fitState(messages, calls, {
      ...fit,
      goal: '',
      preserveRecentMessages: 1,
    }).tokens;
    // Room for exactly one call's questions, so every call gets its own batch.
    const perCall = Math.max(
      ...calls.map((call) => estimateTokens(JSON.stringify(questionsFor(call)))),
    );
    const output = await compact(
      messages,
      fakeJev((name) => (name.startsWith('call_') ? 0.9 : 0.1), seen),
      {
        preserveRecentMessages: 1,
        maxRequestTokens: stateTokens + perCall + 25,
        resultPreviewChars: 0,
      },
    );

    expect(output.stats.requests).toBe(seen.length);
    expect(seen.length).toBeGreaterThan(1);
    expect(seen.flatMap((r) => r.questions).sort()).toEqual([
      'call_t1',
      'call_t2',
      'call_t3',
      'result_t1',
      'result_t2',
      'result_t3',
    ]);
    expect(new Set(seen.map((r) => JSON.stringify(r.state))).size).toBe(1);
    expect(output.decisions.map((d) => d.action)).toEqual(['drop_result', 'drop_result', 'drop_result']);
    expect(output.messages).toHaveLength(messages.length);
    expect(output.stats).toMatchObject({ resultsDropped: 3, kept: 0, callsDropped: 0, pinned: 0 });
    expect(reductionRatio(output)).toBeGreaterThan(0);
  });

  it('keeps everything without calling Jev when no tool call is a candidate', async () => {
    const seen: Seen[] = [];
    const messages = [message('user', 'hello'), message('assistant', 'hi')];
    const output = await compact(messages, fakeJev(() => 0, seen));
    expect(seen).toHaveLength(0);
    expect(output.stats).toMatchObject({ requests: 0, stateStage: '', calls: 0 });
    expect(output.messages).toEqual(messages);
  });

  it('reports a tiny reduction when Jev wants everything kept', async () => {
    const output = await compact(transcript(), fakeJev(() => 0.95), { preserveRecentMessages: 1 });
    expect(output.decisions.every((d) => d.action === 'keep')).toBe(true);
    expect(reductionRatio(output)).toBe(0);
  });

  it('rejects malformed answers', async () => {
    const broken: JevAsker = {
      ask: async () => ({ answers: { call_t1: { noul: 0.5 } } }),
    };
    await expect(compact(transcript(), broken, { preserveRecentMessages: 1 })).rejects.toThrow(
      /Invalid Jev answer/,
    );
  });

  it.each([
    ['call_t1', -0.1],
    ['call_t1', 1.1],
    ['result_t1', -0.1],
    ['result_t1', 1.1],
  ])('rejects an out-of-range probability for %s: %s', async (name, probability) => {
    await expect(
      compact(transcript(), fakeJev((key) => (key === name ? probability : 0.5)), {
        preserveRecentMessages: 1,
      }),
    ).rejects.toThrow(`Invalid Jev answer for ${name}`);
  });

  it('rejects an out-of-range keepThreshold', () => {
    expect(() => resolveOptions({ keepThreshold: 1.5 })).toThrow(RangeError);
    expect(() => resolveOptions({ keepThreshold: -0.1 })).toThrow(RangeError);
  });

  it('rejects an array-shaped answers map', () => {
    expect(() => parseJevResponse(200, true, '{"answers":[]}')).toThrow(/missing answers/);
  });

  it('leaves calls that share a tool_use_id out of the candidates', () => {
    const messages = [
      call('dup', 'Read', { file_path: 'a.ts' }, ''),
      result('dup', 'a'),
      call('dup', 'Read', { file_path: 'b.ts' }, ''),
      result('dup', 'b'),
      call('ok', 'Read', { file_path: 'c.ts' }, ''),
      result('ok', 'c'),
    ];
    expect(collectToolCalls(messages, 0).map((c) => c.tool_use_id)).toEqual(['ok']);
  });

  it('leaves a call with a duplicate tool_result out of the candidates', () => {
    const messages = [call('dup', 'Read', { file_path: 'a.ts' }, ''), result('dup', 'a'), result('dup', 'b')];
    expect(collectToolCalls(messages, 0)).toEqual([]);
  });

  it('rejects a tool_result that precedes its call', () => {
    const messages = [result('t1', 'too early'), call('t1', 'Read', {}, '')];
    expect(() => collectToolCalls(messages, 0)).toThrow(/precedes its call/);
  });

  it('carries an unpaired tool_use as a pending call, never a deletion candidate', () => {
    const messages = [...transcript(), call('tool-4', 'Bash', { command: 'still running' }, '')];
    const calls = collectToolCalls(messages, 0);
    expect(calls.some((c) => c.tool_use_id === 'tool-4')).toBe(false);
    const state = fitState(messages, calls, fit);
    const entry = state.state.history.find((e) => e.pending_calls?.some((p) => p.tool_use_id === 'tool-4'));
    expect(entry?.pending_calls).toContainEqual(
      expect.objectContaining({ tool_use_id: 'tool-4', tool: 'Bash' }),
    );
  });

  it('runs batches over a bounded pool of concurrent requests', async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    const messages = transcript();
    const stateTokens = fitState(messages, collectToolCalls(messages, 1), {
      ...fit,
      goal: '',
      preserveRecentMessages: 1,
    }).tokens;
    const perCall = Math.max(
      ...collectToolCalls(messages, 1).map((c) => estimateTokens(JSON.stringify(questionsFor(c)))),
    );
    const asker: JevAsker = {
      async ask(_state, questions) {
        inFlight += 1;
        maxInFlight = Math.max(maxInFlight, inFlight);
        await new Promise((resolve) => setTimeout(resolve, 5));
        inFlight -= 1;
        return {
          answers: Object.fromEntries(
            Object.keys(questions).map((key) => [key, { type: 'noul' as const, noul: 0.1 }]),
          ),
        };
      },
    };
    const output = await compact(messages, asker, {
      preserveRecentMessages: 1,
      maxRequestTokens: stateTokens + perCall + 25,
      maxConcurrentRequests: 2,
    });
    expect(output.stats.requests).toBeGreaterThan(2);
    expect(maxInFlight).toBeLessThanOrEqual(2);
  });

  it('salvages the answers of the batches that succeeded when only some fail', async () => {
    const messages = transcript();
    const stateTokens = fitState(messages, collectToolCalls(messages, 1), {
      ...fit,
      goal: '',
      preserveRecentMessages: 1,
    }).tokens;
    const perCall = Math.max(
      ...collectToolCalls(messages, 1).map((c) => estimateTokens(JSON.stringify(questionsFor(c)))),
    );
    let call = 0;
    const asker: JevAsker = {
      async ask(_state, questions) {
        call += 1;
        if (call === 1) throw new Error('Jev request failed (503): boom');
        return {
          answers: Object.fromEntries(
            Object.keys(questions).map((key) => [key, { type: 'noul' as const, noul: 0.9 }]),
          ),
        };
      },
    };
    const output = await compact(messages, asker, {
      preserveRecentMessages: 1,
      maxRequestTokens: stateTokens + perCall + 25,
      maxConcurrentRequests: 1,
    });
    expect(output.stats.failedRequests).toBe(1);
    expect(output.stats.requests).toBeGreaterThan(1);
    // The failed batch's calls fall back to the safe default (kept).
    expect(output.decisions.some((d) => d.action === 'keep')).toBe(true);
  });

  it('throws when every batch fails, naming the underlying failure', async () => {
    const asker: JevAsker = {
      ask: async () => {
        throw new Error('Jev request failed (500): boom');
      },
    };
    await expect(
      compact(transcript(), asker, { preserveRecentMessages: 1 }),
    ).rejects.toThrow(/boom/);
  });

  it('reserves room for the largest question before fitting the state', async () => {
    const input = [
      message('user', 'start'),
      call('x', 'Read', { file_path: 'a' }, 'x'),
      result('x', 'x'),
      message('assistant', 'word '.repeat(2000)),
      message('user', 'finish'),
    ];
    let seenState = '';
    const output = await compact(
      input,
      {
        async ask(state, questions) {
          seenState = JSON.stringify(state);
          return {
            answers: Object.fromEntries(
              Object.keys(questions).map((key) => [key, { type: 'noul' as const, noul: 1 }]),
            ),
          };
        },
      },
      { maxRequestTokens: 1000, maxStateTokens: 25_000, preserveRecentMessages: 1, goal: 'finish' },
    );
    expect(estimateTokens(seenState)).toBeLessThan(1000);
    expect(output.stats.stateStage).not.toBe('full');
    expect(output.messages).toHaveLength(input.length);
  });

  it('throws instead of sending anything when the request budget cannot fit state and questions', async () => {
    const asker: JevAsker = {
      ask: async () => {
        throw new Error('unexpected');
      },
    };
    await expect(
      compact(transcript(), asker, { maxRequestTokens: 1, preserveRecentMessages: 0 }),
    ).rejects.toThrow(/no room/);
  });
});

describe('HTTP client', () => {
  it('builds a System One request', () => {
    const request = buildJevRequest({ apiKey: 'k' }, { a: 1 }, {
      q: { type: 'noul', instructions: 'x' },
    });
    expect(request.url).toBe('https://api.typesafe.ai/v1/systemone');
    expect(request.headers.authorization).toBe('Bearer k');
    expect(JSON.parse(request.body)).toEqual({
      model: 'jev-latest',
      state: { a: 1 },
      questions: { q: { type: 'noul', instructions: 'x' } },
    });
  });

  it('explains a 403 HTML firewall response without echoing the page', () => {
    expect(() => parseJevResponse(403, false, '<!DOCTYPE html><html>blocked</html>')).toThrow(
      'Jev request failed (403): the endpoint returned an HTML error page; a web firewall or proxy may be blocking this request',
    );
  });

  it('rejects failed and malformed responses', () => {
    expect(() => parseJevResponse(500, false, 'boom')).toThrow(/500/);
    expect(() => parseJevResponse(200, true, 'not json')).toThrow(/malformed/);
    expect(() => parseJevResponse(200, true, '{}')).toThrow(/missing answers/);
    expect(parseJevResponse(200, true, '{"answers":{}}')).toEqual({ answers: {} });
  });

  it('asks over fetch and refuses to run without a key', async () => {
    const bodies: string[] = [];
    const client = new JevClient({
      apiKey: 'k',
      model: 'jev-test',
      fetch: (async (_url: string | URL | Request, init?: RequestInit) => {
        bodies.push(String(init?.body));
        return new Response(JSON.stringify({ answers: { q: { noul: 0.4 } } }), { status: 200 });
      }) as typeof fetch,
    });
    const response = await client.ask('state', { q: { type: 'noul', instructions: 'x' } });
    expect(response.answers.q).toEqual({ noul: 0.4 });
    expect(JSON.parse(bodies[0]!).model).toBe('jev-test');

    const keyless = new JevClient({ apiKey: '' });
    await expect(keyless.ask('s', {})).rejects.toThrow(/TYPESAFE_API_KEY is not configured/);
    await expect(
      compactMessages(transcript(), { apiKey: '', preserveRecentMessages: 1, ...LEGACY }),
    ).rejects.toThrow(/TYPESAFE_API_KEY is not configured/);
  });

  it('reads TYPESAFE_API_KEY from the environment', async () => {
    const original = { ts: process.env.TYPESAFE_API_KEY };
    try {
      process.env.TYPESAFE_API_KEY = 'typesafe-key';
      const bodies: Array<Record<string, string>> = [];
      const fetchFn = (async (_url: string | URL | Request, init?: RequestInit) => {
        bodies.push(init?.headers as Record<string, string>);
        return new Response(JSON.stringify({ answers: {} }), { status: 200 });
      }) as typeof fetch;
      await new JevClient({ fetch: fetchFn }).ask('s', {});
      expect(bodies[0]!.authorization).toBe('Bearer typesafe-key');

      process.env.TYPESAFE_API_KEY = '';
      await expect(new JevClient({ fetch: fetchFn }).ask('s', {})).rejects.toThrow(
        /TYPESAFE_API_KEY is not configured/,
      );
    } finally {
      if (original.ts === undefined) delete process.env.TYPESAFE_API_KEY;
      else process.env.TYPESAFE_API_KEY = original.ts;
    }
  });

  it('times out a hung request and retries once on a 429, honoring retry-after', async () => {
    vi.useFakeTimers();
    try {
      let calls = 0;
      const client = new JevClient({
        apiKey: 'k',
        timeoutMs: 50,
        fetch: (async (_url: string | URL | Request, init?: RequestInit) => {
          calls += 1;
          if (calls === 1) {
            return new Response('{"error":"rate limited"}', {
              status: 429,
              headers: { 'retry-after': '0' },
            });
          }
          return new Response(JSON.stringify({ answers: { q: { noul: 0.7 } } }), { status: 200 });
        }) as typeof fetch,
      });
      const response = await client.ask('s', { q: { type: 'noul', instructions: 'x' } });
      expect(response.answers.q).toEqual({ noul: 0.7 });
      expect(calls).toBe(2);
    } finally {
      vi.useRealTimers();
    }
  });

  it('aborts a hung request once the deadline passes', async () => {
    const client = new JevClient({
      apiKey: 'k',
      timeoutMs: 10,
      fetch: (async (_url: string | URL | Request, init?: RequestInit) => {
        await new Promise((resolve, reject) => {
          init?.signal?.addEventListener('abort', () => reject(init.signal!.reason));
        });
        throw new Error('unreachable');
      }) as typeof fetch,
    });
    await expect(client.ask('s', { q: { type: 'noul', instructions: 'x' } })).rejects.toThrow(
      /timed out/,
    );
  });
});

describe('jev context limits', () => {
  const call = (id: string): ToolCall => ({
    id,
    tool_use_id: `toolu_${id}`,
    tool: 'Read',
    input: { file_path: 'src/a.ts' },
    callIndex: 1,
    resultIndex: 2,
    resultChars: 100,
    isError: false,
    pinned: false,
  });

  it('uses the whole 64k request budget, not 32k', () => {
    const calls = Array.from({ length: 60 }, (_, i) => call(`t${i + 1}`));
    // One request under the real 64k limit, several under the old 30k ceiling.
    expect(batchCalls(calls, 25_000, { maxRequestTokens: 60_000 })).toHaveLength(1);
    expect(
      batchCalls(calls, 25_000, { maxRequestTokens: 30_000 }).length,
    ).toBeGreaterThan(2);
  });

  it('rejects a state that leaves no room for a single question', () => {
    expect(() => batchCalls([call('t1')], 31_990, { maxRequestTokens: 60_000 })).toThrow(
      /state plus one question/,
    );
  });
});

describe('keep threshold', () => {
  const call = { id: 't1', tool: 'Read', pinned: false };

  it('keeps a call Jev is unsure about', () => {
    const decision = decideCall(call, { keepCall: 0.5, keepResult: 0.5 }, resolveOptions());
    expect(decision.action).toBe('keep');
  });

  it('drops only when Jev is confident the call is spent', () => {
    const options = resolveOptions();
    expect(decideCall(call, { keepCall: 0.2, keepResult: 0.05 }, options).action).toBe(
      'drop_result',
    );
    expect(decideCall(call, { keepCall: 0.01, keepResult: 0.03 }, options).action).toBe(
      'drop_call',
    );
  });
});

describe('goalFromMessages', () => {
  it('excludes host-written text from the default goal', () => {
    const messages = [
      message('user', 'fix the flaky test'),
      message('user', '<system-reminder>context about the repo</system-reminder>'),
      message('user', '<command-name>compact</command-name>'),
      message('user', 'also check the retry logic'),
    ];
    expect(isMachineText(messages[1]!.text)).toBe(true);
    expect(isMachineText(messages[2]!.text)).toBe(true);
    expect(isMachineText(messages[0]!.text)).toBe(false);
    expect(goalFromMessages(messages)).toBe('fix the flaky test\nalso check the retry logic');
  });
});

describe('result preview', () => {
  it('quotes the head and tail of a result in its keep question', () => {
    const long = `START ${'x '.repeat(500)}END`;
    const messages = [message('user', 'hi'), call('a', 'Bash', { command: 'ls' }, ''), result('a', long)];
    const [withPreview] = collectToolCalls(messages, 0, 20);
    expect(withPreview!.resultPreview).toMatch(/^START .* … .*END$/);
    const q = Object.values(questionsFor(withPreview!))[1]!.instructions;
    expect(q).toContain('Output preview: START');
    expect(q).toContain('END');
  });

  it('adds nothing when disabled', () => {
    const messages = [message('user', 'hi'), call('a', 'Bash', { command: 'ls' }, ''), result('a', 'plain output')];
    const [plain] = collectToolCalls(messages, 0);
    expect(plain!.resultPreview).toBeUndefined();
    expect(Object.values(questionsFor(plain!))[1]!.instructions).not.toContain('preview');
  });
});

describe('request body', () => {
  it('replaces lone surrogates from raw tool output and keeps real pairs', () => {
    const request = buildJevRequest({ apiKey: 'k' }, { output: 'a\uD83D b \uDE00 c 😀' } as never, {});
    expect(request.body).not.toMatch(/\\ud[89a-f][0-9a-f]{2}/i);
    expect(JSON.parse(request.body).state.output).toBe('a� b � c 😀');
  });
});

describe('windowed states when the history does not fit', () => {
  function long(n: number): Message[] {
    const out: Message[] = [message('user', 'Fix the failing build; never touch src/generated.')];
    for (let i = 0; i < n; i++) {
      out.push(message('assistant', `Step ${i}: ${'checking the next module and its callers '.repeat(6)}`));
      out.push(call(`w${i}`, 'Bash', { command: `npm test -- module${i}` }, ''));
      out.push(result(`w${i}`, `module${i}: 3 passed, 1 failed (exit code 1)\n${'x'.repeat(400)}`));
    }
    out.push(message('user', 'go on'));
    return out;
  }
  const options = { maxStateTokens: 3_000, maxRequestTokens: 6_000, preserveRecentMessages: 2 };

  it('splits into windows that each fit, and every candidate is asked', async () => {
    const messages = long(400);
    const calls = collectToolCalls(messages, 2);
    expect(() => fitState(messages, calls, resolveOptions(options))).toThrow(/history too large/);
    const seen: Seen[] = [];
    const out = await compact(messages, fakeJev(() => 0.1, seen), options);
    expect(out.stats.stateStage).toMatch(/^windows:\d+$/);
    expect(out.stats.stateTokens).toBeLessThanOrEqual(options.maxStateTokens);
    const asked = new Set(seen.flatMap((s) => s.questions).filter((q) => q.startsWith('call_')));
    expect(asked.size).toBe(calls.filter((c) => !c.pinned).length);
    expect(out.stats.requests).toBe(seen.length);
    // Calls inside a window keep their results: none reads as never returned.
    expect(JSON.stringify(seen.map((s) => s.state))).not.toContain('pending_calls');
  });

  it('keeps a call that no window can fit, instead of throwing', async () => {
    const messages = long(5);
    const seen: Seen[] = [];
    const out = await compact(messages, fakeJev(() => 0.1, seen), { ...options, maxStateTokens: 200, goal: 'g'.repeat(5_000) });
    expect(seen.length).toBe(0);
    expect(out.stats.stateStage).toMatch(/unasked:\d+/);
    expect(out.decisions.every((d) => d.action === 'keep')).toBe(true);
    expect(out.messages).toEqual(messages);
  });
});
