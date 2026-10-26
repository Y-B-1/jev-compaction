import { describe, expect, test } from 'vitest';

import { ERROR_LINE, SCAN } from '../src/salvage.js';
import { elideBody, heredocBodies, reassemble, splitProse } from '../src/prose.js';
import { applyProse, callActions, ladderRescue, scoreProse, type ProseTrimOptions } from '../src/prose.js';
import { messageChars } from '../src/compact.js';
import { collectToolCalls } from '../src/state.js';
import type { CallAction, CallDecision, CompactResult, JevAsker, JevQuestions, Message, ToolUse } from '../src/types.js';

/** One entry per unit: `c` scored, `k` kept, then the unit's text without its separators. */
function view(text: string): string[] {
  return splitProse(text).map((u) => `${u.scored ? 'c' : 'k'} ${text.slice(u.start, u.end).trim()}`);
}

describe('splitProse: design table', () => {
  test('opener, then a clause split before "and"', () => {
    expect(view('Okay, let me take a look at this, and see what I can figure out.')).toEqual([
      'c Okay,',
      'c let me take a look at this,',
      'c and see what I can figure out.',
    ]);
  });

  test('a negation binds the whole sentence', () => {
    expect(view("Don't run the migration on prod, it drops the users table.")).toEqual([
      "c Don't run the migration on prod, it drops the users table.",
    ]);
  });

  test('inline code keeps the whole sentence', () => {
    expect(view('Run the build, but only after `npm ci` finishes.')).toEqual([
      'k Run the build, but only after `npm ci` finishes.',
    ]);
  });

  test('opener split, bound remainder scored whole', () => {
    expect(view('Great question; the cache is stale because the TTL never resets.')).toEqual([
      'c Great question;',
      'c the cache is stale because the TTL never resets.',
    ]);
  });
});

describe('splitProse: protected spans', () => {
  // [name, text, needle]: every unit that holds the needle must be kept unscored.
  const kept: [string, string, string][] = [
    ['P1 fenced code', 'Intro here today.\n```\nplain words in a fence\n```\nOutro here today.', 'plain words'],
    ['P1 tilde fence', '~~~\nplain words in a fence\n~~~', 'plain words'],
    ['P1 unclosed fence', 'Intro here today.\n```\nplain words in a fence\nmore plain words', 'more plain'],
    ['P2 indented line', 'Intro here today.\n    plain words indented here', 'plain words'],
    ['P2 tab line', 'Intro here today.\n\tplain words tabbed here', 'plain words'],
    ['P3 inline code', 'Please look at `this` now.', 'look at'],
    ['P4 table row', '| plain | words |', 'plain'],
    ['P4 heading', '# Plain heading words', 'Plain'],
    ['P5 url', 'Please open https://example.com/page now.', 'open'],
    ['P5 posix path', 'Please open src/app/main.ts now.', 'open'],
    ['P5 windows path', 'Please open C:\\repos\\app\\main.ts now.', 'open'],
    ['P5 name=value', 'Please set NODE_ENV=production now.', 'set'],
    ['P6 number', 'We saw 42 users today.', 'saw'],
    ['P6 version', 'We shipped v0.6.1 today.', 'shipped'],
    ['P6 id', 'We closed #88 today.', 'closed'],
    ['P7 dot', 'Please read req.session now.', 'read'],
    ['P7 underscore', 'Please read my_value now.', 'read'],
    ['P7 double colon', 'Please read Foo::bar now.', 'read'],
    ['P7 camel', 'Please add the cookieParser middleware now.', 'add'],
    ['P8 command line', '$ ls the plain folder', 'plain'],
    ['P8 percent command', '% ls the plain folder', 'plain'],
    ['P8 flag', 'Please use --force carefully now.', 'use'],
    ['P8 short flag', 'Please use -rf carefully now.', 'use'],
    ['P9 error line', 'The plain build failed here today.', 'plain'],
    ['P9 quoted', 'Please say "hello there" now.', 'say'],
    ['marker line', '[jev-compaction elided 5 chars of Write.content for a.ts; Read the file if needed]', 'elided'],
  ];

  test.each(kept)('%s is kept unscored', (_name, text, needle) => {
    const hits = splitProse(text).filter((u) => text.slice(u.start, u.end).includes(needle));
    expect(hits.length).toBeGreaterThan(0);
    for (const u of hits) expect(u.scored).toBe(false);
  });

  test('an acronym is scored, a camelCase identifier is kept', () => {
    expect(view('Please check the TTL value again.')[0]).toMatch(/^c /);
    expect(view('Please check the cookieParser value again.')[0]).toMatch(/^k /);
  });

  test('a short apostrophe pair is not a quote', () => {
    expect(view("We didn't know it, and they can't say.")[0]).toMatch(/^c /);
  });
});

describe('splitProse: sentences and clauses', () => {
  test.each(['e.g.', 'i.e.', 'etc.', 'vs.', 'cf.'])('no split after %s', (abbr) => {
    const text = `Pick one ${abbr} the first choice here today.`;
    expect(splitProse(text)).toHaveLength(1);
  });

  test('sentences split at a terminal mark', () => {
    expect(view('The cache is stale. The queue is empty!')).toEqual(['c The cache is stale.', 'c The queue is empty!']);
  });

  test.each([
    ['condition', 'Some long lead text goes here, if the cache is stale, and then more words follow.'],
    ['contrast', 'Some long lead text goes here, and more words follow, but fewer remain now.'],
    ['obligation', 'Some long lead text goes here, and then you must restart, and more follow.'],
    ['cause', 'Some long lead text goes here, and then it broke since the cache was cold.'],
  ])('bound word class %s scores whole', (_name, text) => {
    expect(splitProse(text)).toHaveLength(1);
  });

  test('a clause under 3 words merges into its neighbour', () => {
    expect(view('Take a look at the cache layout today, and go.')).toEqual(['c Take a look at the cache layout today, and go.']);
  });

  test('a semicolon splits unbound sentences', () => {
    expect(view('Take a look at the cache layout; then look at the queue layout.')).toEqual([
      'c Take a look at the cache layout;',
      'c then look at the queue layout.',
    ]);
  });

  test('a spaced dash splits unbound sentences', () => {
    expect(view('Take a look at the cache layout - then look at the queue layout.')).toEqual([
      'c Take a look at the cache layout -',
      'c then look at the queue layout.',
    ]);
  });

  test('every byte belongs to exactly one unit, in order', () => {
    const text = '\nOkay, let me look at this, and see.\n\n- first item here. second item here.\n```\ncode\n```\nTail words here today.\n';
    const units = splitProse(text);
    expect(units[0]?.start).toBe(0);
    expect(units.at(-1)?.end).toBe(text.length);
    for (let i = 1; i < units.length; i++) expect(units[i]?.start).toBe(units[i - 1]?.end);
    expect(units.map((u) => text.slice(u.start, u.end)).join('')).toBe(text);
  });

  test('blank text has no units', () => {
    expect(splitProse('')).toEqual([]);
    expect(splitProse('  \n\n ')).toEqual([]);
  });
});

describe('reassemble', () => {
  const dropScored = () => false;

  test('keeping everything returns the text unchanged', () => {
    const text = 'Okay, look here. Then more words follow.\n\n\n\nEnd.';
    expect(reassemble(text, splitProse(text), () => true)).toBe(text);
  });

  test('kept units are byte-identical and in order', () => {
    const text = 'Alpha beta gamma here. Delta epsilon zeta here. Eta theta iota here.';
    const units = splitProse(text);
    const out = reassemble(text, units, (u) => units.indexOf(u) !== 1);
    expect(out).toBe('Alpha beta gamma here. Eta theta iota here.');
  });

  test('a cut-edge comma takes the sentence end mark', () => {
    const text = 'Okay, let me take a look at this, and see what I can figure out.';
    const units = splitProse(text);
    expect(reassemble(text, units, (u) => units.indexOf(u) < 2)).toBe('Okay, let me take a look at this.');
  });

  test('a cut-edge semicolon takes the sentence end mark', () => {
    const text = 'Take a look at the cache layout; then look at the queue layout!';
    const units = splitProse(text);
    expect(reassemble(text, units, (u) => units.indexOf(u) < 1)).toBe('Take a look at the cache layout!');
  });

  test('a comma followed by a kept clause is left alone', () => {
    const text = 'Okay, let me take a look at this, and see what I can figure out.';
    const units = splitProse(text);
    expect(reassemble(text, units, (u) => units.indexOf(u) !== 1)).toBe('Okay, and see what I can figure out.');
  });

  test('an emptied list block loses its marker, and the line break goes with it', () => {
    const text = 'Intro words here today.\n- drop this whole item.\n- keep `this` item.';
    const units = splitProse(text);
    const out = reassemble(text, units, (u) => !u.scored || text.slice(u.start, u.end).includes('keep'));
    expect(out).toBe('- keep `this` item.');
  });

  test('a list item with a kept later sentence keeps its marker', () => {
    const text = '- first words here today. second words here today.';
    const units = splitProse(text);
    expect(reassemble(text, units, (u) => units.indexOf(u) === 1)).toBe('- second words here today.');
  });

  test('3 or more newlines collapse to 2 after a drop', () => {
    const text = 'Keep `one` here.\n\nDrop this paragraph here.\n\nKeep `two` here.';
    const units = splitProse(text);
    expect(reassemble(text, units, (u) => !u.scored)).toBe('Keep `one` here.\n\nKeep `two` here.');
    const gappy = 'Keep `one` here.\n\n\nDrop this paragraph here.\n\n\n\nKeep `two` here.';
    expect(reassemble(gappy, splitProse(gappy), (u) => !u.scored)).toBe('Keep `one` here.\n\nKeep `two` here.');
  });

  test('no trailing space is left before a line break', () => {
    const text = 'Keep `one` here. Drop this sentence here.\nNext line `kept` here.';
    expect(reassemble(text, splitProse(text), (u) => !u.scored)).toBe('Keep `one` here.\nNext line `kept` here.');
  });

  test('dropping everything yields an empty string', () => {
    const text = 'Drop this sentence here. And this one too.';
    expect(reassemble(text, splitProse(text), dropScored)).toBe('');
  });
});

describe('heredocBodies', () => {
  const cases: [string, string, string][] = [
    ['<<EOF', 'cat > a.txt <<EOF\nline one\nline two\nEOF\n', 'EOF'],
    ["<<'EOF'", "cat > a.txt <<'EOF'\nline one\nline two\nEOF\n", 'EOF'],
    ['<<"EOF"', 'cat > a.txt <<"EOF"\nline one\nline two\nEOF\n', 'EOF'],
    ['<<-EOF', 'cat > a.txt <<-EOF\n\tline one\n\tline two\n\tEOF\n', '\tEOF'],
  ];

  test.each(cases)('%s: command line and closing marker lie outside the body', (_name, command, closing) => {
    const spans = heredocBodies(command);
    expect(spans).toHaveLength(1);
    const { start, end } = spans[0] as { start: number; end: number };
    expect(command.slice(start, end)).toMatch(/^\t?line one\n\t?line two\n$/);
    expect(command.slice(0, start)).toMatch(/<<-?['"]?EOF['"]?\n$/);
    expect(command.slice(end)).toBe(`${closing}\n`);
  });

  test('a command without a heredoc has none', () => {
    expect(heredocBodies('echo hi && cat <<< "here string"')).toEqual([]);
  });

  test('an unclosed heredoc runs to the end', () => {
    const command = 'cat <<EOF\nline one\nline two';
    const [span] = heredocBodies(command);
    expect(command.slice(span?.start, span?.end)).toBe('line one\nline two');
  });

  test('two heredocs on one line get two bodies in order', () => {
    const command = 'diff <(cat <<A\nfirst\nA\n) <(cat <<B\nsecond\nB\n)';
    const bodies = heredocBodies(command).map((s) => command.slice(s.start, s.end));
    expect(bodies).toEqual(['first\n', 'second\n']);
  });
});

describe('elideBody', () => {
  const label = 'Write.content for src/app.ts';

  test('returns the text itself when it is not past head + 120', () => {
    const text = 'x'.repeat(420);
    expect(elideBody(text, label, 300, 0)).toBe(text);
  });

  test('keeps a byte-identical head and a note naming tool, field, path and length', () => {
    const text = `${'a'.repeat(300)}${'b'.repeat(12000)}`;
    const out = elideBody(text, label, 300, 0);
    expect(out.startsWith(`${'a'.repeat(300)}\n`)).toBe(true);
    expect(out).toContain('[jev-compaction elided 12,000 chars of Write.content for src/app.ts; Read the file if needed]');
    expect(out.length).toBeLessThan(text.length);
  });

  test('never splits a surrogate pair at the cut', () => {
    const text = `${'a'.repeat(299)}\u{1F600}${'b'.repeat(1000)}`;
    const out = elideBody(text, label, 300, 0);
    expect(out.startsWith(`${'a'.repeat(299)}\n`)).toBe(true);
    expect(out).not.toMatch(/[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]/);
  });

  test('salvages a path from the cut tail', () => {
    const text = `${'a'.repeat(300)}${'filler words\n'.repeat(60)}see src/deep/file.ts for more\n`;
    expect(elideBody(text, label, 300, 200)).toContain('src/deep/file.ts');
  });

  test('returns the original when the output would not be shorter', () => {
    const text = 'a'.repeat(425);
    expect(elideBody(text, 'L'.repeat(400), 300, 0)).toBe(text);
  });
});

describe('salvage exports', () => {
  test('SCAN and ERROR_LINE are usable through the public module', () => {
    expect([...'see src/a/b.ts'.matchAll(new RegExp(SCAN.source, SCAN.flags))]).toHaveLength(1);
    expect(ERROR_LINE.test('boom: fatal')).toBe(true);
  });
});

describe('T4 repair: line breaks and bound remainders', () => {
  const dropAt = (text: string, n: number): string => {
    const units = splitProse(text);
    return reassemble(text, units, (u) => units.indexOf(u) !== n);
  };

  test.each([
    ['single newline', 'Keep `one` here.\nDrop this sentence here. Keep `two` here.', 'Keep `one` here.\nKeep `two` here.'],
    ['paragraph break', 'Keep `one` here.\n\nDrop this sentence here. Keep `two` here.', 'Keep `one` here.\n\nKeep `two` here.'],
    ['indented line', 'Keep `one` here.\n  Drop this sentence here. Keep `two` here.', 'Keep `one` here.\n  Keep `two` here.'],
  ])('a dropped first unit passes its %s to the next kept unit', (_name, text, expected) => {
    expect(dropAt(text, 1)).toBe(expected);
  });

  test.each([
    'Deploy now, but not on prod.',
    'Run it on staging, unless it is Friday.',
    'Restart the server, only after the backup.',
    'Use the cache, never the database.',
  ])('a bound remainder keeps %j as one unit', (text) => {
    expect(view(text)).toHaveLength(1);
  });

  test('an opener before a plain remainder still splits', () => {
    expect(view('Great question; the cache is stale because the TTL never resets.')).toHaveLength(2);
  });
});

describe('scoreProse + applyProse (Jev trim stage)', () => {
  const trimProse = async (...args: Parameters<typeof scoreProse>) => applyProse(await scoreProse(...args), 0.15);

  const FLUFF = 'Let me have a good look at the layout of this module now.';
  const FACT = 'The cache layer returns stale entries after a restart.';
  const OPTS: ProseTrimOptions = {
    preserveRecentMessages: 2,
    truncateHeadChars: 300,
    salvageChars: 300,
    maxRequestTokens: 60_000,
    goal: 'fix the stale cache',
  };

  type Req = { state: { window: Record<string, Record<string, unknown>> }; questions: JevQuestions };
  type Answer = (id: string, req: Req) => number | undefined | 'throw';

  function asker(answer: Answer, seen: Req[] = []): JevAsker {
    return {
      async ask(state, questions) {
        const req = { state, questions } as Req;
        seen.push(req);
        const answers: Record<string, { type: 'noul'; noul: number }> = {};
        for (const id of Object.keys(questions)) {
          const a = answer(id, req);
          if (a === 'throw') throw new Error('boom');
          if (a !== undefined) answers[id] = { type: 'noul', noul: a };
        }
        return { answers };
      },
    };
  }

  const msg = (role: Message['role'], text: string, toolUses: ToolUse[] = []): Message => ({ role, text, toolUses });
  const use = (id: string, tool: string, input: Record<string, unknown>): ToolUse => ({ tool_use_id: id, tool, input, text: 'out' });
  const res = (id: string): Message => ({ role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: id, text: 'ok' }] });
  const pad = (): Message[] => [msg('user', 'pad one'), msg('user', 'pad two')];

  function decisionsFor(messages: Message[], actions: Record<string, CallAction>, preserve = 2): CallDecision[] {
    return collectToolCalls(messages, preserve).map((c) => {
      const action = actions[c.tool_use_id] ?? 'keep';
      const reason = c.pinned ? 'pinned' : action === 'keep' ? 'kept' : action === 'drop_result' ? 'result_dropped' : 'call_dropped';
      return { id: c.id, tool: c.tool, keepCall: 1, keepResult: 1, action, reason };
    });
  }

  const actionsOf = (messages: Message[], acts: Record<string, CallAction> = {}) =>
    callActions(messages, decisionsFor(messages, acts), 2);

  const body = `${'const value = compute();\n'.repeat(80)}`;

  test('callActions pairs calls with decisions by position and throws on a length mismatch', () => {
    const messages = [msg('user', 'go'), msg('assistant', '', [use('x1', 'Read', { file_path: 'a' })]), res('x1'), ...pad(), msg('assistant', 'end'), msg('user', 'end')];
    const decisions = decisionsFor(messages, { x1: 'drop_result' });
    const map = callActions(messages, decisions, 2);
    expect([...map.keys()]).toEqual(['x1']);
    expect(map.get('x1')?.action).toBe('drop_result');
    expect(() => callActions(messages, [], 2)).toThrow();
  });

  test('no request holds a protected span; scored parts are exactly the scored units', async () => {
    const protectedLine = 'Run `npm ci` in src/app.ts before v1.2.3 ships.';
    const messages = [msg('user', 'go'), msg('assistant', `${FLUFF} ${protectedLine}\n${FACT}`), ...pad(), msg('assistant', 'end'), msg('user', 'end')];
    const seen: Req[] = [];
    await trimProse(messages, new Map(), asker(() => 0.9, seen), OPTS);
    expect(seen).toHaveLength(1);
    const wire = JSON.stringify(seen[0]!.questions);
    for (const secret of ['npm ci', 'src/app.ts', 'v1.2.3']) expect(wire).not.toContain(secret);
    const parts = Object.entries(seen[0]!.state.window['m1'] ?? {}).filter(([k]) => /^c\d+$/.test(k));
    expect(parts.map(([, v]) => v)).toEqual([FLUFF, FACT]);
    expect(Object.keys(seen[0]!.questions)).toEqual(parts.map(([k]) => k));
  });

  test('user, pinned and untouched messages return as the same objects', async () => {
    const messages = [msg('user', FLUFF), msg('assistant', FLUFF), msg('user', FLUFF), msg('user', 'pad'), msg('assistant', FLUFF), msg('user', FLUFF)];
    const out = await trimProse(messages, new Map(), asker(() => 0), OPTS);
    expect(out.messages).toHaveLength(5);
    out.messages.forEach((m, i) => expect(m).toBe(messages[i === 0 ? 0 : i + 1]));
    expect(out.stats.dropped).toBe(1);
    expect(out.stats.proseCharsSaved).toBe(FLUFF.length);
  });

  test('a dropped clause leaves the kept clause bytes identical', async () => {
    const messages = [msg('user', 'go'), msg('assistant', `${FLUFF} ${FACT}`), ...pad(), msg('assistant', 'end'), msg('user', 'end')];
    const out = await trimProse(messages, new Map(), asker((id, r) => (r.state.window['m1']?.[id] === FLUFF ? 0.01 : 0.9)), OPTS);
    expect(out.messages[1]?.text).toBe(FACT);
    expect(out.decisions.map((d) => [d.message, d.kept])).toEqual([[1, false], [1, true]]);
  });

  test('drop_call and unmapped calls are untouched; keep and drop_result bodies are elided', async () => {
    const mk = (id: string) => use(id, 'Write', { file_path: 'src/app.ts', content: body });
    const messages = [
      msg('user', 'go'),
      msg('assistant', '', [mk('k1'), mk('d1'), mk('c1'), mk('u1')]),
      res('k1'), res('d1'), res('c1'),
      ...pad(), msg('assistant', 'end'), msg('user', 'end'),
    ];
    const actions = actionsOf(messages, { k1: 'keep', d1: 'drop_result', c1: 'drop_call' });
    const out = await trimProse(messages, actions, asker(() => 0.9), OPTS);
    const uses = out.messages[1]!.toolUses;
    const input = (i: number) => uses[i]!.input['content'] as string;
    for (const i of [0, 1]) {
      expect(input(i).startsWith(body.slice(0, 300))).toBe(true);
      expect(input(i)).toContain('[jev-compaction elided');
      expect(input(i)).toContain('Write.content for src/app.ts');
      expect(input(i).length).toBeLessThan(body.length);
    }
    expect(uses[2]).toBe(messages[1]!.toolUses[2]);
    expect(uses[3]).toBe(messages[1]!.toolUses[3]);
    expect(out.stats.inputsElided).toBe(2);
    expect(out.stats.inputCharsSaved).toBe(2 * (body.length - input(0).length));
  });

  test('a call whose result is in the pinned tail is untouched', async () => {
    const messages = [msg('user', 'go'), msg('assistant', 'x'), ...pad(), msg('assistant', '', [use('p1', 'Write', { file_path: 'a.ts', content: body })]), res('p1')];
    const map = new Map<string, CallDecision>([['p1', { id: 't1', tool: 'Write', keepCall: 1, keepResult: 1, action: 'keep', reason: 'kept' }]]);
    const out = await trimProse(messages, map, asker(() => 0.9), OPTS);
    expect(out.messages[4]).toBe(messages[4]);
  });

  test('a stale-only map elides only drop_result calls', async () => {
    const mk = (id: string) => use(id, 'Write', { file_path: 'src/app.ts', content: body });
    const messages = [msg('user', 'go'), msg('assistant', '', [mk('k1'), mk('d1')]), res('k1'), res('d1'), ...pad(), msg('assistant', 'end'), msg('user', 'end')];
    const full = actionsOf(messages, { k1: 'keep', d1: 'drop_result' });
    const stale = new Map([...full].filter(([, d]) => d.action === 'drop_result'));
    const out = await trimProse(messages, stale, asker(() => 0.9), OPTS);
    expect(out.messages[1]!.toolUses[0]).toBe(messages[1]!.toolUses[0]);
    expect(out.messages[1]!.toolUses[1]!.input['content']).toContain('[jev-compaction elided');
  });

  test('a prose input of 200 chars is scored, 199 is not', async () => {
    const sentence = (n: number) => `${'Please look at the way the cache layer behaves overall. '.repeat(10)}`.slice(0, n - 1) + '.';
    for (const [n, asked] of [[200, 1], [199, 0]] as const) {
      const messages = [msg('user', 'go'), msg('assistant', '', [use('a1', 'Agent', { prompt: sentence(n), subagent_type: 'x' })]), res('a1'), ...pad(), msg('assistant', 'end'), msg('user', 'end')];
      const seen: Req[] = [];
      await trimProse(messages, actionsOf(messages), asker(() => 0.9, seen), OPTS);
      expect(seen.length).toBe(asked);
    }
  });

  test('an edited call is a fresh ToolUse and input; name, id, keys and other fields are equal', async () => {
    const original = use('w1', 'Write', { file_path: 'src/app.ts', content: body, mode: 'x' });
    const messages = [msg('user', 'go'), msg('assistant', '', [original]), res('w1'), ...pad(), msg('assistant', 'end'), msg('user', 'end')];
    const out = await trimProse(messages, actionsOf(messages), asker(() => 0.9), OPTS);
    const edited = out.messages[1]!.toolUses[0]!;
    expect(edited).not.toBe(original);
    expect(edited.input).not.toBe(original.input);
    expect(edited.tool).toBe('Write');
    expect(edited.tool_use_id).toBe('w1');
    expect(edited.text).toBe('out');
    expect(Object.keys(edited.input)).toEqual(['file_path', 'content', 'mode']);
    expect(edited.input['file_path']).toBe('src/app.ts');
    expect(edited.input['mode']).toBe('x');
    expect(original.input['content']).toBe(body);
  });

  test('a heredoc body is cut and the command line and closing marker stay', async () => {
    const command = `cat > src/x.ts <<'EOF'\n${body}EOF\necho done`;
    const messages = [msg('user', 'go'), msg('assistant', '', [use('b1', 'Bash', { command })]), res('b1'), ...pad(), msg('assistant', 'end'), msg('user', 'end')];
    const out = await trimProse(messages, actionsOf(messages), asker(() => 0.9), OPTS);
    const cut = out.messages[1]!.toolUses[0]!.input['command'] as string;
    expect(cut.startsWith("cat > src/x.ts <<'EOF'\n")).toBe(true);
    expect(cut.endsWith('\nEOF\necho done')).toBe(true);
    expect(cut).toContain('[jev-compaction elided');
    expect(cut.length).toBeLessThan(command.length);
  });

  describe('archiving cut bodies', () => {
    const writeMessages = (): Message[] => [
      msg('user', 'go'),
      msg('assistant', '', [use('w1', 'Write', { file_path: 'src/app.ts', content: body })]),
      res('w1'),
      ...pad(),
      msg('assistant', 'end'),
      msg('user', 'end'),
    ];
    const recorder = (path: string | null = 'arch/out.txt') => {
      const calls: Array<[string, string]> = [];
      const archive = (id: string, text: string): string | undefined => {
        calls.push([id, text]);
        return path === null ? undefined : `${path}#${calls.length}`;
      };
      return { calls, archive };
    };

    test('a cut Write body is archived whole and the marker cites the path', async () => {
      const messages = writeMessages();
      const { calls, archive } = recorder();
      const out = await trimProse(messages, actionsOf(messages), asker(() => 0.9), { ...OPTS, archive });
      expect(calls).toHaveLength(1);
      expect(calls[0]![0]).toContain('w1');
      expect(calls[0]![1]).toBe(body);
      const cut = out.messages[1]!.toolUses[0]!.input['content'] as string;
      expect(cut).toContain('; full output: arch/out.txt#1]');
      expect(cut).not.toContain('Read the file if needed');
    });

    test('a cut heredoc body is archived whole under its own id', async () => {
      const command = `cat > src/x.ts <<'EOF'\n${body}EOF\necho done`;
      const messages = [msg('user', 'go'), msg('assistant', '', [use('b1', 'Bash', { command })]), res('b1'), ...pad(), msg('assistant', 'end'), msg('user', 'end')];
      const { calls, archive } = recorder();
      const out = await trimProse(messages, actionsOf(messages), asker(() => 0.9), { ...OPTS, archive });
      expect(calls.map(([, text]) => text)).toEqual([body]);
      const cut = out.messages[1]!.toolUses[0]!.input['command'] as string;
      expect(cut).toContain('full output: arch/out.txt#1]');
      expect(cut.endsWith('\nEOF\necho done')).toBe(true);
    });

    test('each cut field gets a distinct archive id', async () => {
      const messages = [
        msg('user', 'go'),
        msg('assistant', '', [use('e1', 'Edit', { file_path: 'a.ts', old_string: body, new_string: body + 'x' })]),
        res('e1'),
        ...pad(),
        msg('assistant', 'end'),
        msg('user', 'end'),
      ];
      const { calls, archive } = recorder();
      await trimProse(messages, actionsOf(messages), asker(() => 0.9), { ...OPTS, archive });
      expect(calls).toHaveLength(2);
      expect(new Set(calls.map(([id]) => id)).size).toBe(2);
    });

    test.each([
      ['no archive', undefined],
      ['an archive that returns undefined', recorder(null).archive],
    ])('with %s the output equals the unarchived output', async (_name, archive) => {
      const messages = writeMessages();
      const plain = await trimProse(messages, actionsOf(messages), asker(() => 0.9), OPTS);
      const out = await trimProse(messages, actionsOf(messages), asker(() => 0.9), { ...OPTS, ...(archive ? { archive } : {}) });
      expect(out.messages).toEqual(plain.messages);
      expect(out.messages[1]!.toolUses[0]!.input['content']).toContain('Read the file if needed');
    });
  });

  test('a missing answer keeps the unit', async () => {
    const messages = [msg('user', 'go'), msg('assistant', FLUFF), ...pad(), msg('assistant', 'end'), msg('user', 'end')];
    const out = await trimProse(messages, new Map(), asker(() => undefined), OPTS);
    expect(out.messages[1]).toBe(messages[1]);
    expect(out.decisions).toEqual([{ id: 'c1', message: 1, kept: true }]);
  });

  test.each([[0.14, false], [0.15, true]])('noul %s kept=%s', async (noul, kept) => {
    const messages = [msg('user', 'go'), msg('assistant', FLUFF), ...pad(), msg('assistant', 'end'), msg('user', 'end')];
    const out = await trimProse(messages, new Map(), asker(() => noul), OPTS);
    expect(out.decisions[0]).toEqual({ id: 'c1', message: 1, noul, kept });
  });

  test('one failed batch keeps its units while the others apply', async () => {
    const messages = [msg('user', 'go'), msg('assistant', FLUFF), msg('assistant', FACT), ...pad(), msg('assistant', 'end'), msg('user', 'end')];
    const seen: Req[] = [];
    const out = await trimProse(
      messages,
      new Map(),
      asker((_id, r) => (JSON.stringify(r.state).includes('m1') ? 'throw' : 0), seen),
      { ...OPTS, windowTokens: 1 },
    );
    expect(seen).toHaveLength(2);
    expect(out.messages[1]).toBe(messages[1]);
    expect(out.messages.some((m) => m.text === FACT)).toBe(false);
    expect(out.stats).toMatchObject({ requests: 2, failedRequests: 1, dropped: 1 });
    expect(out.stats.failed).toBeUndefined();
  });

  test('every batch failing returns the input objects, cuts no body and sets stats.failed', async () => {
    const messages = [msg('user', 'go'), msg('assistant', FLUFF, [use('w1', 'Write', { file_path: 'a.ts', content: body })]), res('w1'), ...pad(), msg('assistant', 'end'), msg('user', 'end')];
    const out = await trimProse(messages, actionsOf(messages), asker(() => 'throw'), OPTS);
    expect(out.messages).toHaveLength(messages.length);
    out.messages.forEach((m, i) => expect(m).toBe(messages[i]));
    expect(out.decisions).toEqual([]);
    expect(out.stats.failed).toBeTruthy();
    expect(out.stats).toMatchObject({ dropped: 0, inputsElided: 0, proseCharsSaved: 0, inputCharsSaved: 0 });
  });

  test('windows split at the token limit and run at most 4 requests at a time', async () => {
    const messages = [msg('user', 'go'), ...Array.from({ length: 9 }, () => msg('assistant', FLUFF)), ...pad(), msg('assistant', 'end'), msg('user', 'end')];
    let live = 0;
    let peak = 0;
    const slow: JevAsker = {
      async ask(_s, questions) {
        live++;
        peak = Math.max(peak, live);
        await new Promise((r) => setTimeout(r, 5));
        live--;
        return { answers: Object.fromEntries(Object.keys(questions).map((k) => [k, { type: 'noul' as const, noul: 0.9 }])) };
      },
    };
    const out = await trimProse(messages, new Map(), slow, { ...OPTS, windowTokens: 1 });
    expect(out.stats.requests).toBe(9);
    expect(peak).toBe(4);
  });

  test('applyProse reapplies one set of scores at other cut-offs with no further requests', async () => {
    const messages = [msg('user', 'go'), msg('assistant', FLUFF), ...pad(), msg('assistant', 'end'), msg('user', 'end')];
    const seen: Req[] = [];
    const scores = await scoreProse(messages, new Map(), asker(() => 0.1, seen), OPTS);
    expect(seen).toHaveLength(1);
    expect(applyProse(scores, 0.05).messages[1]).toBe(messages[1]);
    expect(applyProse(scores, 0.15).stats).toMatchObject({ scored: 1, dropped: 1 });
    expect(applyProse(scores, 0.05).stats).toMatchObject({ scored: 1, dropped: 0 });
    expect(seen).toHaveLength(1);
  });
  describe('ladderRescue', () => {
    const base = (messages: Message[]): CompactResult => {
      const chars = messages.reduce((n, m) => n + messageChars(m), 0);
      return { messages, decisions: [], stats: { charsBefore: chars, charsAfter: chars } } as unknown as CompactResult;
    };
    const fluffy = (): Message[] => [msg('user', 'go'), msg('assistant', Array(10).fill(FLUFF).join(' ')), ...pad(), msg('assistant', 'end'), msg('user', 'end')];

    test('takes the first cut-off whose result reaches the minimum, measured against charsBefore', async () => {
      const messages = fluffy();
      const result = base(messages);
      const scores = await scoreProse(messages, new Map(), asker(() => 0.1), OPTS);
      const out = ladderRescue(scores, result, 0.25);
      expect(out?.cutoff).toBe(0.15);
      expect(out?.result.stats.charsAfter).toBeLessThan(result.stats.charsAfter);
      expect(out?.result.stats.charsBefore).toBe(result.stats.charsBefore);
      expect(out?.stats.dropped).toBe(10);
    });

    test('is undefined when no cut-off reaches the minimum', async () => {
      const messages = fluffy();
      const scores = await scoreProse(messages, new Map(), asker(() => 0.9), OPTS);
      expect(ladderRescue(scores, base(messages), 0.25)).toBeUndefined();
    });
  });
});
