import { describe, expect, it } from 'vitest';
import { compactSession } from '../hooks/fast-jev.ts';
import { resolveOptions } from '../src/compact.js';
import { applyInjected, findBlocks, goalText, injectedSpans, scoreInjected } from '../src/inject.js';
import { goalFromMessages } from '../src/state.js';
import type { JevAsker, Message } from '../src/types.js';

const user = (text: string): Message => ({ role: 'user', text, toolUses: [] });
const assistant = (text: string): Message => ({ role: 'assistant', text, toolUses: [] });
const options = { ...resolveOptions({ preserveRecentMessages: 2 }), goal: 'Ship the button.' };

const SKILL_BODY = 'Step: do the thing carefully.\n'.repeat(120); // 3,600 chars
const skill = (args?: string) =>
  `Base directory for this skill: /x/skills/demo\n\n${SKILL_BODY}${args === undefined ? '' : `\nARGUMENTS: ${args}`}`;
const ELEMENT_BODY = '<div class="a b c" style="color: red">hello</div>\n'.repeat(80); // 4,000 chars
const element = (typed: string) => `<launch-selected-element>\n${ELEMENT_BODY}</launch-selected-element>\n\n${typed}`;

/** Three short prompts at the end form the goal; anything before them is eligible. */
function session(...older: Message[]): Message[] {
  return [
    user('start'),
    ...older,
    assistant('done'),
    user('first recent prompt'),
    assistant('ok'),
    user('second recent prompt'),
    assistant('ok'),
    user('third recent prompt'),
    assistant('ok'),
  ];
}
function asker(answer: (key: string) => number | undefined, calls: string[][] = []): JevAsker {
  return {
    async ask(_state, questions) {
      calls.push(Object.keys(questions));
      const answers: Record<string, unknown> = {};
      for (const key of Object.keys(questions)) {
        const n = answer(key);
        if (n !== undefined) answers[key] = { noul: n };
      }
      return { answers };
    },
  };
}
const cite = (id: string) => `.claude/jev-compaction/${id}.txt`;
/** A pin count that reaches back to the first older row of `session(row)`. */
const messagesAfter = (_row: Message) => session(_row).length - 1;

describe('injected spans', () => {
  it('finds a skill body and stops before the typed ARGUMENTS', () => {
    const text = skill('make the bar thinner');
    const [span] = injectedSpans(text);
    expect(span?.kind).toBe('skill');
    expect(text.slice(span!.span[1])).toBe('\nARGUMENTS: make the bar thinner');
    expect(text.slice(0, span!.span[0])).toBe('Base directory for this skill: /x/skills/demo\n');
  });

  it('finds an element selection and leaves the typed request outside it', () => {
    const text = element('make this thinner');
    const [span] = injectedSpans(text);
    expect(span?.kind).toBe('element');
    expect(text.slice(span!.span[1])).toBe('</launch-selected-element>\n\nmake this thinner');
  });

  it('marks a long row with no known block as long, and a short one as nothing', () => {
    expect(injectedSpans('x'.repeat(3000))).toEqual([{ kind: 'long', span: [0, 3000] }]);
    expect(injectedSpans('x'.repeat(2999))).toEqual([]);
  });

  it('reads the goal from typed text only', () => {
    expect(goalText(skill('make the bar thinner'))).toBe('make the bar thinner');
    expect(goalText(skill())).toBe('');
    expect(goalText(element('make this thinner'))).toBe('<launch-selected-element></launch-selected-element>\n\nmake this thinner');
    const goal = goalFromMessages([user(element('make this thinner')), assistant('ok'), user(skill())]);
    expect(goal).not.toContain('class="a b c"');
    expect(goal).not.toContain('Step: do the thing');
    expect(goal).toContain('make this thinner');
  });
});

describe('injected-content fold', () => {
  it('never offers the last three prompts, pinned rows or tool-result rows', () => {
    const messages = session(user(skill('old')), user('x'.repeat(5000)));
    const blocks = findBlocks(messages, 2, 150);
    expect(blocks.map((b) => [b.index, b.kind])).toEqual([
      [1, 'skill'],
      [2, 'long'],
    ]);
    const recent = [user('start'), user(skill('a')), user(skill('b')), user(skill('c'))];
    expect(findBlocks(recent, 0, 150)).toEqual([]);
    expect(findBlocks([user(skill('first row')), ...recent.slice(1)], 0, 150)).toEqual([]);
    expect(findBlocks(session(user(skill('a'))), messagesAfter(user(skill('a'))), 150)).toEqual([]);
  });

  it('folds a stale skill body, keeps the typed ARGUMENTS and archives the whole row', async () => {
    const original = skill('make the bar thinner');
    const messages = session(user(original));
    const scores = await scoreInjected(messages, asker(() => 0.1), options);
    const archived: [string, string][] = [];
    const applied = applyInjected(scores, 0.35, (id, text) => (archived.push([id, text]), cite(id)));
    const out = applied.messages[1]!.text;
    expect(out.startsWith('Base directory for this skill: /x/skills/demo\n\nStep: do the thing')).toBe(true);
    expect(out).toContain('[jev-compaction folded ');
    expect(out).toContain(`full text: ${cite(applied.archived[0]!.id)}`);
    expect(out.endsWith('\nARGUMENTS: make the bar thinner')).toBe(true);
    expect(out.length).toBeLessThan(original.length / 4);
    expect(archived).toEqual([[applied.archived[0]!.id, original]]);
    expect(applied.messages.slice(2)).toEqual(messages.slice(2));
    expect(applied.messages[0]).toBe(messages[0]);
  });

  it('keeps every typed character around a folded element selection', async () => {
    const original = element('Also the avatar still feels too small.');
    const messages = session(user(original));
    const applied = applyInjected(await scoreInjected(messages, asker(() => 0.1), options), 0.35, cite);
    const out = applied.messages[1]!.text;
    expect(out.startsWith('<launch-selected-element>\n')).toBe(true);
    expect(out.endsWith('</launch-selected-element>\n\nAlso the avatar still feels too small.')).toBe(true);
    expect(out).toContain('folded ');
    expect(out).toContain('element selection');
  });

  it('keeps a block Jev scores as still needed, and one with no answer', async () => {
    const messages = session(user(skill('a')), user(element('b')));
    const scores = await scoreInjected(messages, asker((k) => (k === 'i1' ? 0.9 : undefined)), options);
    expect(applyInjected(scores, 0.35, cite).messages).toEqual(messages);
  });

  it('at the all rung folds skills and elements without an answer but never an unscored long row', async () => {
    const long = 'my own long notes\n'.repeat(300);
    const messages = session(user(skill('a')), user(long));
    const scores = await scoreInjected(messages, asker(() => undefined), options);
    const applied = applyInjected(scores, 'all', cite);
    expect(applied.messages[1]!.text).toContain('folded ');
    expect(applied.messages[2]!.text).toBe(long);
  });

  it('changes nothing without an archive path, or when Jev fails', async () => {
    const messages = session(user(skill('a')));
    const scores = await scoreInjected(messages, asker(() => 0), options);
    expect(applyInjected(scores, 0.35, () => undefined).messages).toEqual(messages);
    expect(applyInjected(scores, 0.35).messages).toEqual(messages);
    const failing: JevAsker = { ask: async () => Promise.reject(new Error('down')) };
    const failed = await scoreInjected(messages, failing, options);
    expect(failed.stats.failed).toBe('down');
    expect(applyInjected(failed, 'all', cite).messages).toEqual(messages);
  });

  it('does not fold a row it already folded', async () => {
    const messages = session(user(skill('a')));
    const once = applyInjected(await scoreInjected(messages, asker(() => 0), options), 0.35, cite).messages;
    expect(findBlocks(once, 2, 150)).toEqual([]);
  });
});

describe('injected-content fold through compactSession', () => {
  const fetchAll = (noul: number, bodies: string[] = []) => async (_url: string, init?: { body?: string }) => {
    bodies.push(init?.body ?? '');
    const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
    const answers = Object.fromEntries(Object.keys(questions).map((k) => [k, { type: 'noul', noul }]));
    return { status: 200, ok: true, text: JSON.stringify({ answers }) };
  };
  const archive = () => {
    const files = new Map<string, string>();
    const pathOf = (id: string) => `arch/${id}.txt`;
    return { files, pathOf, cite: (id: string, text: string) => (files.set(id, text), pathOf(id)), flush: async () => {} };
  };
  const messages = (): Message[] => session(user(skill('use it')), user(element('make it blue')));

  it('reaches the minimum on injected content alone and archives each original', async () => {
    const a = archive();
    const out = await compactSession(messages(), { apiKey: 'k', minReductionRatio: 0.5, archiveResults: true }, fetchAll(0.05), a);
    expect(out.injected?.rung).toBeDefined();
    expect(out.injected?.stats.folded).toBe(2);
    expect([...a.files.values()]).toEqual(expect.arrayContaining([messages()[1]!.text, messages()[2]!.text]));
    expect(out.messages[1]!.text.endsWith('\nARGUMENTS: use it')).toBe(true);
    expect(out.messages[2]!.text.endsWith('\n\nmake it blue')).toBe(true);
    expect(out.messages[0]).toEqual(messages()[0]);
    expect(out.messages.slice(3)).toEqual(messages().slice(3));
  });

  it('does nothing when injectTrim is off', async () => {
    const out = await compactSession(
      messages(),
      { apiKey: 'k', minReductionRatio: 0.5, archiveResults: true, injectTrim: false, notificationTrim: false, proseTrim: false },
      fetchAll(0.05),
      archive(),
    );
    expect(out.injected).toBeUndefined();
    expect(out.messages).toEqual(messages());
  });

  it('does nothing without an archive writer', async () => {
    const out = await compactSession(messages(), { apiKey: 'k', minReductionRatio: 0.5, archiveResults: false }, fetchAll(0.05));
    expect(out.messages).toEqual(messages());
  });
});

describe('overlapping injected spans', () => {
  it('folds only the outer block of a paste that holds element tags and keeps the typed text after it', () => {
    const inner = `<launch-selected-element>${'e'.repeat(2000)}</launch-selected-element>`;
    const text = `TYPED-BEFORE <pasted_content id="p1">${'a'.repeat(2000)}${inner}${'b'.repeat(2000)}</pasted_content> TYPED-AFTER keep this exact sentence.`;
    const spans = injectedSpans(text);
    expect(spans).toHaveLength(1);
    expect(spans[0]!.kind).toBe('paste');
    const messages = [user('start'), user(text), user('a'), user('b'), user('c'), user('d')];
    const blocks = findBlocks(messages, 2, 200);
    const noul = new Map(blocks.map((b) => [b.key, 0.01]));
    const out = applyInjected({ messages, blocks, noul, headChars: 200, salvageChars: 0, stats: { eligible: blocks.length, scored: blocks.length, requests: 1, failedRequests: 0, stateTokens: 0 } } as any, 0.05, (id) => `arch/${id}.txt`);
    const row = out.messages[1]!.text;
    expect(row.startsWith('TYPED-BEFORE <pasted_content id="p1">')).toBe(true);
    expect(row.endsWith('</pasted_content> TYPED-AFTER keep this exact sentence.')).toBe(true);
    expect(row.length).toBeLessThan(text.length);
  });
});
