import { describe, expect, it } from 'vitest';
import { foldStepNotes } from '../src/stepnotes.js';
import type { Message } from '../src/types.js';

const user = (text: string): Message => ({ role: 'user', text, toolUses: [] });
const reply = (text: string): Message => ({ role: 'assistant', text, toolUses: [] });
const call = (id: string): Message => ({ role: 'assistant', text: '', toolUses: [{ tool_use_id: id, tool: 'Bash', input: { command: 'ls' } }] });
const result = (id: string): Message => ({ role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: id, text: 'out', isError: false }] });
const step = (n: number, rows = 2, calls = 1, name = 'jev-compaction') =>
  `[${name} archived ${rows} rows of earlier work, ${calls} tool call${calls > 1 ? 's' : ''} (Bash), 3000 chars; full text: /w/.claude/${name}/step-${n}.txt (Read it if needed)]\nNote: finished part ${n}.`;
const note = (n: number, rows = 2, calls = 1, name = 'jev-compaction') => reply(step(n, rows, calls, name));

function cite() {
  const files = new Map<string, string>();
  return { files, cite: (id: string, text: string) => (files.set(id, text), `/w/.claude/jev-compaction/${id}.txt`) };
}

const FOLDED = /^\[jev-compaction archived (\d+) steps of this prompt's work \((\d+) rows, (\d+) tool calls\); each step's note and file: (\S+) \(Read it if needed\)\]$/;

describe('foldStepNotes', () => {
  it("gives one prompt's work one note in place of its step notes, keeping every other row in order", () => {
    const messages = [
      user('build the canvas'),
      note(1, 4, 3),
      reply('Set up the project.'),
      call('a'),
      result('a'),
      note(2),
      reply('Wired the routes.'),
      note(3, 6, 2),
      user('now the tests'),
      reply('working'),
    ];
    const c = cite();
    const out = foldStepNotes(messages, 2, c.cite);
    expect(out.folded).toBe(3);
    expect(out.messages.map((m) => m.text.slice(0, 22))).toEqual([
      'build the canvas',
      expect.stringMatching(/^\[jev-compaction archiv/),
      'Set up the project.',
      '',
      '',
      'Wired the routes.',
      'now the tests',
      'working',
    ]);
    const m = FOLDED.exec(out.messages[1]!.text)!;
    expect([m[1], m[2], m[3]]).toEqual(['3', '12', '6']);
    const file = c.files.get(m[4]!.replace(/^.*\//, '').replace(/\.txt$/, ''))!;
    expect(file).toContain('prompt: build the canvas');
    for (const n of [1, 2, 3]) expect(file).toContain(step(n, n === 1 ? 4 : n === 3 ? 6 : 2, n === 1 ? 3 : n === 3 ? 2 : 1));
    expect(out.messages[3]).toBe(messages[3]);
    expect(out.messages[4]).toBe(messages[4]);
  });

  it('leaves a lone step note alone and never joins notes across prompts', () => {
    const messages = [user('one'), note(1), reply('done one'), user('two'), note(2), reply('done two'), user('three'), reply('now')];
    const out = foldStepNotes(messages, 1, cite().cite);
    expect(out.folded).toBe(0);
    expect(out.messages).toBe(messages);
  });

  it('never touches a note in the pinned recent rows', () => {
    const messages = [user('task'), note(1), reply('a'), note(2), reply('b'), note(3), reply('c')];
    const out = foldStepNotes(messages, 3, cite().cite);
    expect(out.folded).toBe(2);
    expect(out.messages.slice(-3)).toEqual(messages.slice(-3));
    expect(out.messages.filter((x) => x.text.startsWith('[jev-compaction archived 2 rows')).length).toBe(1);
  });

  it("removes a step note merged into a reply but keeps the reply's own text", () => {
    const messages = [user('task'), reply(`Checked the build.\n${step(1)}`), reply('next'), note(2), user('later'), reply('x'), reply('y')];
    const out = foldStepNotes(messages, 2, cite().cite);
    expect(out.folded).toBe(2);
    expect(out.messages.map((m) => m.text.slice(0, 22))).toEqual([
      'task',
      expect.stringMatching(/^\[jev-compaction archiv/),
      'Checked the build.',
      'next',
      'later',
      'x',
      'y',
    ]);
  });

  it('folds old-name step notes and an earlier combined note into the new one', () => {
    const c = cite();
    const first = foldStepNotes([user('task'), note(1), reply('a'), note(2), user('next'), reply('z')], 1, c.cite).messages;
    const again = [...first.slice(0, 3), note(3, 2, 1, 'fast-jev-compaction'), reply('b'), ...first.slice(3)];
    const out = foldStepNotes(again, 1, c.cite);
    expect(out.messages.filter((m) => FOLDED.test(m.text)).length).toBe(1);
    const m = FOLDED.exec(out.messages.find((x) => FOLDED.test(x.text))!.text)!;
    expect([m[1], m[2], m[3]]).toEqual(['3', '6', '3']);
    expect(out.messages.some((x) => x.text.includes('fast-jev-compaction archived'))).toBe(false);
  });

  it('changes nothing when the archive gives no path', () => {
    const messages = [user('task'), note(1), reply('a'), note(2), user('next'), reply('z')];
    expect(foldStepNotes(messages, 1, () => undefined).messages).toBe(messages);
  });
});
