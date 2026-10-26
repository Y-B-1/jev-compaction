import { describe, expect, it } from 'vitest';
import { foldNoteChains } from '../src/chains.js';
import type { Message } from '../src/types.js';

const user = (text: string): Message => ({ role: 'user', text, toolUses: [] });
const reply = (text: string): Message => ({ role: 'assistant', text, toolUses: [] });
const note = (n: number, name = 'jev-compaction') =>
  reply(`[${name} archived 2 rows of earlier work, 1 tool call (Bash), 3000 chars; full text: /w/.claude/${name}/step-${n}.txt (Read it if needed)]\nNote: finished part ${n}.`);

function cite() {
  const files = new Map<string, string>();
  return { files, cite: (id: string, text: string) => (files.set(id, text), `/w/.claude/jev-compaction/${id}.txt`) };
}

describe('foldNoteChains', () => {
  it('folds notes separated only by prompts into one note after the last prompt', () => {
    const messages = [user('start'), reply('ok'), user('first task'), note(1), user('second task'), note(2), user('third task'), note(3), user('now'), reply('working')];
    const c = cite();
    const out = foldNoteChains(messages, 2, c.cite);
    expect(out.folded).toBe(3);
    expect(out.messages.map((m) => m.text.split('\n')[0]!.slice(0, 40))).toEqual([
      'start',
      'ok',
      'first task',
      'second task',
      'third task',
      expect.stringMatching(/^\[jev-compaction folded 3 notes/),
      'now',
      'working',
    ]);
    const folded = out.messages[5]!;
    expect(folded.role).toBe('assistant');
    expect(folded.text).toMatch(/^\[jev-compaction folded 3 notes on the prompts above into one file, each naming its file: \/w\/\.claude\/jev-compaction\/notes-[0-9a-f]{12}\.txt \(Read it if needed\)\]$/);
    const [file] = [...c.files.values()];
    for (const n of [1, 2, 3]) expect(file).toContain(`step-${n}.txt`);
    expect(file).toContain('after: second task');
    expect(out.messages[0]).toBe(messages[0]);
    expect(out.messages[4]).toBe(messages[6]);
  });

  it('folds notes left by the old name too', () => {
    const messages = [user('start'), reply('ok'), user('a'), note(1, 'fast-jev-compaction'), user('b'), note(2), user('now'), reply('working')];
    expect(foldNoteChains(messages, 2, cite().cite).folded).toBe(2);
  });

  it('leaves a chain broken by anything but a prompt, a lone note, and notes among the pinned rows', () => {
    const c = cite();
    const broken = [user('start'), reply('ok'), user('a'), note(1), reply('a kept reply'), user('b'), note(2), user('now'), reply('working')];
    expect(foldNoteChains(broken, 2, c.cite)).toEqual({ messages: broken, folded: 0 });
    const lone = [user('start'), reply('ok'), user('a'), note(1), user('now'), reply('working')];
    expect(foldNoteChains(lone, 2, c.cite).folded).toBe(0);
    const pinned = [user('start'), reply('ok'), user('a'), note(1), user('b'), note(2)];
    expect(foldNoteChains(pinned, 2, c.cite).folded).toBe(0);
    const withText = [user('start'), reply('ok'), user('a'), reply(`${note(1).text}\nAnd my own words.`), user('b'), note(2), user('now'), reply('working')];
    expect(foldNoteChains(withText, 2, c.cite).folded).toBe(0);
    expect(c.files.size).toBe(0);
  });

  it('changes nothing without an archive path', () => {
    const messages = [user('start'), reply('ok'), user('a'), note(1), user('b'), note(2), user('now'), reply('working')];
    expect(foldNoteChains(messages, 2, () => undefined)).toEqual({ messages, folded: 0 });
  });
});
