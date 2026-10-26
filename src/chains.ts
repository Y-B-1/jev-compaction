import { sha256Hex } from './sha256.js';
import { MARKER, hasMarker, isPinned, isPromptRow, ownText } from './state.js';
import type { Message } from './types.js';

/**
 * Note chains: after a few compactions, each old prompt is often followed by nothing but one
 * archive note (`prompt, note, prompt, note, …`), which fills the transcript with notes. A chain
 * of two or more note-only rows that only prompt rows separate becomes one note after the last
 * prompt of the chain. That note names one file holding the original notes, each with the prompt
 * it followed, and each still naming its own archive file. Every prompt stays word for word.
 */

const PROMPT_HEAD_CHARS = 200;

/** A row that is only this plugin's notes: an assistant row with no tool call and no text of its own. */
const isNoteRow = (m: Message): boolean => m.role === 'assistant' && m.toolUses.length === 0 && hasMarker(m.text) && ownText(m.text) === '';

/**
 * `messages` with each note chain outside the pinned recent rows folded (see above). `cite` takes
 * the file's id and text and returns its path; without a path the chain stays. Untouched rows come
 * back as the same objects.
 */
export function foldNoteChains<M extends Message>(
  messages: readonly M[],
  preserveRecentMessages: number,
  cite: (id: string, text: string) => string | undefined,
): { messages: M[]; folded: number } {
  const total = messages.length;
  const free = (i: number) => !isPinned(i, total, preserveRecentMessages);
  const out: M[] = [];
  let folded = 0;
  let i = 0;
  while (i < total) {
    // A chain: note rows at `notes`, with only prompt rows between consecutive ones.
    const notes: number[] = [];
    if (isNoteRow(messages[i] as M) && free(i)) {
      notes.push(i);
      let j = i + 1;
      for (;;) {
        let k = j;
        while (k < total && free(k) && isPromptRow(messages[k] as M)) k += 1;
        if (k === j || k >= total || !free(k) || !isNoteRow(messages[k] as M)) break;
        notes.push(k);
        j = k + 1;
      }
    }
    if (notes.length < 2) {
      out.push(messages[i] as M);
      i += 1;
      continue;
    }
    const parts = notes.map((n) => {
      let p = n - 1;
      while (p >= 0 && !isPromptRow(messages[p] as M)) p -= 1;
      const prompt = p >= 0 ? (messages[p] as M).text.trim().slice(0, PROMPT_HEAD_CHARS) : '';
      return `${prompt ? `after: ${prompt}\n` : ''}${(messages[n] as M).text}`;
    });
    const text = parts.join('\n\n');
    const path = cite(`notes-${sha256Hex(text).slice(0, 12)}`, text);
    const last = notes.at(-1) as number;
    if (!path) {
      out.push(...messages.slice(i, last + 1));
      i = last + 1;
      continue;
    }
    for (let r = i; r < last; r++) if (!notes.includes(r)) out.push(messages[r] as M);
    out.push({
      role: 'assistant',
      text: `${MARKER}folded ${notes.length} notes on the prompts above into one file, each naming its file: ${path} (Read it if needed)]`,
      toolUses: [],
    } as unknown as M);
    folded += notes.length;
    i = last + 1;
  }
  return folded > 0 ? { messages: out, folded } : { messages: messages as M[], folded: 0 };
}
