import { sha256Hex } from './sha256.js';
import { MARKER, isPinned, isPromptRow } from './state.js';
import type { Message } from './types.js';

/**
 * Step notes: step trim leaves one note per archived step, between the rows Jev kept, so a long
 * piece of work under one prompt can end up with dozens of notes. Each prompt's step notes
 * outside the pinned recent rows become one note, at the place of the first. That note names one
 * file listing the original notes in order, each still naming its own archive file. An earlier
 * combined note counts as one of them, so a prompt keeps one note across compactions. Every other
 * row, and every other line of a row, stays as it was.
 */

const PROMPT_HEAD_CHARS = 200;

/** A step note's first line, under the current or the old name. */
const STEP_LINE = /^\s*\[(?:fast-)?jev-compaction archived (\d+) rows? of earlier work(?:, (\d+) tool calls?)?/;
/** A combined note's line (this module's own). */
const FOLDED_LINE = /^\s*\[(?:fast-)?jev-compaction archived (\d+) steps of this prompt's work \((\d+) rows, (\d+) tool calls\)/;
/** The line a step note may add after its marker. */
const NOTE_LINE = /^(?:Note|Last reply): /;

type Member = { lines: string[]; steps: number; rows: number; calls: number };

/** The step notes in `text`, and the text without them. */
function split(text: string): { members: Member[]; rest: string } {
  const members: Member[] = [];
  const kept: string[] = [];
  let current: Member | undefined;
  for (const line of text.split('\n')) {
    const folded = FOLDED_LINE.exec(line);
    const stepLine = folded ? null : STEP_LINE.exec(line);
    if (folded || stepLine) {
      current = folded
        ? { lines: [line.trim()], steps: Number(folded[1]), rows: Number(folded[2]), calls: Number(folded[3]) }
        : { lines: [line.trim()], steps: 1, rows: Number(stepLine![1]), calls: Number(stepLine![2] ?? 0) };
      members.push(current);
    } else if (current && NOTE_LINE.test(line)) {
      current.lines.push(line);
      current = undefined;
    } else {
      current = undefined;
      kept.push(line);
    }
  }
  return { members, rest: kept.join('\n').trim() };
}

/**
 * `messages` with each prompt's step notes folded (see above). `cite` takes the file's id and
 * text and returns its path; without a path that prompt's notes stay. Untouched rows come back as
 * the same objects.
 */
export function foldStepNotes<M extends Message>(
  messages: readonly M[],
  preserveRecentMessages: number,
  cite: (id: string, text: string) => string | undefined,
): { messages: M[]; folded: number } {
  const total = messages.length;
  const out: M[] = [];
  let folded = 0;
  let start = 0;
  while (start < total) {
    // One prompt's work: the prompt row (if any) and every row up to the next prompt.
    let end = start + 1;
    while (end < total && !isPromptRow(messages[end] as M)) end += 1;
    const rows = messages.slice(start, end);
    const parts = rows.map((m, k) =>
      // A tool-call row counts too: an earlier compaction may have merged a note into it (resumeSafeMessages).
      m.role === 'assistant' && !isPinned(start + k, total, preserveRecentMessages)
        ? split(m.text)
        : { members: [] as Member[], rest: m.text },
    );
    const members = parts.flatMap((p) => p.members);
    const prompt = isPromptRow(rows[0] as M) ? (rows[0] as M).text.trim().slice(0, PROMPT_HEAD_CHARS) : '';
    const path =
      members.length > 1
        ? (() => {
            const text = `${prompt ? `prompt: ${prompt}\n\n` : ''}${members.map((x) => x.lines.join('\n')).join('\n\n')}`;
            return cite(`stepnotes-${sha256Hex(text).slice(0, 12)}`, text);
          })()
        : undefined;
    if (!path) {
      out.push(...rows);
      start = end;
      continue;
    }
    const sum = (key: 'steps' | 'rows' | 'calls') => members.reduce((a, x) => a + x[key], 0);
    const note = {
      role: 'assistant',
      text: `${MARKER}archived ${sum('steps')} steps of this prompt's work (${sum('rows')} rows, ${sum('calls')} tool calls); each step's note and file: ${path} (Read it if needed)]`,
      toolUses: [],
    } as unknown as M;
    let placed = false;
    rows.forEach((m, k) => {
      const part = parts[k]!;
      if (part.members.length === 0) {
        out.push(m);
        return;
      }
      if (!placed) {
        out.push(note);
        placed = true;
      }
      if (part.rest.length > 0 || m.toolUses.length > 0) {
        const copy = { ...m, text: part.rest } as M & { handle?: string };
        delete copy.handle;
        out.push(copy);
      }
    });
    folded += members.length;
    start = end;
  }
  return folded > 0 ? { messages: out, folded } : { messages: messages as M[], folded: 0 };
}
