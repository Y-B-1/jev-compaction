import { messageChars, reductionRatio, splitsSurrogatePair } from './compact.js';
import { MAX_QUESTIONS_PER_REQUEST, noulAnswer } from './request.js';
import { ERROR_LINE, SCAN, salvageBlock } from './salvage.js';
import { sha256Hex } from './sha256.js';
import { MARKER, collectToolCalls, startsWithMarker, estimateTokens, goalFromMessages, isPinned, truncate } from './state.js';
import type { ArchiveResult, CallDecision, CompactResult, JevAsker, JevQuestions, Message, ToolUse } from './types.js';

/**
 * Clause splitter for assistant prose and prose tool inputs, plus rule-based
 * elision of code bodies. Pure string work: no Node, timer or Intl APIs, so it
 * runs in the hook's JavaScriptCore runtime.
 */

/** A slice `[start, end)` of the text. Units tile the text; `scored` units may be asked about. */
export type ProseUnit = { start: number; end: number; scored: boolean };

// SCAN is /g; a non-global copy keeps `test` free of lastIndex state.
const SCAN_ONCE = new RegExp(SCAN.source);
const LIST_MARKER = /^[ \t]*(?:[-*+]|\d+[.)])[ \t]+/;
const FENCE = /^[ \t]*(?:```|~~~)/;
const ABBREVIATIONS = new Set(['e.g', 'i.e', 'etc', 'vs', 'cf']);
const QUOTED = /"[^"\n]{4,}"|“[^”\n]{4,}”|(?<![\w])'[^'\n]{4,}'(?![\w])/;
const BOUND =
  /\b(?:not|no|never|none|nothing|neither|nor|without|cannot|if|unless|only|except|until|before|after|when|otherwise|instead|but|however|although|though|yet|because|since|so that|must|needs?|required|always|avoid)\b|n't\b|n’t\b/i;
const BOUND_START = new RegExp(`^(?:${BOUND.source})`, 'i');

/** Whether `s` holds any protected span (P3, P5-P9; line-level spans are handled by the caller). */
function isProtected(s: string): boolean {
  if (s.includes('`') || SCAN_ONCE.test(s) || QUOTED.test(s)) return true;
  for (const raw of s.split(/\s+/)) {
    const lead = raw.replace(/^[("'[{“‘]+/, '');
    if (/^--?[A-Za-z]/.test(lead)) return true;
    const token = lead.replace(/[.,;:!?)"'\]}”’]+$/, '');
    if (token === '' || ABBREVIATIONS.has(token.toLowerCase())) continue;
    if (/\d/.test(token) || /[._/]|::/.test(token) || /[a-z][A-Z]/.test(token)) return true;
  }
  return false;
}

/** Whole-line protection: P1 (fence lines are tracked by the caller), P2, P4, P8, P9 and the marker line. */
function isProtectedLine(line: string): boolean {
  return (
    /^(?: {4,}|\t)/.test(line) ||
    /^ {0,3}[|#]/.test(line) ||
    /^[ \t]*[$%] /.test(line) ||
    startsWithMarker(line) ||
    ERROR_LINE.test(line) ||
    !/[A-Za-z]/.test(line)
  );
}

type Span = { cs: number; ce: number; scored: boolean };

/** Sentence ranges `[from, to)` inside `text`, ending at . ! ? plus whitespace, except after an abbreviation. */
function sentences(text: string, from: number, to: number): [number, number][] {
  const out: [number, number][] = [];
  let begin = from;
  const end = /[.!?]+["')\]]*(?=\s|$)/g;
  const body = text.slice(0, to);
  end.lastIndex = from;
  for (let m = end.exec(body); m !== null; m = end.exec(body)) {
    const stop = m.index + m[0].length;
    const word = body.slice(begin, m.index + (m[0].startsWith('.') ? 0 : 1)).split(/\s+/).pop() ?? '';
    if (m[0] === '.' && ABBREVIATIONS.has(word.toLowerCase())) continue;
    out.push([begin, stop]);
    begin = stop;
    while (begin < to && /\s/.test(text.charAt(begin))) begin++;
  }
  if (begin < to) out.push([begin, to]);
  return out;
}

function wordCount(s: string): number {
  return s.split(/\s+/).filter((w) => /\w/.test(w)).length;
}

/** Cut points (exclusive end of each clause) inside `[from, to)` of an unbound sentence. */
function clauseCuts(text: string, from: number, to: number): number[] {
  const body = text.slice(from, to);
  const cuts: number[] = [];
  for (const m of body.matchAll(/;(?=\s)|(?<=\s)(?:—|–|--?)(?=\s)|,(?=\s+(?:and|so|then|also|plus|which)\b)/g)) {
    cuts.push(from + m.index + m[0].length);
  }
  return cuts;
}

function sentenceSpans(text: string, from: number, to: number): Span[] {
  // Protection wins: a sentence holding a protected span is one kept unit.
  if (isProtected(text.slice(from, to))) return [{ cs: from, ce: to, scored: false }];
  const units: [number, number][] = [];
  let at = from;
  // Opener: a short lead-in ending in , or ; stands alone, unless it holds a bound word.
  const lead = /^[^,;\n]*[,;](?=\s)/.exec(text.slice(from, to));
  const rest = lead ? text.slice(from + lead[0].length, to).trimStart() : '';
  if (lead && wordCount(lead[0]) <= 4 && !BOUND.test(lead[0]) && rest !== '' && !BOUND_START.test(rest)) {
    units.push([from, from + lead[0].length]);
    at = from + lead[0].length;
    while (at < to && /\s/.test(text.charAt(at))) at++;
  }
  if (!BOUND.test(text.slice(at, to))) {
    const pieces: [number, number][] = [];
    let begin = at;
    for (const cut of [...clauseCuts(text, at, to), to]) {
      if (cut <= begin) continue;
      pieces.push([begin, cut]);
      begin = cut;
      while (begin < to && /\s/.test(text.charAt(begin))) begin++;
    }
    // A clause under 3 words merges into the next one, or into the previous at the end.
    for (let i = 0; i < pieces.length && pieces.length > 1; ) {
      const piece = pieces[i] as [number, number];
      if (wordCount(text.slice(piece[0], piece[1])) >= 3) {
        i++;
      } else if (i + 1 < pieces.length) {
        pieces.splice(i, 2, [piece[0], (pieces[i + 1] as [number, number])[1]]);
      } else {
        pieces.splice(i - 1, 2, [(pieces[i - 1] as [number, number])[0], piece[1]]);
        i--;
      }
    }
    units.push(...pieces);
  } else if (at < to) {
    units.push([at, to]);
  }
  return units.map(([cs, ce]) => ({ cs, ce, scored: true }));
}

/**
 * Splits `text` into units that tile it exactly. A unit holds its own content, the
 * spaces that follow it on its line, and the line break, indent and list marker that
 * lead its line when it starts one. `scored` is false for protected units.
 */
export function splitProse(text: string): ProseUnit[] {
  const spans: Span[] = [];
  let fenced = false;
  let pos = 0;
  while (pos <= text.length) {
    const nl = text.indexOf('\n', pos);
    const lineEnd = nl === -1 ? text.length : nl;
    const line = text.slice(pos, lineEnd).replace(/\r$/, '');
    const trimmed = line.trim();
    if (trimmed !== '') {
      const cs = pos + line.indexOf(trimmed);
      const ce = cs + trimmed.length;
      if (FENCE.test(line)) {
        fenced = !fenced;
        spans.push({ cs, ce, scored: false });
      } else if (fenced || isProtectedLine(line)) {
        spans.push({ cs, ce, scored: false });
      } else {
        const lead = LIST_MARKER.exec(line)?.[0].length ?? line.length - line.trimStart().length;
        for (const [from, to] of sentences(text, pos + lead, ce)) {
          spans.push(...sentenceSpans(text, from, to));
        }
      }
    }
    if (nl === -1) break;
    pos = nl + 1;
  }
  return spans.map((s, i) => {
    const prev = spans[i - 1];
    const start = i === 0 ? 0 : text.slice((prev as Span).ce, s.cs).includes('\n') ? (prev as Span).ce : s.cs;
    const next = spans[i + 1];
    const end = next === undefined ? text.length : text.slice(s.ce, next.cs).includes('\n') ? s.ce : next.cs;
    return { start, end, scored: s.scored };
  });
}

const SENTENCE_END = /([.!?])["')\]]*$/;
const DANGLING = /(?:[,;]|\s(?:—|–|--?))([ \t]*)$/;

/**
 * The text with every unit `keep` rejects removed. A kept unit keeps its bytes; a dropped unit
 * takes its trailing separator; a dangling `,` `;` or dash at a cut edge takes the sentence's end mark;
 * a list item left with no kept unit loses its marker; 3+ newlines collapse to 2.
 */
export function reassemble(text: string, units: ProseUnit[], keep: (u: ProseUnit) => boolean): string {
  const kept = units.map((u) => keep(u));
  if (kept.every(Boolean)) return text;
  const slice = (i: number): string => text.slice((units[i] as ProseUnit).start, (units[i] as ProseUnit).end);
  const startsBlock = (i: number): boolean => i === 0 || /^\s*\n/.test(slice(i));
  let out = '';
  let dropped = false;
  for (let i = 0; i < units.length; i++) {
    let piece = slice(i);
    if (!kept[i]) {
      dropped = true;
      // A block whose first unit goes but later ones stay keeps its marker, or its break and indent.
      if (startsBlock(i)) {
        let j = i + 1;
        while (j < units.length && !startsBlock(j)) j++;
        const marker = /^\s*(?:[-*+]|\d+[.)])[ \t]+/.exec(piece)?.[0] ?? /^\s*/.exec(piece)?.[0];
        if (marker && kept.slice(i + 1, j).some(Boolean)) out += marker;
      }
      continue;
    }
    if (dropped && /^\s*\n/.test(piece)) out = out.replace(/[ \t]+$/, '');
    dropped = false;
    if (i + 1 < units.length && !kept[i + 1] && DANGLING.test(piece)) {
      const mark = endMark(i);
      if (mark) piece = piece.replace(DANGLING, (_m, ws: string) => `${mark}${ws}`);
    }
    out += piece;
  }
  if (dropped) out = out.replace(/[ \t]+$/, '');
  if (!kept[0]) out = out.replace(/^\n+/, '');
  return out.replace(/\n{3,}/g, '\n\n');

  /** The end mark for a cut sentence, or '' when a kept unit continues it. */
  function endMark(i: number): string {
    for (let j = i + 1; j < units.length; j++) {
      if (kept[j]) return '';
      const mark = SENTENCE_END.exec(slice(j).trimEnd())?.[1];
      if (mark) return mark;
      if (j + 1 >= units.length || startsBlock(j + 1)) return '.';
    }
    return '.';
  }
}

/** Body spans of the heredocs in a shell command. The command line and the closing marker lie outside. */
export function heredocBodies(command: string): { start: number; end: number }[] {
  const spans: { start: number; end: number }[] = [];
  const queue: { mark: string; dash: boolean }[] = [];
  let current: { mark: string; dash: boolean; bodyStart: number } | null = null;
  let pos = 0;
  while (pos < command.length) {
    const nl = command.indexOf('\n', pos);
    const lineEnd = nl === -1 ? command.length : nl;
    const next = nl === -1 ? command.length : nl + 1;
    const line = command.slice(pos, lineEnd).replace(/\r$/, '');
    if (current) {
      if ((current.dash ? line.replace(/^\t+/, '') : line) === current.mark) {
        spans.push({ start: current.bodyStart, end: pos });
        const following = queue.shift();
        current = following ? { ...following, bodyStart: next } : null;
      }
    } else {
      for (const m of line.matchAll(/(?<!<)<<(?!<)(-?)[ \t]*(?:'([^']+)'|"([^"]+)"|\\?([A-Za-z_]\w*))/g)) {
        queue.push({ mark: (m[2] ?? m[3] ?? m[4]) as string, dash: m[1] === '-' });
      }
      const first = queue.shift();
      if (first) current = { ...first, bodyStart: next };
    }
    pos = next;
  }
  if (current && current.bodyStart < command.length) spans.push({ start: current.bodyStart, end: command.length });
  return spans;
}

function withCommas(n: number): string {
  return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

export const ELIDED_MARKER = '[jev-compaction elided ';

/**
 * A code body cut to its first `headChars`, a note naming what was elided (`label` is
 * "Tool.field for path"), and a salvage block for the tail. The original when that is not shorter.
 */
export function elideBody(
  text: string,
  label: string,
  headChars: number,
  salvageChars: number,
  archivedPath?: string,
): string {
  // A body an earlier pass already cut stays as it is: cutting it again would
  // archive the stub over the full text saved the first time.
  if (text.length <= headChars + 120 || text.includes(ELIDED_MARKER) || text.includes('[fast-jev-compaction elided ')) return text;
  const cut = Math.max(0, splitsSurrogatePair(text, headChars) ? headChars - 1 : headChars);
  const head = cut > 0 ? `${text.slice(0, cut)}\n` : '';
  const note = `${ELIDED_MARKER}${withCommas(text.length - cut)} chars of ${label}${
    archivedPath ? `; full output: ${archivedPath}` : '; Read the file if needed'
  }]`;
  const salvaged = salvageBlock(text.slice(cut), salvageChars);
  const out = salvaged ? `${head}${note}\n${salvaged}` : `${head}${note}`;
  return out.length < text.length ? out : text;
}

/** Keep cut-offs `applyProse` tries in turn, mildest first: a unit is dropped when its noul is below the cut-off. */
export const PROSE_KEEP_LADDER: readonly number[] = [0.05, 0.15, 0.35];

export type ProseTrimOptions = {
  preserveRecentMessages: number;
  truncateHeadChars: number;
  salvageChars: number;
  maxRequestTokens: number;
  goal: string;
  /** Library only, not a plugin option. Estimated tokens per window. Default 20000. */
  windowTokens?: number;
  /** Saves a cut code body whole; the returned path is cited in the marker. Undefined (or a result of undefined) leaves the marker as it is. */
  archive?: ArchiveResult;
};

/**
 * Archives the original of a message whose prose a rung cuts and returns the path its marker will
 * name; undefined leaves that message's prose whole. The id is derived from the original, so a
 * later call with the same original returns the same path.
 */
export type ProseCite = (id: string, text: string) => string | undefined;

/** A prose cut must save more than its marker plus this margin, or the message stays whole. */
const MARKER_MARGIN = 120;

export type ProseApplied = {
  messages: Message[];
  decisions: ProseDecision[];
  stats: ProseStats;
  /** The originals this rung cites (only with a `cite`); hand them to the real archive writer once the rung is chosen. */
  archived: { id: string; text: string; path: string }[];
};

/** One scored unit's outcome. `message` is the index in the input messages. */
export type ProseDecision = { id: string; message: number; noul?: number; kept: boolean };

export type ProseStats = {
  scored: number;
  dropped: number;
  inputsElided: number;
  proseCharsSaved: number;
  inputCharsSaved: number;
  requests: number;
  failedRequests: number;
  failed?: string;
};

/** Pairs each tool call with its decision by position; `decisions` must come from the same messages. */
export function callActions(
  messages: readonly Message[],
  decisions: readonly CallDecision[],
  preserveRecentMessages: number,
): Map<string, CallDecision> {
  const calls = collectToolCalls(messages, preserveRecentMessages);
  if (calls.length !== decisions.length) {
    throw new Error(`${decisions.length} decisions for ${calls.length} tool calls`);
  }
  return new Map(calls.map((call, i) => [call.tool_use_id, decisions[i] as CallDecision]));
}

const PROSE_CONTEXT =
  'A coding assistant conversation is being compacted. `window` is a stretch of it, oldest first. Assistant text and long tool-call prompts are split into numbered parts: `c` parts may be deleted, `k` parts always stay. Each question asks whether one `c` part must stay. A deleted part is gone for good; everything else stays word for word.';

const DEFAULT_WINDOW_TOKENS = 20_000;
const CONCURRENCY = 4;
const OVERHEAD_TOKENS = 20;
const STATE_PLUS_QUESTION_TOKENS = 32_000;
const PROSE_INPUT_MIN_CHARS = 200;
const CONTEXT_TEXT_CHARS = 600;
const TOOL_INPUT_CHARS = 60;
const PROSE_INPUTS: Record<string, string> = { Agent: 'prompt', Task: 'prompt', SendMessage: 'message' };
const BODY_FIELDS: Record<string, string[]> = {
  Write: ['content'],
  Edit: ['old_string', 'new_string'],
  NotebookEdit: ['new_source'],
  Workflow: ['script'],
};

type Slot = { id: string; unit: ProseUnit; source: string; input: number };
type Elision = { input: Record<string, unknown>; fields: number; saved: number };
type Plan = {
  text: ProseUnit[];
  /** Prose inputs by tool-use index: the field, its text and its units. */
  prose: Map<number, { field: string; value: string; units: ProseUnit[] }>;
  elided: Map<number, Elision>;
  slots: Slot[];
};

function pathOf(input: Record<string, unknown>): string | undefined {
  for (const key of ['file_path', 'notebook_path', 'path']) {
    const v = input[key];
    if (typeof v === 'string' && v !== '') return v;
  }
  return undefined;
}

/** The input of `tool` with its code bodies cut, or undefined when no body is long enough. */
function elideInput(tool: ToolUse, head: number, salvage: number, archive?: ArchiveResult): Elision | undefined {
  const where = pathOf(tool.input);
  const label = (field: string): string => `${tool.tool}.${field}${where ? ` for ${where}` : ''}`;
  const input: Record<string, unknown> = { ...tool.input };
  let fields = 0;
  let saved = 0;
  const cut = (text: string, name: string): string => {
    let out = elideBody(text, label(name), head, salvage);
    if (out !== text) {
      const path = archive?.(`${tool.tool_use_id}-${name.replace(/[^A-Za-z0-9]+/g, '-')}-${fields + 1}`, text);
      if (path) {
        const cited = elideBody(text, label(name), head, salvage, path);
        if (cited !== text) out = cited;
      }
      fields++;
      saved += text.length - out.length;
    }
    return out;
  };
  for (const field of BODY_FIELDS[tool.tool] ?? []) {
    const v = input[field];
    if (typeof v === 'string') input[field] = cut(v, field);
  }
  if (tool.tool === 'MultiEdit' && Array.isArray(input['edits'])) {
    input['edits'] = input['edits'].map((edit: unknown) => {
      if (edit === null || typeof edit !== 'object') return edit;
      const copy = { ...(edit as Record<string, unknown>) };
      for (const field of ['old_string', 'new_string']) {
        const v = copy[field];
        if (typeof v === 'string') copy[field] = cut(v, field);
      }
      return copy;
    });
  }
  const command = input['command'];
  if (tool.tool === 'Bash' && typeof command === 'string') {
    let rebuilt = '';
    let at = 0;
    for (const span of heredocBodies(command)) {
      const original = command.slice(span.start, span.end);
      let out = cut(original, 'command heredoc');
      if (out !== original && original.endsWith('\n') && !out.endsWith('\n')) out += '\n';
      rebuilt += command.slice(at, span.start) + out;
      at = span.end;
    }
    input['command'] = rebuilt + command.slice(at);
  }
  return fields > 0 ? { input, fields, saved } : undefined;
}

type Batch = { state: object; questions: JevQuestions; ids: string[]; error?: string };

function question(path: string): JevQuestions[string] {
  return {
    type: 'noul',
    instructions: `Part \`${path}\` must stay: deleting it would lose information that the work in \`goal\` still needs and that no other part of \`window\` states`,
    criteria: {
      true: 'It states something specific a later reader needs: a finding, a decision or its reason, a result, a change made, a failure, a limit or condition on an action, a step still to do, or an answer to the user',
      false:
        'It adds nothing of its own: an acknowledgement, greeting or praise; narration of what the assistant is about to do or just did that the tool calls already record; a repeat of another part; hedging, apology or politeness; an offer of more help',
    },
  };
}

/** Jev's scores for the prose units of one message list. Reapply them at any cut-off with `applyProse`. */
export type ProseScores = {
  messages: readonly Message[];
  plans: ReadonlyMap<number, Plan>;
  /** Jev's noul by unit id; absent when an answer was missing or invalid. */
  noul: ReadonlyMap<string, number | undefined>;
  stats: ProseStats;
};

/**
 * Scores the clauses of old assistant text and prose tool inputs with Jev, in one round of requests,
 * and plans the rule-based cut of old code bodies. Nothing is dropped yet: `applyProse` does that at
 * a keep cut-off, as often as wanted, without further requests. User messages, pinned messages and
 * non-body fields are never planned. When every request fails, `stats.failed` is set.
 */
export async function scoreProse(
  messages: readonly Message[],
  actions: ReadonlyMap<string, CallDecision>,
  asker: JevAsker,
  options: ProseTrimOptions,
): Promise<ProseScores> {
  const stats: ProseStats = {
    scored: 0,
    dropped: 0,
    inputsElided: 0,
    proseCharsSaved: 0,
    inputCharsSaved: 0,
    requests: 0,
    failedRequests: 0,
  };
  const unchanged = (failed: string): ProseScores => ({
    messages,
    plans: new Map(),
    noul: new Map(),
    stats: { ...stats, failed },
  });
  const windowTokens = options.windowTokens ?? DEFAULT_WINDOW_TOKENS;
  const total = messages.length;
  const pinned = (i: number): boolean => isPinned(i, total, options.preserveRecentMessages);
  const resultAt = new Map<string, number>();
  messages.forEach((m, i) => m.toolResults?.forEach((r) => resultAt.set(r.tool_use_id, i)));

  const plans = new Map<number, Plan>();
  let counter = 0;
  const slotsOf = (units: ProseUnit[], source: string, input: number, out: Slot[]): void => {
    for (const unit of units) {
      const id = `${unit.scored ? 'c' : 'k'}${++counter}`;
      out.push({ id, unit, source, input });
    }
  };
  try {
    messages.forEach((m, i) => {
      if (m.role !== 'assistant' || pinned(i)) return;
      const plan: Plan = { text: [], prose: new Map(), elided: new Map(), slots: [] };
      if (m.text.trim() !== '') {
        plan.text = splitProse(m.text);
        slotsOf(plan.text, m.text, -1, plan.slots);
      }
      m.toolUses.forEach((tool, k) => {
        const action = actions.get(tool.tool_use_id);
        if (!action || action.reason === 'pinned' || (action.action !== 'keep' && action.action !== 'drop_result')) return;
        const at = resultAt.get(tool.tool_use_id);
        if (at !== undefined && pinned(at)) return;
        const field = PROSE_INPUTS[tool.tool];
        const value = field === undefined ? undefined : tool.input[field];
        if (field !== undefined && typeof value === 'string') {
          if (value.length < PROSE_INPUT_MIN_CHARS) return;
          const units = splitProse(value);
          plan.prose.set(k, { field, value, units });
          slotsOf(units, value, k, plan.slots);
          return;
        }
        const elision = elideInput(tool, options.truncateHeadChars, options.salvageChars, options.archive);
        if (elision) plan.elided.set(k, elision);
      });
      if (plan.slots.length > 0 || plan.elided.size > 0) plans.set(i, plan);
    });
  } catch (error) {
    return unchanged(error instanceof Error ? error.message : String(error));
  }

  // Windows: chunks of scored messages with the messages between them as context.
  const goal = options.goal || goalFromMessages(messages);
  type Entry = { key: string; value: Record<string, unknown>; tokens: number; ids: string[] };
  const entryOf = (i: number, slots: Slot[], first: boolean): Entry => {
    const m = messages[i] as Message;
    const value: Record<string, unknown> = { role: 'assistant' };
    if (first && m.toolUses.length > 0) {
      value['tools'] = m.toolUses.map((t) => `${t.tool} ${truncate(JSON.stringify(t.input) ?? '', TOOL_INPUT_CHARS)}`.trim()).join(' | ');
    }
    const objects = new Map<number, Record<string, unknown>>();
    const ids: string[] = [];
    for (const slot of slots) {
      let target = value;
      if (slot.input >= 0) {
        let obj = objects.get(slot.input);
        if (!obj) {
          obj = { tool: (m.toolUses[slot.input] as ToolUse).tool };
          objects.set(slot.input, obj);
          value[`a${slot.input + 1}`] = obj;
        }
        target = obj;
      }
      target[slot.id] = slot.source.slice(slot.unit.start, slot.unit.end).trim();
      if (slot.unit.scored) ids.push(slot.id);
    }
    return { key: `m${i}`, value, tokens: estimateTokens(JSON.stringify(value)), ids };
  };
  const windows: Entry[][] = [];
  let current: Entry[] = [];
  let currentTokens = 0;
  let pending: Entry[] = [];
  messages.forEach((m, i) => {
    const plan = plans.get(i);
    if (!plan || !plan.slots.some((s) => s.unit.scored)) {
      if (m.text.trim() !== '') {
        const value = { role: m.role, text: truncate(m.text, CONTEXT_TEXT_CHARS) };
        pending.push({ key: `m${i}`, value, tokens: estimateTokens(JSON.stringify(value)), ids: [] });
      }
      return;
    }
    // Split an oversize message at paragraph starts (any unit once it is twice the limit).
    const chunks: Slot[][] = [[]];
    let size = 0;
    for (const slot of plan.slots) {
      const tokens = estimateTokens(slot.source.slice(slot.unit.start, slot.unit.end)) + 4;
      const paragraph = /\n\s*\n/.test(slot.source.slice(slot.unit.start, slot.unit.end).match(/^\s*/)?.[0] ?? '');
      if ((chunks[chunks.length - 1] as Slot[]).length > 0 && ((size >= windowTokens && paragraph) || size >= 2 * windowTokens)) {
        chunks.push([]);
        size = 0;
      }
      (chunks[chunks.length - 1] as Slot[]).push(slot);
      size += tokens;
    }
    chunks.forEach((slots, c) => {
      const entry = entryOf(i, slots, c === 0);
      const group = [...pending, entry];
      pending = [];
      const tokens = group.reduce((sum, e) => sum + e.tokens, 0);
      if (current.length > 0 && currentTokens + tokens > windowTokens) {
        windows.push(current);
        current = [];
        currentTokens = 0;
      }
      current.push(...group);
      currentTokens += tokens;
    });
  });
  if (current.length > 0) windows.push(current);

  const slotPath = new Map<string, string>();
  for (const [i, plan] of plans) {
    for (const slot of plan.slots) {
      slotPath.set(slot.id, `window.m${i}.${slot.input >= 0 ? `a${slot.input + 1}.` : ''}${slot.id}`);
    }
  }
  const batches: Batch[] = [];
  for (const entries of windows) {
    const window: Record<string, unknown> = {};
    for (const e of entries) {
      const prev = window[e.key];
      window[e.key] = prev && typeof prev === 'object' ? { ...(prev as object), ...e.value } : e.value;
    }
    const state = { context: PROSE_CONTEXT, goal, window };
    const stateTokens = estimateTokens(JSON.stringify(state));
    const budget = options.maxRequestTokens - stateTokens - OVERHEAD_TOKENS;
    let batch: Batch = { state, questions: {}, ids: [] };
    let used = 0;
    for (const id of entries.flatMap((e) => e.ids)) {
      const q = question(slotPath.get(id) as string);
      const tokens = estimateTokens(JSON.stringify(q));
      if (stateTokens + tokens + OVERHEAD_TOKENS > STATE_PLUS_QUESTION_TOKENS || tokens > budget) {
        batches.push({ state, questions: {}, ids: [id], error: `state of ~${stateTokens} tokens leaves no room for a question` });
        continue;
      }
      if (batch.ids.length > 0 && (used + tokens > budget || batch.ids.length === MAX_QUESTIONS_PER_REQUEST)) {
        batches.push(batch);
        batch = { state, questions: {}, ids: [] };
        used = 0;
      }
      batch.questions[id] = q;
      batch.ids.push(id);
      used += tokens;
    }
    if (batch.ids.length > 0) batches.push(batch);
  }

  const noul = new Map<string, number | undefined>();
  let failed = 0;
  let lastFailure: unknown;
  let cursor = 0;
  const worker = async (): Promise<void> => {
    while (cursor < batches.length) {
      const batch = batches[cursor++] as Batch;
      try {
        if (batch.error) throw new Error(batch.error);
        const { answers } = await asker.ask(batch.state, batch.questions);
        for (const id of batch.ids) {
          try {
            noul.set(id, noulAnswer(answers, id));
          } catch {
            // A missing or invalid answer keeps the unit.
          }
        }
      } catch (error) {
        failed++;
        lastFailure = error;
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, batches.length) }, worker));
  stats.requests = batches.length;
  stats.failedRequests = failed;
  if (failed > 0 && failed === batches.length) {
    return unchanged(lastFailure instanceof Error ? lastFailure.message : String(lastFailure));
  }

  return { messages, plans, noul, stats };
}

/**
 * Applies `scores` at a keep cut-off: a unit whose noul is below `keepAt` is dropped, a unit with
 * no answer stays, and the planned code-body cuts are made. Untouched messages and tool uses come
 * back as the same objects. Scores that failed completely return the input unchanged.
 */
export function applyProse(
  scores: ProseScores,
  keepAt: number,
  cite?: ProseCite,
): ProseApplied {
  const { messages, plans, noul } = scores;
  const stats: ProseStats = {
    ...scores.stats,
    scored: 0,
    dropped: 0,
    inputsElided: 0,
    proseCharsSaved: 0,
    inputCharsSaved: 0,
  };
  const archived: ProseApplied['archived'] = [];
  if (stats.failed !== undefined) return { messages: [...messages], decisions: [], stats, archived };
  const decisions: ProseDecision[] = [];
  const dropped = new Set<ProseUnit>();
  for (const [i, plan] of plans) {
    for (const slot of plan.slots) {
      if (!slot.unit.scored) continue;
      const n = noul.get(slot.id);
      const kept = n === undefined || n >= keepAt;
      decisions.push({ id: slot.id, message: i, ...(n === undefined ? {} : { noul: n }), kept });
      if (!kept) dropped.add(slot.unit);
    }
  }
  decisions.sort((a, b) => a.message - b.message || Number(a.id.slice(1)) - Number(b.id.slice(1)));
  stats.scored = decisions.length;
  stats.dropped = dropped.size;

  const out: Message[] = [];
  messages.forEach((m, i) => {
    const plan = plans.get(i);
    if (!plan) {
      out.push(m);
      return;
    }
    let text = plan.text.length > 0 ? reassemble(m.text, plan.text, (u) => !dropped.has(u)) : m.text;
    const values = new Map<number, string>();
    for (const [k, prose] of plan.prose) {
      if (plan.elided.has(k)) continue;
      const value = reassemble(prose.value, prose.units, (u) => !dropped.has(u));
      if (value !== prose.value) values.set(k, value);
    }
    let saved = m.text.length - text.length;
    for (const [k, value] of values) saved += (plan.prose.get(k) as { value: string }).value.length - value.length;
    if (cite && saved > 0) {
      // A prose cut goes ahead only with the original archived and cited, and only when it saves
      // more than the marker costs; otherwise this message's prose stays whole.
      const original = proseOriginal(m, plan, values);
      const id = `reply-${sha256Hex(original).slice(0, 12)}`;
      const path = cite(id, original);
      const marker = path ? `${MARKER}trimmed ${withCommas(saved)} chars of this reply; full text: ${path} (Read it if needed)]` : '';
      if (path && saved > marker.length + 1 + MARKER_MARGIN) {
        const cost = text.length > 0 ? marker.length + 1 : marker.length;
        text = text.length > 0 ? `${text}\n${marker}` : marker;
        archived.push({ id, text: original, path });
        saved -= cost;
      } else {
        text = m.text;
        values.clear();
        saved = 0;
      }
    }
    stats.proseCharsSaved += saved;
    let changed = text !== m.text;
    const toolUses = m.toolUses.map((tool, k) => {
      const elision = plan.elided.get(k);
      const prose = plan.prose.get(k);
      let input: Record<string, unknown> | undefined;
      if (elision) {
        input = elision.input;
        stats.inputsElided += elision.fields;
        stats.inputCharsSaved += elision.saved;
      } else if (prose) {
        const value = values.get(k);
        if (value === undefined) return tool;
        input = { ...tool.input, [prose.field]: value };
      }
      if (!input) return tool;
      changed = true;
      const copy: ToolUse = { tool_use_id: tool.tool_use_id, tool: tool.tool, input };
      if (tool.text !== undefined) copy.text = tool.text;
      if (tool.isError !== undefined) copy.isError = tool.isError;
      return copy;
    });
    if (!changed) {
      out.push(m);
      return;
    }
    if (text.trim() === '' && toolUses.length === 0 && (m.toolResults?.length ?? 0) === 0) return;
    const rebuilt: Message = { role: m.role, text, toolUses };
    if (m.toolResults) rebuilt.toolResults = m.toolResults;
    out.push(rebuilt);
  });
  return { messages: out, decisions, stats, archived };
}

/** The text a prose cut archives: the message's text and each prose input it cut, as they were. */
function proseOriginal(m: Message, plan: Plan, values: ReadonlyMap<number, string>): string {
  const parts = [m.text];
  for (const k of values.keys()) {
    const prose = plan.prose.get(k) as { field: string; value: string };
    parts.push(`[${(m.toolUses[k] as ToolUse).tool} input "${prose.field}"]\n${prose.value}`);
  }
  return parts.filter((part) => part.length > 0).join('\n\n');
}

/**
 * Cuts the code bodies of old tool inputs (Write/Edit contents, notebook sources, workflow
 * scripts, Bash heredocs) by rule, with no Jev request: the same cut `applyProse` makes, run on
 * every compaction. Covers the calls `scoreProse` would plan; prose inputs, pinned messages and
 * dropped calls stay as they are. Untouched messages come back as the same objects.
 */
export function elideInputs(
  messages: readonly Message[],
  actions: ReadonlyMap<string, CallDecision>,
  options: Pick<ProseTrimOptions, 'preserveRecentMessages' | 'truncateHeadChars' | 'salvageChars' | 'archive'>,
): { messages: Message[]; fields: number; saved: number } {
  const total = messages.length;
  const pinned = (i: number): boolean => isPinned(i, total, options.preserveRecentMessages);
  const resultAt = new Map<string, number>();
  messages.forEach((m, i) => m.toolResults?.forEach((r) => resultAt.set(r.tool_use_id, i)));
  let fields = 0;
  let saved = 0;
  const out = messages.map((m, i) => {
    if (m.role !== 'assistant' || pinned(i)) return m;
    let changed = false;
    const toolUses = m.toolUses.map((tool) => {
      const action = actions.get(tool.tool_use_id);
      if (!action || action.reason === 'pinned' || (action.action !== 'keep' && action.action !== 'drop_result')) return tool;
      const at = resultAt.get(tool.tool_use_id);
      if (at !== undefined && pinned(at)) return tool;
      const field = PROSE_INPUTS[tool.tool];
      if (field !== undefined && typeof tool.input[field] === 'string') return tool;
      const elision = elideInput(tool, options.truncateHeadChars, options.salvageChars, options.archive);
      if (!elision) return tool;
      changed = true;
      fields += elision.fields;
      saved += elision.saved;
      const copy: ToolUse = { tool_use_id: tool.tool_use_id, tool: tool.tool, input: elision.input };
      if (tool.text !== undefined) copy.text = tool.text;
      if (tool.isError !== undefined) copy.isError = tool.isError;
      return copy;
    });
    return changed ? { ...m, toolUses } : m;
  });
  return { messages: out, fields, saved };
}

/**
 * Walks `PROSE_KEEP_LADDER` on one set of scores and takes the first cut-off whose result reaches
 * `minRatio` against the original `charsBefore` of `result`. Undefined when none does.
 */
export function ladderRescue(
  scores: ProseScores,
  result: CompactResult,
  minRatio: number,
  cite?: ProseCite,
): { cutoff: number; result: CompactResult; stats: ProseStats; archived: ProseApplied['archived'] } | undefined {
  for (const cutoff of PROSE_KEEP_LADDER) {
    const trimmed = applyProse(scores, cutoff, cite);
    const charsAfter = trimmed.messages.reduce((sum, message) => sum + messageChars(message), 0);
    const candidate = { ...result, messages: trimmed.messages, stats: { ...result.stats, charsAfter } };
    if (reductionRatio(candidate) >= minRatio) return { cutoff, result: candidate, stats: trimmed.stats, archived: trimmed.archived };
  }
  return undefined;
}
