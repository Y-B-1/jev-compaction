import { messageChars, reductionRatio, splitsSurrogatePair } from './compact.js';
import { PROSE_KEEP_LADDER } from './prose.js';
import { MAX_QUESTIONS_PER_REQUEST, noulAnswer } from './request.js';
import { ERROR_LINE, capped, redact, salvageErrorLines, salvageIdentifiers } from './salvage.js';
import { MARKER, collectToolCalls, estimateTokens, fitState, hasMarker, isPinned, resultPreview } from './state.js';
import type {
  CompactResult,
  JevAsker,
  JevQuestions,
  Message,
  NoulQuestion,
  ResolvedCompactOptions,
} from './types.js';

/**
 * Notification trim: shrinks the finished-worker reports (`<task-notification>` user messages) that
 * tool compaction and prose trim cannot touch. Two parts, like `src/prose.ts`: `scoreNotifications`
 * asks Jev once, offline `applyNotifications` applies a rung of the ladder. Nothing is reworded:
 * every character of output is a slice of the input, a marker, or a salvage block.
 */

/** A keep cut-off of `PROSE_KEEP_LADDER`, or `'all'`: fold every eligible report without waiting for Jev. */
export type Rung = number | 'all';

/** The prose ladder, then a final rung that folds every eligible notification. */
export const NOTIFICATION_RUNGS: readonly Rung[] = [...PROSE_KEEP_LADDER, 'all'];

/** Names the archive file for `id` and returns its path, or undefined when nothing can be archived. */
export type NotificationCite = (id: string, text: string) => string | undefined;

export type NotificationOptions = Pick<
  ResolvedCompactOptions,
  | 'preserveRecentMessages'
  | 'maxStateTokens'
  | 'maxRequestTokens'
  | 'maxConcurrentRequests'
  | 'truncateHeadChars'
  | 'resultPreviewChars'
  | 'salvageChars'
  | 'goal'
> & {
  /**
   * The conversation holds images: a typed prompt the engine still holds whole
   * (it carries a handle) may be one of them, and folding it would rebuild the
   * row from its text alone and lose the image, so such prompts stay.
   */
  imagesInContext?: boolean;
};

export type NotificationStats = {
  /** Notifications the stage may rewrite. */
  eligible: number;
  /** Notifications Jev gave a valid answer for. */
  scored: number;
  /** Notifications whose result body was folded at this rung. */
  folded: number;
  /** Notifications that only lost their usage and diagnostics at this rung. */
  stripped: number;
  /** Characters the rung removed. */
  charsSaved: number;
  requests: number;
  failedRequests: number;
  stateTokens: number;
  failed?: string;
};

type Entry = { index: number; key: string; archiveId: string; taskId: string; body: [number, number] };

/** Jev's scores for the notifications of one message list. Reapply them at any rung with `applyNotifications`. */
export type NotificationScores = {
  messages: readonly Message[];
  entries: readonly Entry[];
  /** Jev's noul by question key; absent when an answer was missing or invalid. */
  noul: ReadonlyMap<string, number | undefined>;
  headChars: number;
  salvageChars: number;
  stats: NotificationStats;
};

export type NotificationApplied = {
  /** The input messages with the rewritten notifications in place. */
  messages: Message[];
  /** Original notification message to its rewritten copy, for composing with another stage by identity. */
  replaced: ReadonlyMap<Message, Message>;
  /** The originals this rung cites; hand them to the real archive writer once the rung is chosen. */
  archived: { id: string; text: string; path: string }[];
  stats: NotificationStats;
};

const OPEN = '<task-notification>';
const ENVELOPE = /(<usage>[\s\S]*?<\/usage>|<diagnostics>[\s\S]*?<\/diagnostics>)(\r?\n)?/g;
const ARCHIVE_ID_DIR = '.claude/jev-compaction';
const OVERHEAD_TOKENS = 20;
const STATE_PLUS_QUESTION_TOKENS = 32_000;
/** A body this much longer than the head is worth folding (the same margin tool-result truncation uses). */
const FOLD_MARGIN = 120;

function eligible(message: Message, index: number, total: number, preserve: number): Entry | undefined {
  if (message.role !== 'user' || isPinned(index, total, preserve)) return undefined;
  if (message.toolUses.length > 0 || (message.toolResults?.length ?? 0) > 0) return undefined;
  const text = message.text;
  if (!text.trimStart().startsWith(OPEN) || hasMarker(text)) return undefined;
  if (text.indexOf(OPEN, text.indexOf(OPEN) + 1) !== -1) return undefined;
  const open = text.indexOf('<result>');
  if (open === -1 || /<status>([^<]*)<\/status>/.exec(text.slice(0, open))?.[1]?.trim() !== 'completed') return undefined;
  const id = /<task-id>([^<]+)<\/task-id>/.exec(text)?.[1];
  const close = text.lastIndexOf('</result>');
  if (id === undefined || open === -1 || close < open + '<result>'.length) return undefined;
  return { index, key: '', archiveId: '', taskId: id, body: [open + '<result>'.length, close] };
}

function question(entry: Entry, text: string, previewChars: number): JevQuestions[string] {
  const summary = /<summary>([\s\S]*?)<\/summary>/.exec(text)?.[1]?.trim() ?? '';
  const body = text.slice(entry.body[0], entry.body[1]);
  const preview = resultPreview(body, previewChars);
  const q: NoulQuestion = {
    type: 'noul',
    instructions: `The full report of background task \`${entry.taskId}\` (\`history\` entry ${entry.index}, ${body.length} chars) should stay in \`history\` verbatim: the assistant still needs its contents, and re-running the task would not do${
      summary ? `. Summary: ${summary}` : ''
    }${preview ? `. Report preview: ${preview}` : ''}`,
    // The wording of tool compaction's result question, for a report in place of a tool output.
    criteria: {
      true: 'The exact contents are still in use and could not be recovered by re-running the tool: an error the assistant is still diagnosing, output the user asked about, the current state of a file being edited',
      false:
        'The contents are stale, already stated in the assistant text, or trivially re-obtainable: a directory listing already used, a passing test run, a file read before it was rewritten',
    },
  };
  return q;
}

function unchanged(
  messages: readonly Message[],
  stats: NotificationStats,
  failed?: string,
): NotificationScores {
  return {
    messages,
    entries: [],
    noul: new Map(),
    headChars: 0,
    salvageChars: 0,
    stats: failed === undefined ? stats : { ...stats, failed },
  };
}

/** What `askNoul` returns: Jev's noul by question key (absent when an answer was missing or invalid). */
export type NoulAnswers = {
  noul: Map<string, number | undefined>;
  requests: number;
  failedRequests: number;
  stateTokens: number;
  /** Set when every request failed. */
  failed?: string;
};

/**
 * Asks Jev the noul `questions` about `messages`, in batches that fit the request budget, with the
 * whole conversation as state (tool compaction's state builder). Throws when the budget leaves no
 * room for the state and a question. A request that fails leaves its questions unanswered; when
 * every request fails, `failed` is set.
 */
export async function askNoul(
  messages: readonly Message[],
  questions: readonly (readonly [string, JevQuestions[string]])[],
  asker: JevAsker,
  options: Pick<NotificationOptions, 'preserveRecentMessages' | 'maxStateTokens' | 'maxRequestTokens' | 'maxConcurrentRequests' | 'goal'>,
): Promise<NoulAnswers> {
  const tokensOf = (q: JevQuestions[string]): number => estimateTokens(JSON.stringify(q));
  const largest = questions.reduce((m, [, q]) => Math.max(m, tokensOf(q)), 0);
  const stateBudget = options.maxRequestTokens - OVERHEAD_TOKENS - largest;
  if (stateBudget < 1) throw new Error('request budget leaves no room for state and questions');
  const fitted = fitState(messages, collectToolCalls(messages, options.preserveRecentMessages), {
    maxStateTokens: Math.min(options.maxStateTokens, stateBudget),
    preserveRecentMessages: options.preserveRecentMessages,
    goal: options.goal,
  });
  const budget = options.maxRequestTokens - fitted.tokens - OVERHEAD_TOKENS;
  const batches: (readonly [string, JevQuestions[string]])[][] = [];
  let current: (readonly [string, JevQuestions[string]])[] = [];
  let used = 0;
  for (const item of questions) {
    const tokens = tokensOf(item[1]);
    if (fitted.tokens + tokens + OVERHEAD_TOKENS > STATE_PLUS_QUESTION_TOKENS || tokens > budget) {
      throw new Error(`state of ~${fitted.tokens} tokens leaves no room for a question of ~${tokens}`);
    }
    if (current.length > 0 && (used + tokens > budget || current.length === MAX_QUESTIONS_PER_REQUEST)) {
      batches.push(current);
      current = [];
      used = 0;
    }
    current.push(item);
    used += tokens;
  }
  if (current.length > 0) batches.push(current);

  const noul = new Map<string, number | undefined>();
  let failed = 0;
  let lastFailure: unknown;
  let cursor = 0;
  const worker = async (): Promise<void> => {
    while (cursor < batches.length) {
      const batch = batches[cursor++] as (typeof batches)[number];
      try {
        const { answers } = await asker.ask(fitted.state, Object.fromEntries(batch));
        for (const [key] of batch) {
          try {
            noul.set(key, noulAnswer(answers, key));
          } catch {
            // A missing or invalid answer keeps the text.
          }
        }
      } catch (error) {
        failed++;
        lastFailure = error;
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(options.maxConcurrentRequests, batches.length) }, worker));
  return {
    noul,
    requests: batches.length,
    failedRequests: failed,
    stateTokens: fitted.tokens,
    ...(failed > 0 && failed === batches.length
      ? { failed: lastFailure instanceof Error ? lastFailure.message : String(lastFailure) }
      : {}),
  };
}

/**
 * Asks Jev, once per eligible notification, whether its report must stay verbatim. Eligible: an
 * unpinned user message that starts `<task-notification>`, carries a `<task-id>` and a `<result>`, has
 * status `completed` and holds no marker of ours. The state is the whole conversation (tool
 * compaction's state builder over `messages`), so Jev sees the turns that restate a report. Nothing is
 * changed yet. A request that fails leaves its notifications unscored; when every request fails,
 * `stats.failed` is set.
 */
export async function scoreNotifications(
  messages: readonly Message[],
  asker: JevAsker,
  options: NotificationOptions,
): Promise<NotificationScores> {
  const stats: NotificationStats = {
    eligible: 0,
    scored: 0,
    folded: 0,
    stripped: 0,
    charsSaved: 0,
    requests: 0,
    failedRequests: 0,
    stateTokens: 0,
  };
  try {
    const seen = new Map<string, number>();
    const entries: Entry[] = [];
    messages.forEach((message, index) => {
      const entry = eligible(message, index, messages.length, options.preserveRecentMessages);
      if (!entry) return;
      const safe = entry.taskId.replace(/[^A-Za-z0-9_-]+/g, '-');
      const n = (seen.get(safe) ?? 0) + 1;
      seen.set(safe, n);
      entries.push({ ...entry, key: `n${entries.length + 1}`, archiveId: `notif-${safe}${n > 1 ? `-${n}` : ''}` });
    });
    stats.eligible = entries.length;
    if (entries.length === 0) return unchanged(messages, stats);

    const questions = entries.map((e) => [e.key, question(e, (messages[e.index] as Message).text, options.resultPreviewChars)] as const);
    const asked = await askNoul(messages, questions, asker, options);
    stats.stateTokens = asked.stateTokens;
    stats.requests = asked.requests;
    stats.failedRequests = asked.failedRequests;
    stats.scored = [...asked.noul.values()].filter((v) => v !== undefined).length;
    if (asked.failed !== undefined) return unchanged(messages, stats, asked.failed);
    const noul = asked.noul;
    return { messages, entries, noul, headChars: options.truncateHeadChars, salvageChars: options.salvageChars, stats };
  } catch (error) {
    return unchanged(messages, stats, error instanceof Error ? error.message : String(error));
  }
}

/** Removes `<usage>` and `<diagnostics>` from `segment`; `marker`, when given, takes the place of the first one. */
function strip(segment: string, marker?: string): { text: string; removed: boolean } {
  let removed = false;
  const text = segment.replace(ENVELOPE, (_m, _el: string, nl: string | undefined, offset: number, whole: string) => {
    const ownLine = offset === 0 || whole.charAt(offset - 1) === '\n';
    const put = !removed && marker ? marker : '';
    removed = true;
    if (put) return `${put}${nl ?? ''}`;
    return ownLine ? '' : `${nl ?? ''}`;
  });
  return { text, removed };
}

/** The longest error clause kept from a line that is too long to keep whole. */
const CLAUSE_MAX = 160;
const CLAUSE_END = /\\n|\\r|\r|\n|[.!?](?=\s|$)/;

/**
 * The error items of `tail`. A short line that reports a failure is kept whole, as tool compaction
 * does. A longer line (a one-line body is the real shape: escaped JSON) keeps each error token with
 * its text instead, up to `CLAUSE_MAX` characters, a line break (real or escaped) or a sentence end.
 * Every item is a slice of `tail`.
 */
function errorItems(tail: string, budget: number): string[] {
  const max = Math.max(1, Math.min(CLAUSE_MAX, budget - 2));
  const items = new Set<string>();
  for (const raw of tail.split(/\r?\n/)) {
    const line = raw.trim();
    if (line.length <= max) {
      for (const item of salvageErrorLines(line)) items.add(item);
      continue;
    }
    let from = 0;
    while (from < line.length) {
      const hit = ERROR_LINE.exec(line.slice(from));
      if (!hit) break;
      let start = from + hit.index;
      while (start > from && /\w/.test(line.charAt(start - 1))) start--;
      const window = line.slice(start, start + max);
      const stop = CLAUSE_END.exec(window);
      let end = stop ? start + stop.index + (/[.!?]/.test(stop[0]) ? 1 : 0) : start + window.length;
      if (!stop && end - start === max && splitsSurrogatePair(line, end)) end--;
      const clause = line.slice(start, end).trim();
      if (clause) items.add(redact(clause));
      from = Math.max(end, from + hit.index + hit[0].length);
    }
  }
  return [...items];
}

/** The salvage block of the cut tail: the same `errors:` and `refs:` lines as `salvageBlock`, with `errorItems` in place of whole lines. */
function foldSalvage(tail: string, maxChars: number): string {
  if (maxChars <= 0) return '';
  try {
    const errors = capped(errorItems(tail, maxChars), maxChars);
    const spent = errors.reduce((sum, line) => sum + line.length + 2, 0);
    const refs = capped(salvageIdentifiers(tail), maxChars - spent);
    return [
      errors.length > 0 ? `errors: ${errors.join(' | ')}` : '',
      refs.length > 0 ? `refs: ${refs.join(' ')}` : '',
    ]
      .filter((line) => line.length > 0)
      .join('\n');
  } catch {
    return '';
  }
}

export function fold(body: string, headChars: number, salvageChars: number, say: (cut: number) => string): string {
  const cut = Math.max(0, splitsSurrogatePair(body, headChars) ? headChars - 1 : headChars);
  const head = cut > 0 ? `${body.slice(0, cut)}\n` : '';
  const salvaged = foldSalvage(body.slice(cut), salvageChars);
  return `${head}${say(body.length - cut)}\n${salvaged ? `${salvaged}\n` : ''}`;
}

const WHERE = (path: string): string => `full notification: ${path} (Read or grep it if needed)]`;

/** One notification rewritten, or undefined when no change would make it shorter. */
function rewrite(
  text: string,
  entry: Entry,
  wantFold: boolean,
  path: string,
  headChars: number,
  salvageChars: number,
): { text: string; folded: boolean } | undefined {
  const [from, to] = entry.body;
  const body = text.slice(from, to);
  if (wantFold && body.length > headChars + FOLD_MARGIN) {
    const pre = strip(text.slice(0, from));
    const post = strip(text.slice(to));
    const removed = pre.removed || post.removed;
    const folded = fold(body, headChars, salvageChars, (n) =>
      `${MARKER}folded ${n} chars of this task result${removed ? ' and removed its usage and diagnostics' : ''}; ${WHERE(path)}`,
    );
    const out = `${pre.text}${folded}${post.text}`;
    if (out.length < text.length) return { text: out, folded: true };
  }
  const line = `${MARKER}removed usage and diagnostics; ${WHERE(path)}`;
  const pre = strip(text.slice(0, from), line);
  const post = strip(text.slice(to), pre.removed ? undefined : line);
  if (!pre.removed && !post.removed) return undefined;
  const out = `${pre.text}${body}${post.text}`;
  return out.length < text.length ? { text: out, folded: false } : undefined;
}

/**
 * Applies `scores` at a rung. Every eligible notification loses its `<usage>` and `<diagnostics>`; one
 * whose noul is below the rung (every one at `'all'`) also has its result folded to a head, a marker
 * and a salvage block. A missing answer keeps the body. Before a notification changes, `cite` is given
 * its whole original and must return the archive path the marker will name; without `cite`, or when it
 * returns nothing, the notification is left as it is. Other messages come back as the same objects.
 */
export function applyNotifications(
  scores: NotificationScores,
  rung: Rung,
  cite?: NotificationCite,
): NotificationApplied {
  const stats: NotificationStats = { ...scores.stats, folded: 0, stripped: 0, charsSaved: 0 };
  const replaced = new Map<Message, Message>();
  const archived: NotificationApplied['archived'] = [];
  if (stats.failed === undefined && cite) {
    for (const entry of scores.entries) {
      const message = scores.messages[entry.index] as Message;
      const n = scores.noul.get(entry.key);
      const wantFold = rung === 'all' || (n !== undefined && n < rung);
      // The placeholder stands in for the real path before `cite` is asked; a longer real path
      // (an absolute archive directory) can still fail the rewrite, which skips the entry below.
      const probe = rewrite(message.text, entry, wantFold, `${ARCHIVE_ID_DIR}/${entry.archiveId}.txt`, scores.headChars, scores.salvageChars);
      if (!probe) continue;
      const path = cite(entry.archiveId, message.text);
      if (!path) continue;
      const done = rewrite(message.text, entry, wantFold, path, scores.headChars, scores.salvageChars);
      if (!done) continue;
      replaced.set(message, { ...message, text: done.text });
      archived.push({ id: entry.archiveId, text: message.text, path });
      stats.charsSaved += message.text.length - done.text.length;
      if (done.folded) stats.folded++;
      else stats.stripped++;
    }
  }
  return { messages: mergeReplaced(scores.messages, replaced), replaced, archived, stats };
}

/** `messages` with each message in `replaced` swapped for its copy; matching is by object identity, so another stage may have dropped messages before. */
export function mergeReplaced(
  messages: readonly Message[],
  replaced: ReadonlyMap<Message, Message>,
): Message[] {
  return messages.map((message) => replaced.get(message) ?? message);
}

/**
 * Walks `rungs` and takes the first whose `apply` result reaches `minRatio` against the original
 * `charsBefore` of `result`. Undefined when none does. `apply` may combine stages.
 */
export function ladderRescueWith<T extends { messages: Message[] }>(
  apply: (rung: Rung) => T,
  result: CompactResult,
  minRatio: number,
  rungs: readonly Rung[] = NOTIFICATION_RUNGS,
): { rung: Rung; result: CompactResult; applied: T } | undefined {
  for (const rung of rungs) {
    const applied = apply(rung);
    const charsAfter = applied.messages.reduce((sum, message) => sum + messageChars(message), 0);
    const candidate = { ...result, messages: applied.messages, stats: { ...result.stats, charsAfter } };
    if (reductionRatio(candidate) >= minRatio) return { rung, result: candidate, applied };
  }
  return undefined;
}
