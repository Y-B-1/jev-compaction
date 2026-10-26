import { dedupeBlocks, supersededCalls } from './dedupe.js';
import { MAX_QUESTIONS_PER_REQUEST, noulAnswer } from './request.js';
import { salvageBlock } from './salvage.js';
import { splitMediaNotes } from './media.js';
import { collectToolCalls, estimateTokens, fitState, goalFromMessages, isPinned } from './state.js';
import type {
  ArchiveResult,
  CallAnswer,
  CallDecision,
  CompactOptions,
  CompactResult,
  CompactionState,
  JevAnswer,
  JevAsker,
  JevQuestions,
  Message,
  ResolvedCompactOptions,
  ToolCall,
  ToolUse,
} from './types.js';

export const DEFAULT_OPTIONS: ResolvedCompactOptions = {
  goal: '',
  /**
   * A Noul of 0.5 is Jev saying it is unsure, not "half". Distance from 0.5 is
   * the confidence signal, so a threshold of 0.5 puts every uncertain answer on
   * the delete side of an irreversible decision. Deleting context is the
   * high-stakes action here, so the gate sits on the drop: a call is removed
   * only when Jev is fairly sure it is no longer needed. See
   * https://docs.typesafe.ai/confidence.
   */
  keepThreshold: 0.15,
  /** See `CompactOptions.keepCallThreshold`. */
  keepCallThreshold: 0.05,
  preserveRecentMessages: 6,
  maxStateTokens: 25_000,
  maxRequestTokens: 60_000,
  maxConcurrentRequests: 4,
  truncateHeadChars: 150,
  resultPreviewChars: 300,
  salvageChars: 150,
  minResultChars: 500,
  maxKeptResultChars: 4000,
  dedupeRepeats: true,
  minKeepSignalCalls: 3,
  stateWording: 'delete',
  resultChecks: false,
  keepThresholdByTool: {},
};

/** Tokens the request envelope (`model`, key names) adds around state and questions. */
const REQUEST_OVERHEAD_TOKENS = 20;

/**
 * Jev's second context limit: the state plus the *single longest* question must
 * fit this, independently of the whole-request budget. See
 * https://docs.typesafe.ai/models — "64k tokens per request; 32k tokens for
 * `state` plus the longest question".
 */
const MAX_STATE_PLUS_QUESTION_TOKENS = 32_000;

function finite(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export function resolveOptions(options: CompactOptions = {}): ResolvedCompactOptions {
  const keepThreshold = finite(options.keepThreshold, DEFAULT_OPTIONS.keepThreshold);
  if (keepThreshold < 0 || keepThreshold > 1) {
    throw new RangeError('keepThreshold must be between 0 and 1');
  }
  const keepCallThreshold = finite(options.keepCallThreshold, DEFAULT_OPTIONS.keepCallThreshold);
  if (keepCallThreshold < 0 || keepCallThreshold > 1) {
    throw new RangeError('keepCallThreshold must be between 0 and 1');
  }
  return {
    goal: options.goal ?? DEFAULT_OPTIONS.goal,
    keepThreshold,
    keepCallThreshold,
    preserveRecentMessages: Math.max(
      0,
      Math.floor(
        finite(options.preserveRecentMessages, DEFAULT_OPTIONS.preserveRecentMessages),
      ),
    ),
    maxStateTokens: Math.max(1, finite(options.maxStateTokens, DEFAULT_OPTIONS.maxStateTokens)),
    maxRequestTokens: Math.max(
      1,
      finite(options.maxRequestTokens, DEFAULT_OPTIONS.maxRequestTokens),
    ),
    maxConcurrentRequests: Math.max(
      1,
      Math.floor(finite(options.maxConcurrentRequests, DEFAULT_OPTIONS.maxConcurrentRequests)),
    ),
    truncateHeadChars: Math.max(
      0,
      Math.floor(finite(options.truncateHeadChars, DEFAULT_OPTIONS.truncateHeadChars)),
    ),
    resultPreviewChars: Math.max(
      0,
      Math.floor(finite(options.resultPreviewChars, DEFAULT_OPTIONS.resultPreviewChars)),
    ),
    salvageChars: Math.max(
      0,
      Math.floor(finite(options.salvageChars, DEFAULT_OPTIONS.salvageChars)),
    ),
    minResultChars: Math.max(
      0,
      Math.floor(finite(options.minResultChars, DEFAULT_OPTIONS.minResultChars)),
    ),
    maxKeptResultChars: Math.max(
      0,
      Math.floor(finite(options.maxKeptResultChars, DEFAULT_OPTIONS.maxKeptResultChars)),
    ),
    dedupeRepeats: options.dedupeRepeats ?? DEFAULT_OPTIONS.dedupeRepeats,
    stateWording: options.stateWording === 'archive' ? 'archive' : 'delete',
    resultChecks: options.resultChecks === true,
    keepThresholdByTool: Object.fromEntries(
      Object.entries(options.keepThresholdByTool ?? {}).filter(
        ([, value]) => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1,
      ),
    ),
    minKeepSignalCalls: Math.max(
      0,
      Math.floor(finite(options.minKeepSignalCalls, DEFAULT_OPTIONS.minKeepSignalCalls)),
    ),
  };
}

/**
 * The two `noul` questions asked about one call: keep the call, keep its
 * result. Both carry criteria, because the boundary is subtle and the docs ask
 * for `true`/`false` sides whenever it is: https://docs.typesafe.ai/primitives/noul.
 */
export function questionsFor(call: ToolCall, resultChecks = false): JevQuestions {
  if (resultChecks) return { [`call_${call.id}`]: callQuestion(call), ...checkQuestions(call) };
  return {
    [`call_${call.id}`]: callQuestion(call),
    [`result_${call.id}`]: {
      type: 'noul',
      instructions: `The full output of tool call \`${call.id}\` (${call.tool}, ${call.resultChars} chars) should stay in \`history\` verbatim: the assistant still needs its contents, and re-running the tool would not do${
        call.resultPreview ? `. Output preview: ${call.resultPreview}` : ''
      }`,
      criteria: {
        true: 'The exact contents are still in use and could not be recovered by re-running the tool: an error the assistant is still diagnosing, output the user asked about, the current state of a file being edited',
        false: 'The contents are stale, already stated in the assistant text, or trivially re-obtainable: a directory listing already used, a passing test run, a file read before it was rewritten',
      },
    },
  };
}

function callQuestion(call: ToolCall): JevQuestions[string] {
  return {
    type: 'noul',
    instructions: `Tool call \`${call.id}\` (${call.tool}) should stay in \`history\`: knowing this call was made, with its input, still matters for what the assistant does next`,
    criteria: {
      true: 'The call records a change to the world, or a constraint the assistant must not violate again: an edit or write that changed a file, a command that installed, moved or deleted something, a check whose outcome the user was told about',
      false: 'The call only gathered information that has since been superseded or acted upon: a search used to locate a file that was then edited, a read of a file that has since changed, a failing check that has since been fixed',
    },
  };
}

/**
 * `resultChecks`: four concrete checks on one result instead of a single keep
 * question, each phrased so that "yes" means keep (after 0x7067/claude-jev,
 * which measured planted constraints surviving 77% -> 100% with concrete
 * checks over aggregate questions).
 */
const RESULT_CHECKS = {
  err: 'contains an exact error message, failing test output or unexpected result that the assistant is still working on and would have to reproduce to see again',
  con: 'states a requirement or restriction the work must respect: a configuration value, an API contract, a rule or limit written in a file or a reply',
  use: 'holds contents the assistant is still working with verbatim: the current text of a file it is editing, values or identifiers it will copy',
  rerun: 'would come back the same if the tool were run again: a listing, a passing check, build or install output, a file that has not changed since',
} as const;

function checkQuestions(call: ToolCall): JevQuestions {
  const subject = `The output of tool call \`${call.id}\` (${call.tool}, ${call.resultChars} chars)${
    call.resultPreview ? `, preview: ${call.resultPreview},` : ''
  }`;
  return Object.fromEntries(
    Object.entries(RESULT_CHECKS).map(([name, check]) => [
      `${name}_${call.id}`,
      { type: 'noul', instructions: `${subject} ${check}` },
    ]),
  ) as JevQuestions;
}

/**
 * The keep-result probability from the checks: the strongest of error,
 * constraint and in-use, where an error a rerun would reproduce does not count.
 */
export function checksKeep(answers: Record<string, JevAnswer>, id: string): number {
  const p = (name: string) => noulAnswer(answers, `${name}_${id}`);
  const rerun = p('rerun');
  return Math.max(p('con'), p('use'), rerun >= 0.5 ? 0 : p('err'));
}

/**
 * Splits the candidate calls into batches whose questions, together with the
 * (always complete) state, fit one request.
 */
export function batchCalls(
  calls: readonly ToolCall[],
  stateTokens: number,
  options: Pick<ResolvedCompactOptions, 'maxRequestTokens'> & Partial<Pick<ResolvedCompactOptions, 'resultChecks'>>,
): ToolCall[][] {
  const budget = options.maxRequestTokens - stateTokens - REQUEST_OVERHEAD_TOKENS;
  const batches: ToolCall[][] = [];
  let current: ToolCall[] = [];
  let currentTokens = 0;
  let currentQuestions = 0;
  for (const call of calls) {
    const questions = questionsFor(call, options.resultChecks);
    const tokens = estimateTokens(JSON.stringify(questions));
    const count = Object.keys(questions).length;
    if (stateTokens + tokens + REQUEST_OVERHEAD_TOKENS > MAX_STATE_PLUS_QUESTION_TOKENS) {
      throw new Error(
        `state plus one question exceeds Jev's ${MAX_STATE_PLUS_QUESTION_TOKENS}-token limit ` +
          `(~${stateTokens} state + ~${tokens} question); lower maxStateTokens`,
      );
    }
    if (current.length > 0 && (currentTokens + tokens > budget || currentQuestions + count > MAX_QUESTIONS_PER_REQUEST)) {
      batches.push(current);
      current = [];
      currentTokens = 0;
      currentQuestions = 0;
    }
    if (current.length === 0 && tokens > budget) {
      throw new Error(
        `state leaves no room for questions (~${stateTokens} of ${options.maxRequestTokens} tokens)`,
      );
    }
    current.push(call);
    currentTokens += tokens;
    currentQuestions += count;
  }
  if (current.length > 0) batches.push(current);
  return batches;
}

export function decideCall(
  call: Pick<ToolCall, 'id' | 'tool' | 'pinned'>,
  answer: CallAnswer,
  options: Pick<ResolvedCompactOptions, 'keepThreshold'> &
    Partial<Pick<ResolvedCompactOptions, 'keepCallThreshold'>>,
): CallDecision {
  if (
    !Number.isFinite(options.keepThreshold) ||
    options.keepThreshold < 0 ||
    options.keepThreshold > 1
  ) {
    throw new RangeError('keepThreshold must be between 0 and 1');
  }
  // `keepCallThreshold` is newer than this exported function; a caller built
  // against the old signature must not silently start comparing against
  // `undefined` (which would drop every call), so it falls back to the default.
  const keepCallThreshold = finite(options.keepCallThreshold, DEFAULT_OPTIONS.keepCallThreshold);
  if (keepCallThreshold < 0 || keepCallThreshold > 1) {
    throw new RangeError('keepCallThreshold must be between 0 and 1');
  }
  if (
    ![answer.keepCall, answer.keepResult].every(
      (value) => Number.isFinite(value) && value >= 0 && value <= 1,
    )
  ) {
    throw new Error('Invalid keep probabilities');
  }
  const base = { id: call.id, tool: call.tool, ...answer };
  if (call.pinned) return { ...base, action: 'keep', reason: 'pinned' };
  if (answer.keepResult >= options.keepThreshold) {
    return { ...base, action: 'keep', reason: 'kept' };
  }
  if (answer.keepCall >= keepCallThreshold) {
    return { ...base, action: 'drop_result', reason: 'result_dropped' };
  }
  return { ...base, action: 'drop_call', reason: 'call_dropped' };
}

async function askBatch(
  asker: JevAsker,
  state: CompactionState,
  batch: readonly ToolCall[],
  resultChecks = false,
): Promise<Map<string, CallAnswer>> {
  const questions: JevQuestions = Object.assign({}, ...batch.map((call) => questionsFor(call, resultChecks)));
  const { answers } = await asker.ask(state, questions);
  return new Map(
    batch.map((call) => [
      call.id,
      {
        keepCall: noulAnswer(answers, `call_${call.id}`),
        keepResult: resultChecks ? checksKeep(answers, call.id) : noulAnswer(answers, `result_${call.id}`),
      },
    ]),
  );
}

/**
 * Runs the batches over a bounded pool of concurrent requests. A batch that
 * fails (after whatever retries the asker itself makes) is salvaged rather
 * than sinking the whole compaction: its calls are simply left unanswered, so
 * `compact` falls back to its safe default (keep) for them, and the other
 * batches' answers still apply. Only when every batch fails is the last
 * failure re-thrown, so a caller that can fall back to a built-in summary
 * still learns why nothing was pruned.
 */
async function askBatches(
  asker: JevAsker,
  jobs: readonly { state: CompactionState; batch: readonly ToolCall[] }[],
  concurrency: number,
  resultChecks = false,
): Promise<{ answered: Map<string, CallAnswer>[]; failed: number }> {
  const answered: Map<string, CallAnswer>[] = new Array(jobs.length);
  let cursor = 0;
  let failed = 0;
  let lastFailure: unknown;
  const worker = async (): Promise<void> => {
    while (cursor < jobs.length) {
      const index = cursor++;
      try {
        const job = jobs[index]!;
        answered[index] = await askBatch(asker, job.state, job.batch, resultChecks);
      } catch (error) {
        failed += 1;
        lastFailure = error;
        answered[index] = new Map();
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, jobs.length) }, worker));
  if (failed > 0 && failed === jobs.length) throw lastFailure;
  return { answered, failed };
}

type StateGroup = { state: ReturnType<typeof fitState>; calls: ToolCall[] };

function tooLarge(error: unknown): boolean {
  return error instanceof Error && error.message.startsWith('history too large for Jev');
}

/**
 * One state for all candidates when the history fits `maxStateTokens`. When it
 * does not (long sessions), the candidates are split into contiguous windows,
 * halved until each window's state fits: a window's state keeps the goal, the
 * first message, the pinned newest messages and its own messages in full, and
 * blanks the rest (text and calls). Whether an old call is still needed depends
 * mostly on the goal and the recent turns, which every window carries. A call
 * whose window cannot fit even alone is returned in `unasked`; it gets no
 * answer and so stays, like any call Jev was not asked about.
 * (Ported from upstream tamaratran/fast-jev-compaction#119.)
 */
export function stateGroups(
  messages: readonly Message[],
  calls: readonly ToolCall[],
  candidates: readonly ToolCall[],
  options: ResolvedCompactOptions,
): { groups: StateGroup[]; unasked: ToolCall[]; stage: string } {
  // A rough per-message cost (text, then each input as fitState first cuts it):
  // a history many times the budget goes straight to windows, sized up front,
  // since every failed fitState walks its whole shrink ladder over the history.
  const costs = messages.map(
    (message) =>
      estimateTokens(message.text.length > 4000 ? message.text.slice(0, 4000) : message.text) +
      message.toolUses.reduce((sum, tool) => sum + Math.min(300, estimateTokens(safeJson(tool.input).slice(0, 1000))), 0) +
      10,
  );
  const prefix = [0];
  for (const cost of costs) prefix.push(prefix[prefix.length - 1]! + cost);
  const span = (lo: number, hi: number) => prefix[hi + 1]! - prefix[lo]!;
  if (prefix[prefix.length - 1]! <= options.maxStateTokens * WHOLE_STATE_FACTOR) {
    try {
      const state = fitState(messages, calls, options);
      return { groups: [{ state, calls: [...candidates] }], unasked: [], stage: state.stage };
    } catch (error) {
      if (!tooLarge(error)) throw error;
    }
  }
  const windowOptions = { ...options, goal: options.goal || goalFromMessages(messages) };
  const groups: StateGroup[] = [];
  const unasked: ToolCall[] = [];
  const fit = (group: ToolCall[]): void => {
    const lo = Math.min(...group.map((call) => call.callIndex));
    const hi = Math.max(...group.map((call) => call.resultIndex));
    const view = messages.map((message, i) =>
      (i >= lo && i <= hi) || isPinned(i, messages.length, options.preserveRecentMessages)
        ? message
        : // Outside the window a call left in place would read as pending.
          { role: message.role, text: '', toolUses: [] },
    );
    // Every call whose call and result both sit in the shown rows keeps its
    // pairing (settled calls and other windows' candidates included), so Jev
    // does not read them as calls that never returned.
    const shown = (i: number) => (i >= lo && i <= hi) || isPinned(i, messages.length, options.preserveRecentMessages);
    try {
      const state = fitState(view, calls.filter((call) => shown(call.callIndex) && shown(call.resultIndex)), windowOptions);
      groups.push({ state, calls: group });
    } catch (error) {
      if (!tooLarge(error)) throw error;
      if (group.length === 1) {
        unasked.push(group[0]!);
        return;
      }
      const half = Math.ceil(group.length / 2);
      fit(group.slice(0, half));
      fit(group.slice(half));
    }
  };
  // Contiguous windows whose rows cost at most WINDOW_SHARE of the budget, so
  // most fit at the first try.
  const sorted = [...candidates].sort((a, b) => a.callIndex - b.callIndex);
  let group: ToolCall[] = [];
  let lo = 0;
  for (const call of sorted) {
    if (group.length === 0) lo = call.callIndex;
    const hi = Math.max(call.resultIndex, ...group.map((c) => c.resultIndex));
    if (group.length > 0 && span(lo, hi) > options.maxStateTokens * WINDOW_SHARE) {
      fit(group);
      group = [];
      lo = call.callIndex;
    }
    group.push(call);
  }
  if (group.length > 0) fit(group);
  const stage = `windows:${groups.length}${unasked.length > 0 ? ` unasked:${unasked.length}` : ''}`;
  return { groups, unasked, stage };
}

/** A history estimated above this many times `maxStateTokens` skips the single-state attempt. */
const WHOLE_STATE_FACTOR = 4;
/** Share of `maxStateTokens` a window's own rows may cost, leaving room for the goal and pinned rows. */
const WINDOW_SHARE = 0.5;

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value ?? {});
  } catch {
    return '{}';
  }
}

/** Whether ending `text` at `index` would separate the two halves of a surrogate pair. */
export function splitsSurrogatePair(text: string, index: number): boolean {
  const before = text.charCodeAt(index - 1);
  const after = text.charCodeAt(index);
  return before >= 0xd800 && before <= 0xdbff && after >= 0xdc00 && after <= 0xdfff;
}

/** The first `chars` characters of `text`, one fewer when the cut would split a surrogate pair. */
export function headOf(text: string, chars: number): string {
  return text.slice(0, splitsSurrogatePair(text, chars) ? chars - 1 : chars);
}

/**
 * How much longer than `maxKeptResultChars` a kept result must be before it is
 * capped: the note, the salvaged lines and the archive path take room too, and
 * a cap must never make a result longer.
 */
const CAP_MARGIN = 600;

/** Starts the note of a result Jev kept whole but that was cut to `maxKeptResultChars`. */
export const CAPPED_MARKER = '[jev-compaction kept the head and tail of ';

/**
 * A result Jev kept whole, cut to about `capChars`: its head, then its tail
 * (where test and build summaries and the last error sit; half of the cap for
 * an error result, a quarter otherwise), with the identifiers and error lines
 * of the omitted middle salvaged between them.
 */
function cappedResultText(
  text: string,
  isError: boolean,
  capChars: number,
  archivedPath?: string,
  salvageChars = 0,
): string {
  if (text.length <= capChars + CAP_MARGIN) return text;
  const tailChars = Math.floor(capChars * (isError ? 0.5 : 0.25));
  const head = headOf(text, capChars - tailChars);
  let tailStart = text.length - tailChars;
  if (splitsSurrogatePair(text, tailStart)) tailStart += 1;
  const middle = text.slice(head.length, tailStart);
  const where = archivedPath ? `; full output: ${archivedPath}` : '; re-run the tool if needed';
  const note = `${CAPPED_MARKER}${text.length} chars, ${middle.length} omitted${isError ? ' (error)' : ''}${where}]`;
  const salvaged = salvageBlock(middle, salvageChars);
  return `${head}\n${note}${salvaged ? `\n${salvaged}` : ''}\n${text.slice(tailStart)}`;
}

/**
 * Whether `text` is a result an earlier compaction cut: its note starts a line
 * and carries a count, so a tool output that only quotes the marker (this
 * plugin's own source, a Read with line numbers) is not mistaken for one.
 */
export function wasTruncated(text: string): boolean {
  return /(?:^|\n)\[(?:fast-)?jev-compaction truncated \d+ chars/.test(text);
}

/** Whether `text` is a result an earlier compaction capped (see `wasTruncated`). */
export function wasCapped(text: string): boolean {
  return /\n\[(?:fast-)?jev-compaction kept the head and tail of \d+ chars/.test(text);
}

/** The archive path an earlier cap cited in `text`, if any. */
function cappedPath(text: string): string | undefined {
  const at = text.search(/\[(?:fast-)?jev-compaction kept the head and tail of /);
  if (at === -1) return undefined;
  return /; full output: ([^\]\n]+)\]/.exec(text.slice(at))?.[1];
}

/** Characters from the end of a failing result kept when it is truncated. */
const ERROR_TAIL_CHARS = 800;

function truncatedResultText(
  text: string,
  isError: boolean,
  headChars: number,
  archivedPath?: string,
  salvageChars = 0,
): string {
  if (text.length <= headChars + 120) return text;
  // End the head on a code point boundary: cutting a surrogate pair in half
  // would leave a lone surrogate in the rebuilt transcript, which the
  // Messages API rejects on every subsequent turn.
  const cut = splitsSurrogatePair(text, headChars) ? headChars - 1 : headChars;
  const head = cut > 0 ? `${text.slice(0, cut)}\n` : '';
  const where = archivedPath ? `; full output: ${archivedPath}` : '; re-run the tool if needed';
  const note = `[jev-compaction truncated ${text.length - cut} chars${isError ? ' (error)' : ''}${where}]`;
  // Identifiers and error lines recovered from the dropped tail: the head is
  // preamble for a listing or a build log, and the useful parts live further
  // down where they would otherwise be lost until the tool is re-run.
  // A failing result keeps its end too: the deepest frame of a stack, the
  // final error code or the failing assertion sit there, not in the head.
  let tail = '';
  if (isError && text.length - cut > ERROR_TAIL_CHARS * 2) {
    let start = text.length - ERROR_TAIL_CHARS;
    if (splitsSurrogatePair(text, start)) start += 1;
    tail = `\n${text.slice(start)}`;
  }
  const salvaged = salvageBlock(text.slice(cut, tail ? text.length - tail.length + 1 : undefined), salvageChars);
  return `${head}${note}${salvaged ? `\n${salvaged}` : ''}${tail}`;
}

/** Tools whose output a re-run reproduces; other tools observed something that may not happen again. */
const REPRODUCIBLE_TOOLS = new Set(['Read', 'Glob', 'Grep', 'LS', 'NotebookRead']);

/**
 * One line standing in for a dropped call, naming where its output went. For a
 * tool whose output a re-run may not reproduce (a command, a test run, a
 * request), the error lines and identifiers of that output follow it.
 */
function droppedCallNote(tool: ToolUse, archivedPath: string | undefined, salvageChars = 0): string {
  let input = '';
  try {
    input = JSON.stringify(tool.input);
  } catch {
    input = '{}';
  }
  if (input.length > 160) input = `${headOf(input, 159)}…`;
  const note = `[jev-compaction removed the ${tool.tool} call ${input}${
    archivedPath ? `; full output: ${archivedPath}` : ''
  }]`;
  const salvaged = REPRODUCIBLE_TOOLS.has(tool.tool) ? '' : salvageBlock(tool.text ?? '', salvageChars);
  return salvaged ? `${note}\n${salvaged}` : note;
}

function isEmptyMessage(message: {
  text: string;
  toolUses: readonly unknown[];
  toolResults?: readonly unknown[];
}): boolean {
  return (
    message.text.trim().length === 0 &&
    message.toolUses.length === 0 &&
    (message.toolResults?.length ?? 0) === 0
  );
}

/**
 * Rebuilds the conversation from the decisions. A dropped call disappears
 * together with its result; a dropped result keeps a bounded head and note.
 * Messages that lose all their content are removed; untouched messages are
 * returned as the same objects they came in as.
 */
export function applyDecisions(
  messages: readonly Message[],
  decisions: readonly CallDecision[],
  calls: readonly ToolCall[],
  headChars: number,
  archive?: ArchiveResult,
  salvageChars = 0,
  capChars = 0,
  pinnedFrom = Infinity,
): Message[] {
  const byId = new Map(calls.map((call) => [call.id, call]));
  // `cap`: a result Jev kept whole that is longer than `capChars`, cut to its
  // first `capChars` like a dropped result.
  const actions = new Map<string, CallDecision['action'] | 'cap'>();
  for (const decision of decisions) {
    const call = byId.get(decision.id);
    if (!call) continue;
    if (decision.action !== 'keep') actions.set(call.tool_use_id, decision.action);
    else if (decision.reason === 'kept' && capChars > 0 && call.resultChars > capChars + CAP_MARGIN) {
      actions.set(call.tool_use_id, 'cap');
    }
  }
  // An image's or document's placeholder (see `placeMedia`) stays whole; only the rest is cut.
  const cutResult = (id: string, text: string, isError: boolean): string => {
    const { rest, notes } = splitMediaNotes(text);
    if (notes.length === 0) return cutText(id, text, isError);
    return [cutText(id, rest, isError), ...notes].filter(Boolean).join('\n');
  };
  const cutText = (id: string, text: string, isError: boolean): string => {
    const action = actions.get(id);
    if (action !== 'drop_result' && action !== 'cap') return text;
    // A result an earlier compaction capped is not capped again, and when Jev
    // now drops it, the archive of its full output is cited, never overwritten
    // with the capped copy.
    if (wasCapped(text)) {
      return action === 'cap' ? text : truncatedResultText(text, isError, headChars, cappedPath(text), salvageChars);
    }
    // Archive only what is actually cut: a short duplicate stays as it is.
    if (text.length <= (action === 'cap' ? capChars + CAP_MARGIN : headChars + 120)) return text;
    return action === 'cap'
      ? cappedResultText(text, isError, capChars, archive?.(id, text), salvageChars)
      : truncatedResultText(text, isError, headChars, archive?.(id, text), salvageChars);
  };
  // A dropped call's result goes with it: its media placeholders move to the call's note.
  const mediaNotes = new Map<string, string[]>();
  for (const message of messages) {
    for (const result of message.toolResults ?? []) {
      const { notes } = splitMediaNotes(result.text);
      if (notes.length > 0) mediaNotes.set(result.tool_use_id, notes);
    }
  }
  const kept: Message[] = [];
  for (const [index, message] of messages.entries()) {
    const touched =
      message.toolUses.some((tool) => actions.has(tool.tool_use_id)) ||
      (message.toolResults ?? []).some((result) => actions.has(result.tool_use_id));
    if (!touched) {
      // An assistant message that came in already fully empty (no text, no
      // tool call, no tool result) carries nothing for Jev's history and
      // nothing to answer with later; a user or tool-carrier message is
      // never dropped this way, since even an empty user turn is a real
      // step in the conversation. Within the pinned newest messages it stays:
      // there it can be the thinking block of the response still running,
      // which the API requires back.
      if (message.role === 'assistant' && isEmptyMessage(message) && index < pinnedFrom) continue;
      kept.push(message);
      continue;
    }
    // A dropped call leaves one line naming it and where its output was
    // archived, so a detail inside it can still be recovered.
    const droppedCallNotes = message.toolUses
      .filter((tool) => actions.get(tool.tool_use_id) === 'drop_call')
      .map((tool) =>
        [
          droppedCallNote(
            tool,
            // A result an earlier compaction capped already has its full output
            // archived: cite that file, never overwrite it with the capped copy.
            wasCapped(tool.text ?? '') ? cappedPath(tool.text ?? '') : archive?.(tool.tool_use_id, tool.text ?? ''),
            salvageChars,
          ),
          ...(mediaNotes.get(tool.tool_use_id) ?? []),
        ].join('\n'),
      );
    const toolUses = message.toolUses
      .filter((tool) => actions.get(tool.tool_use_id) !== 'drop_call')
      .map((tool) => {
        const text = cutResult(tool.tool_use_id, tool.text ?? '', tool.isError ?? false);
        if ((tool.text ?? '') === text) return tool;
        const copy: ToolUse = {
          tool_use_id: tool.tool_use_id,
          tool: tool.tool,
          input: tool.input,
          text,
        };
        if (tool.isError) copy.isError = true;
        return copy;
      });
    const toolResults = (message.toolResults ?? [])
      .filter((result) => actions.get(result.tool_use_id) !== 'drop_call')
      .map((result) => {
        const text = cutResult(result.tool_use_id, result.text, result.isError ?? false);
        return text === result.text
          ? result
          : {
              tool_use_id: result.tool_use_id,
              text,
              isError: result.isError,
            };
      });
    if (
      !message.toolUses.some(
        (tool) => actions.get(tool.tool_use_id) === 'drop_call',
      ) &&
      !(message.toolResults ?? []).some(
        (result) => actions.get(result.tool_use_id) === 'drop_call',
      ) &&
      toolUses.every((tool, index) => tool === message.toolUses[index]) &&
      toolResults.every(
        (result, index) => result === message.toolResults?.[index],
      )
    ) {
      kept.push(message);
      continue;
    }
    const text =
      droppedCallNotes.length > 0
        ? [message.text, ...droppedCallNotes].filter(Boolean).join('\n')
        : message.text;
    if (isEmptyMessage({ text, toolUses, toolResults })) {
      continue;
    }
    const rebuilt: Message = { role: message.role, text, toolUses };
    if (toolResults.length > 0) rebuilt.toolResults = toolResults;
    kept.push(rebuilt);
  }
  return kept;
}

/** Characters of text, tool input and tool output a message holds. */
export function messageChars(message: Message): number {
  let total = message.text.length;
  for (const tool of message.toolUses) {
    try {
      total += JSON.stringify(tool.input).length;
    } catch {
      total += 20;
    }
  }
  for (const result of message.toolResults ?? []) total += result.text.length;
  return total;
}

export function reductionRatio(result: Pick<CompactResult, 'stats'>): number {
  const { charsBefore, charsAfter } = result.stats;
  return charsBefore === 0 ? 0 : (charsBefore - charsAfter) / charsBefore;
}

function count(decisions: readonly CallDecision[], reason: CallDecision['reason']): number {
  return decisions.filter((decision) => decision.reason === reason).length;
}

/**
 * Compacts a transcript by asking Jev, for every tool call outside the pinned
 * first and newest messages, whether the call and whether its result must
 * stay. The whole history (results omitted, fitted into `maxStateTokens`) is
 * sent as state with every batch of questions. Throws when Jev fails or the
 * history cannot be fitted; the caller decides whether to fall back.
 */
export async function compact(
  messages: readonly Message[],
  asker: JevAsker,
  options: CompactOptions = {},
): Promise<CompactResult> {
  const started = Date.now();
  const resolved = resolveOptions(options);
  const charsBefore = messages.reduce((sum, message) => sum + messageChars(message), 0);
  const deduped = resolved.dedupeRepeats
    ? dedupeBlocks(messages, resolved.preserveRecentMessages)
    : { messages: messages.slice(), removed: 0 };
  const source = deduped.messages;
  const calls = collectToolCalls(
    source,
    resolved.preserveRecentMessages,
    resolved.resultPreviewChars,
  );
  // Settled without asking Jev, most protective first: a result an earlier
  // compaction already cut (cutting it again would overwrite its archived
  // full output with the stub), a result too small to be worth a question,
  // then an earlier copy of a call a later identical call supersedes.
  const resultTexts = new Map<string, string>();
  for (const message of source) {
    for (const result of message.toolResults ?? []) resultTexts.set(result.tool_use_id, result.text);
  }
  const superseded = resolved.dedupeRepeats ? supersededCalls(calls) : new Set<string>();
  const settled = new Map<string, CallDecision>();
  for (const call of calls) {
    if (call.pinned) continue;
    const base = { id: call.id, tool: call.tool, keepCall: 1, keepResult: 1 };
    if (wasTruncated(resultTexts.get(call.tool_use_id) ?? '')) {
      settled.set(call.id, { ...base, action: 'keep', reason: 'compacted' });
    } else if (call.resultChars < resolved.minResultChars) {
      settled.set(call.id, { ...base, action: 'keep', reason: 'small' });
    } else if (superseded.has(call.id)) {
      settled.set(call.id, { ...base, keepResult: 0, action: 'drop_result', reason: 'duplicate' });
    }
  }
  const candidates = calls.filter((call) => !call.pinned && !settled.has(call.id));

  let fitted: { tokens: number; stage: string } = { tokens: 0, stage: '' };
  let requests = 0;
  let failedRequests = 0;
  const answers = new Map<string, CallAnswer>();
  if (candidates.length > 0) {
    // Reserve room for the largest question before fitting the state, so a
    // tight maxRequestTokens shrinks the state instead of batching down to a
    // state that leaves no room for even one question.
    const largestQuestion = candidates.reduce(
      (largest, call) => Math.max(largest, estimateTokens(JSON.stringify(questionsFor(call, resolved.resultChecks)))),
      0,
    );
    const stateBudget = resolved.maxRequestTokens - REQUEST_OVERHEAD_TOKENS - largestQuestion;
    if (stateBudget < 1) {
      throw new Error('request budget leaves no room for state and questions');
    }
    const { groups, stage } = stateGroups(source, calls, candidates, {
      ...resolved,
      maxStateTokens: Math.min(resolved.maxStateTokens, stateBudget),
      // Saying dropped output is archived is only true when it is.
      stateWording: options.archive ? resolved.stateWording : 'delete',
    });
    fitted = { tokens: Math.max(0, ...groups.map((group) => group.state.tokens)), stage };
    const jobs = groups.flatMap((group) =>
      batchCalls(group.calls, group.state.tokens, resolved).map((batch) => ({ state: group.state.state, batch })),
    );
    requests = jobs.length;
    const { answered, failed } =
      jobs.length > 0 ? await askBatches(asker, jobs, resolved.maxConcurrentRequests, resolved.resultChecks) : { answered: [], failed: 0 };
    failedRequests = failed;
    for (const map of answered) for (const [id, answer] of map) answers.set(id, answer);
  }

  const decisions = calls.map(
    (call) =>
      settled.get(call.id) ??
      decideCall(call, answers.get(call.id) ?? { keepCall: 1, keepResult: 1 }, {
        ...resolved,
        keepThreshold: resolved.keepThresholdByTool[call.tool] ?? resolved.keepThreshold,
      }),
  );
  // Answers that keep nothing at all, not even a call, are what a Jev
  // answering 0 to every question produces: they say nothing about the
  // transcript, yet they pass any reduction floor with the largest cut there
  // is. Treat them as a failure so the caller can fall back.
  const answered = decisions.filter((decision) => answers.has(decision.id));
  if (
    resolved.minKeepSignalCalls > 0 &&
    answered.length >= resolved.minKeepSignalCalls &&
    answered.every((decision) => decision.action === 'drop_call')
  ) {
    throw new Error(
      `no keep signal: Jev dropped all ${answered.length} calls it answered`,
    );
  }
  const kept = applyDecisions(
    source,
    decisions,
    calls,
    resolved.truncateHeadChars,
    options.archive,
    resolved.salvageChars,
    resolved.maxKeptResultChars,
    source.length - resolved.preserveRecentMessages,
  );
  const capped = new Set(
    calls
      .filter(
        (call) =>
          resolved.maxKeptResultChars > 0 &&
          call.resultChars > resolved.maxKeptResultChars + CAP_MARGIN &&
          !wasCapped(resultTexts.get(call.tool_use_id) ?? ''),
      )
      .map((call) => call.id),
  );
  return {
    messages: kept,
    decisions,
    toolUseIds: Object.fromEntries(calls.map((call) => [call.id, call.tool_use_id])),
    stats: {
      messagesBefore: messages.length,
      messagesAfter: kept.length,
      charsBefore,
      charsAfter: kept.reduce((sum, message) => sum + messageChars(message), 0),
      calls: calls.length,
      kept: count(decisions, 'kept'),
      resultsDropped: count(decisions, 'result_dropped'),
      callsDropped: count(decisions, 'call_dropped'),
      pinned: count(decisions, 'pinned'),
      small: count(decisions, 'small'),
      compacted: count(decisions, 'compacted'),
      duplicates: count(decisions, 'duplicate'),
      resultsCapped: decisions.filter((d) => d.reason === 'kept' && capped.has(d.id)).length,
      inputsElided: 0,
      blocksDeduped: deduped.removed,
      stateTokens: fitted.tokens,
      stateStage: fitted.stage,
      requests,
      failedRequests,
      ms: Date.now() - started,
    },
  };
}
