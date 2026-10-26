import type {
  Hook,
  On,
  PluginOptions,
  Register,
  SessionMessage,
  ToolResultSummary,
  ToolUseSummary,
} from 'claude-code';

import { compact, headOf, messageChars, reductionRatio, resolveOptions } from '../src/compact.js';
import { trimOutput } from '../src/output.js';
import {
  buildJevRequest,
  DEFAULT_MODEL,
  isRetryableStatus,
  parseJevResponse,
  retryDelayMs,
} from '../src/request.js';
import { type ExchangeStats, applyExchanges, exchangeLadder, scoreExchanges } from '../src/exchanges.js';
import { type StepOutcome, type StepStats, mildSteps, scoreSteps, stepLadder } from '../src/steps.js';
import type { NotificationStats, Rung } from '../src/notifications.js';
import {
  PROSE_KEEP_LADDER,
  callActions,
  elideInputs,
  ladderRescue,
  scoreProse,
  type ProseStats,
} from '../src/prose.js';
import { redactCompactionState, redactText } from '../src/redact.js';
import { estimateTokens, goalFromMessages, hasMarker } from '../src/state.js';
import { type MediaArchive, findMedia, mediaNote, placeMedia, splitMediaNotes } from '../src/media.js';
import { MID_TURN_LEAD, type MidTurn, prunePresent, recordMidTurn, restoreMidTurn } from '../src/midturn.js';
import type { InjectStats } from '../src/inject.js';
import { failedProseStats, trimRescue } from '../src/trim.js';
import type {
  CompactOptions,
  CompactResult,
  JevAsker,
  Message,
  ToolResult,
  ToolUse,
} from '../src/types.js';

/**
 * Which compactions Jev could not do may go on to Claude Code's built-in
 * summary: `auto` (default) only the engine's own, `always` every one,
 * `never` none.
 */
export type BuiltinFallback = 'auto' | 'always' | 'never';

const HOOK_DEFAULTS = {
  archiveResults: true,
  minReductionRatio: 0.25,
  model: DEFAULT_MODEL,
  builtinFallback: 'auto' as BuiltinFallback,
  /** On by default: when tool compaction falls short of `minReductionRatio`, also drop filler from old assistant prose before falling back. */
  proseTrim: true,
  /** On by default: when tool compaction falls short, also strip and fold the reports of finished background workers (`<task-notification>` messages), archiving each original. Needs `archiveResults`. */
  notificationTrim: true,
  /** Opt-in: trims a Bash command's stdout through Jev at tool.call time, before compaction ever sees it. Off by default — a second Jev call site of its own, see `jevPruneBashOutput`. */
  jevPruneBashOutput: false,
  /** stdout shorter than this is never sent to Jev for trimming. */
  jevPruneMinChars: 4_000,
  /** On by default: hand messages back in the shape that survives `--resume`; see `resumeSafeMessages`. */
  stripHandles: true,
  /** On by default: cut the code bodies of old tool inputs (Write/Edit contents, Bash heredocs) on every compaction, by rule; see `elideInputs`. */
  inputTrim: true,
  /**
   * On by default: when trimming alone frees too little, have the built-in
   * summarizer summarize only the older part of the history and keep the
   * recent turns trimmed, instead of summarizing everything; see `hybridSplit`.
   * Runs only where `builtinFallback` allows the built-in summary.
   */
  hybridSummary: true,
  /**
   * Opt-in: the hybrid summary is written from the original older messages
   * rather than their trimmed form (which carries truncation notes and archive
   * paths instead of the text). Measure with `eval/variants.ts` before turning on.
   */
  hybridOriginals: false,
  /** On by default: append one line per compaction to the archive's `log.jsonl`, and note when an archive file is read back; see `eval/calibrate.ts`. */
  compactionLog: true,
  /** On by default: when trimming falls short, fold old skill bodies, browser element selections, pastes and long user rows to a head and a marker, archiving each original; typed text around them and the last three prompts stay. Needs `archiveResults`; see `src/inject.ts`. */
  injectTrim: true,
  /** On by default: when tool compaction already reaches the minimum, also archive old replies and work steps before the last prompt that Jev scores below the mildest cut-off; and archive each prose cut's original reply. Needs `archiveResults`; see `src/steps.ts`. */
  replyTrim: true,
  /** On by default: as the last trim rung before any built-in summary, archive the work of old exchanges Jev scores as no longer needed, keeping each prompt verbatim; see `src/exchanges.ts`. Needs `archiveResults`. */
  exchangeDrop: true,
  /** On by default: the last stage before any built-in summary. Archives single steps of work (a tool call with its result, a reply, a host reminder) Jev scores lowest, first before the last prompt, then the current task's oldest, until the minimum is met. Prompts, row 0 and the recent rows stay. Needs `archiveResults`; see `src/steps.ts`. */
  stepTrim: true,
};


export type HookFetchInit = {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
};

export type HookFetchResponse = {
  status: number;
  ok: boolean;
  text: string;
  /** Lower-cased response headers, when the engine gives them (for `retry-after`). */
  headers?: Record<string, string>;
};

/** The shape of `$.http.fetch`, so the hook can be driven without an engine. */
export type HookFetch = (url: string, init?: HookFetchInit) => Promise<HookFetchResponse>;

export type HookConfig = CompactOptions & {
  apiKey?: string;
  /** Set per compaction from the API form of the conversation; see `NotificationOptions.imagesInContext`. */
  imagesInContext?: boolean;
  archiveResults: boolean;
  minReductionRatio: number;
  model: string;
  builtinFallback: BuiltinFallback;
  proseTrim: boolean;
  /** Absent unless the user set it; read as `HOOK_DEFAULTS.notificationTrim` (true) then. */
  notificationTrim?: boolean;
  jevPruneBashOutput: boolean;
  jevPruneMinChars: number;
  /** See `resumeSafeMessages`. */
  stripHandles: boolean;
  /** See `HOOK_DEFAULTS.inputTrim`. */
  inputTrim: boolean;
  /** See `HOOK_DEFAULTS.hybridSummary`. */
  hybridSummary: boolean;
  /** See `HOOK_DEFAULTS.hybridOriginals`. */
  hybridOriginals: boolean;
  /** See `HOOK_DEFAULTS.compactionLog`. */
  compactionLog: boolean;
  /** See `HOOK_DEFAULTS.injectTrim`. */
  injectTrim: boolean;
  /** See `HOOK_DEFAULTS.replyTrim`. */
  replyTrim: boolean;
  /** See `HOOK_DEFAULTS.exchangeDrop`. */
  exchangeDrop: boolean;
  /** Absent unless the user set it; read as `HOOK_DEFAULTS.stepTrim` (true) then. */
  stepTrim?: boolean;
};

function optionNumber(options: PluginOptions, key: string, fallback: number): number {
  const value = options[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function optionString(options: PluginOptions, key: string): string | undefined {
  const value = options[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function optionFallback(options: PluginOptions): BuiltinFallback {
  const value = options['builtinFallback'];
  return value === 'auto' || value === 'always' || value === 'never'
    ? value
    : HOOK_DEFAULTS.builtinFallback;
}

/**
 * Whether a compaction Jev could not do may be handed to the built-in summary.
 * Only the engine's own `auto` compaction (its threshold, or a prompt too
 * long) must shrink the conversation; `/compact` (`manual`), a plugin's
 * request and a `precompute` can leave it as it is, which is free, where the
 * summary is a long model call that rewrites the verbatim history.
 */
export function mayUseBuiltin(trigger: string | undefined, mode: BuiltinFallback): boolean {
  if (mode === 'always') return true;
  if (mode === 'never') return false;
  return trigger === 'auto';
}

/** Reads the plugin's `userConfig` values; anything missing takes the defaults. */
export function resolveHookConfig(options: PluginOptions): HookConfig {
  const numbers: Partial<Omit<CompactOptions, 'goal'>> = {};
  for (const key of [
    'keepThreshold',
    'keepCallThreshold',
    'preserveRecentMessages',
    'maxStateTokens',
    'maxRequestTokens',
    'truncateHeadChars',
    'resultPreviewChars',
    'salvageChars',
    'minResultChars',
    'maxKeptResultChars',
    'minKeepSignalCalls',
  ] as const) {
    const value = options[key];
    if (typeof value === 'number' && Number.isFinite(value)) numbers[key] = value;
  }
  const config: HookConfig = {
    ...numbers,
    archiveResults:
      typeof options['archiveResults'] === 'boolean'
        ? options['archiveResults']
        : HOOK_DEFAULTS.archiveResults,
    minReductionRatio: optionNumber(
      options,
      'minReductionRatio',
      HOOK_DEFAULTS.minReductionRatio,
    ),
    model: optionString(options, 'model') ?? HOOK_DEFAULTS.model,
    builtinFallback: optionFallback(options),
    proseTrim:
      typeof options['proseTrim'] === 'boolean' ? options['proseTrim'] : HOOK_DEFAULTS.proseTrim,
    ...(typeof options['notificationTrim'] === 'boolean'
      ? { notificationTrim: options['notificationTrim'] }
      : {}),
    jevPruneBashOutput:
      typeof options['jevPruneBashOutput'] === 'boolean'
        ? options['jevPruneBashOutput']
        : HOOK_DEFAULTS.jevPruneBashOutput,
    jevPruneMinChars: Math.max(
      0,
      Math.floor(optionNumber(options, 'jevPruneMinChars', HOOK_DEFAULTS.jevPruneMinChars)),
    ),
    stripHandles:
      typeof options['stripHandles'] === 'boolean'
        ? options['stripHandles']
        : HOOK_DEFAULTS.stripHandles,
    inputTrim:
      typeof options['inputTrim'] === 'boolean' ? options['inputTrim'] : HOOK_DEFAULTS.inputTrim,
    hybridSummary:
      typeof options['hybridSummary'] === 'boolean' ? options['hybridSummary'] : HOOK_DEFAULTS.hybridSummary,
    hybridOriginals:
      typeof options['hybridOriginals'] === 'boolean' ? options['hybridOriginals'] : HOOK_DEFAULTS.hybridOriginals,
    compactionLog:
      typeof options['compactionLog'] === 'boolean' ? options['compactionLog'] : HOOK_DEFAULTS.compactionLog,
    ...(options['stateWording'] === 'archive' || options['stateWording'] === 'delete'
      ? { stateWording: options['stateWording'] }
      : {}),
    ...(typeof options['resultChecks'] === 'boolean' ? { resultChecks: options['resultChecks'] } : {}),
    ...(parseToolThresholds(options['keepThresholdByTool'])
      ? { keepThresholdByTool: parseToolThresholds(options['keepThresholdByTool']) }
      : {}),
    injectTrim:
      typeof options['injectTrim'] === 'boolean' ? options['injectTrim'] : HOOK_DEFAULTS.injectTrim,
    replyTrim:
      typeof options['replyTrim'] === 'boolean' ? options['replyTrim'] : HOOK_DEFAULTS.replyTrim,
    exchangeDrop:
      typeof options['exchangeDrop'] === 'boolean' ? options['exchangeDrop'] : HOOK_DEFAULTS.exchangeDrop,
    ...(typeof options['stepTrim'] === 'boolean' ? { stepTrim: options['stepTrim'] } : {}),
    ...(typeof options['dedupeRepeats'] === 'boolean'
      ? { dedupeRepeats: options['dedupeRepeats'] }
      : {}),
  };
  const apiKey = optionString(options, 'apiKey');
  if (apiKey) config.apiKey = apiKey;
  const goal = optionString(options, 'goal');
  if (goal) config.goal = goal;
  return config;
}

/**
 * `keepThresholdByTool` as written in the plugin options, `Bash=0.2, Read=0.1`
 * (what `eval/calibrate.ts` prints): tool name to a threshold in [0, 1].
 * Undefined when nothing valid is given; invalid entries are left out.
 */
export function parseToolThresholds(value: unknown): Record<string, number> | undefined {
  if (typeof value !== 'string' || !value.trim()) return undefined;
  const out: Record<string, number> = {};
  for (const part of value.split(',')) {
    const match = /^\s*([A-Za-z0-9_:.-]+)\s*=\s*([0-9]*\.?[0-9]+)\s*$/.exec(part);
    if (!match) continue;
    const threshold = Number(match[2]);
    if (threshold >= 0 && threshold <= 1) out[match[1]!] = threshold;
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

/** Resolves after `ms`; rejects with the signal's reason when `signal` aborts. */
export type Sleep = (ms: number, signal?: AbortSignal) => Promise<void>;

export type JevAskerOptions = {
  /** TypeSafe's System One endpoint; `TYPESAFE_BASE_URL` overrides it (a staging host, a local test server). */
  baseUrl?: string;
  /** Deadline for each attempt. Default 20000 ms. */
  timeoutMs?: number;
  /** The hook's `next.signal`: ends a pending request when the dispatch ends. */
  signal?: AbortSignal;
  /**
   * The timer behind the deadline and the retry delay. The hook passes
   * `$.clock.sleep`: its runtime has no `setTimeout`, `AbortSignal.timeout`
   * or `AbortSignal.any`. Defaults to `setTimeout`, for use outside a hook.
   */
  sleep?: Sleep;
  /**
   * The hook's remaining budget (`next.budget.remainingMs`). A `$.clock` wait
   * is the one `$` call that spends it, so a retry delay it cannot afford is
   * skipped rather than overrunning the hook.
   */
  remainingMs?: () => number;
};

const JEV_TIMEOUT_MS = 20_000;
/** Budget a retry must leave for its own request and the hook's work after it. */
const RETRY_MARGIN_MS = 2_000;

function abortReason(signal: AbortSignal | undefined): unknown {
  return signal?.reason ?? new Error('Jev request aborted');
}

type Timers = {
  setTimeout: (fn: () => void, ms: number) => unknown;
  clearTimeout: (timer: unknown) => void;
};

const timerSleep: Sleep = (ms, signal) =>
  new Promise((resolve, reject) => {
    // Not in the hook runtime's globals; only reached outside a hook.
    const { setTimeout, clearTimeout } = globalThis as unknown as Timers;
    if (signal?.aborted) return reject(abortReason(signal));
    const onAbort = () => {
      clearTimeout(timer);
      reject(abortReason(signal));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal?.addEventListener('abort', onAbort, { once: true });
  });

/**
 * Races `promise` against `timeoutMs` and `signal`. `$.http.fetch` takes no
 * signal of its own, so a hung call is abandoned rather than cancelled; what
 * matters is that the hook stops waiting on it. Once the race settles, the
 * deadline's wait is cancelled too.
 */
async function bounded<T>(
  promise: Promise<T>,
  timeoutMs: number,
  signal: AbortSignal | undefined,
  what: string,
  sleep: Sleep,
): Promise<T> {
  // A call abandoned by a lost race may still reject later; that is expected.
  promise.catch(() => {});
  if (signal?.aborted) throw abortReason(signal);
  const stop = new AbortController();
  const forward = () => stop.abort(abortReason(signal));
  signal?.addEventListener('abort', forward, { once: true });
  const deadline = sleep(timeoutMs, stop.signal).then(
    () => {
      const error = new Error(`${what} timed out after ${timeoutMs} ms`);
      error.name = 'TimeoutError';
      throw error;
    },
    (error: unknown) => {
      throw signal?.aborted ? abortReason(signal) : error;
    },
  );
  deadline.catch(() => {});
  try {
    return await Promise.race([promise, deadline]);
  } finally {
    signal?.removeEventListener('abort', forward);
    if (!stop.signal.aborted) stop.abort();
  }
}

/**
 * A `JevAsker` over the engine's `$.http.fetch`: each attempt is bounded by a
 * timeout and the hook's signal, and a 429 or 5xx is retried once, after the
 * `retry-after` delay when the engine reports one. Once one attempt times out,
 * every later ask fails at once: the stages after it would otherwise each wait
 * out their own timeout before the fallback runs.
 */
export function jevAsker(
  fetchFn: HookFetch,
  apiKey: string,
  model: string,
  options: JevAskerOptions = {},
): JevAsker {
  const timeoutMs = options.timeoutMs ?? JEV_TIMEOUT_MS;
  const { signal } = options;
  const sleep = options.sleep ?? timerSleep;
  const remainingMs = options.remainingMs ?? (() => Infinity);
  let timedOut: Error | undefined;
  return {
    async ask(state, questions) {
      if (timedOut) throw new Error(`Jev unavailable: ${timedOut.message}`);
      // The state carries transcript text and tool inputs verbatim, and each
      // keep question quotes a head and tail of its tool result; a command,
      // an env var, or a pasted key in either must not leave the machine.
      const request = buildJevRequest(
        { apiKey, model, ...(options.baseUrl ? { baseUrl: options.baseUrl } : {}) },
        redactCompactionState(state, apiKey),
        redactCompactionState(questions, apiKey),
      );
      const attempt = () =>
        bounded(
          fetchFn(request.url, { method: request.method, headers: request.headers, body: request.body }),
          timeoutMs,
          signal,
          'Jev request',
          sleep,
        ).catch((error: unknown) => {
          if (error instanceof Error && error.name === 'TimeoutError') timedOut = error;
          throw error;
        });
      let response = await attempt();
      if (!response.ok && isRetryableStatus(response.status)) {
        const delay = retryDelayMs(response.headers?.['retry-after']);
        if (delay < remainingMs() - RETRY_MARGIN_MS) {
          if (delay > 0) await sleep(delay, signal);
          response = await attempt();
        }
      }
      return parseJevResponse(response.status, response.ok, response.text);
    },
  };
}

function toolUseSummary(tool: ToolUse): ToolUseSummary {
  const summary: ToolUseSummary = {
    tool_use_id: tool.tool_use_id,
    tool: tool.tool,
    input: tool.input,
  };
  if (tool.text !== undefined) summary.text = tool.text;
  if (tool.isError) summary.isError = true;
  return summary;
}

function toolResultSummary(result: ToolResult): ToolResultSummary {
  return {
    tool_use_id: result.tool_use_id,
    text: result.text,
    isError: result.isError ?? false,
  };
}

/**
 * Maps the library's output back onto session messages. Whatever came back
 * unchanged (a message, a tool use, a tool result) is the engine's own object,
 * handle included; anything rebuilt is a fresh message without a handle, so the
 * engine takes the edited content instead of its original.
 */
export function toSessionMessages(
  input: readonly SessionMessage[],
  output: readonly Message[],
): SessionMessage[] {
  const messages = new Map<Message, SessionMessage>();
  const uses = new Map<ToolUse, ToolUseSummary>();
  const results = new Map<ToolResult, ToolResultSummary>();
  for (const message of input) {
    messages.set(message, message);
    for (const tool of message.toolUses) uses.set(tool, tool);
    for (const result of message.toolResults ?? []) results.set(result, result);
  }
  return output.map((message) => {
    const own = messages.get(message);
    if (own) return own;
    const rebuilt: SessionMessage = {
      role: message.role,
      text: message.text,
      toolUses: message.toolUses.map((tool) => uses.get(tool) ?? toolUseSummary(tool)),
    };
    if (message.toolResults && message.toolResults.length > 0) {
      rebuilt.toolResults = message.toolResults.map(
        (result) => results.get(result) ?? toolResultSummary(result),
      );
    }
    return rebuilt;
  });
}

function mergeRun(run: readonly SessionMessage[]): SessionMessage {
  const first = run[0]!;
  const merged: SessionMessage = {
    role: first.role,
    text: run.map((message) => message.text).filter((text) => text.length > 0).join('\n\n'),
    toolUses: run.flatMap((message) => message.toolUses),
  };
  const toolResults = run.flatMap((message) => message.toolResults ?? []);
  if (toolResults.length > 0) merged.toolResults = toolResults;
  return merged;
}

/**
 * Puts the compacted conversation in the shape that survives `--resume` and
 * keeps parallel tool calls paired. Two engine behaviours drive it
 * (anthropics/claude-code#95328, tamaratran/fast-jev-compaction#89 and #137):
 *
 * - A kept message that still carries its `handle` is re-saved pointing
 *   behind the compaction, and on `--resume` the engine walks back through it
 *   and restores the whole history the compaction removed. Every message
 *   holding tool results and every assistant message except the newest run
 *   therefore goes back without its handle. User text keeps its handle: it
 *   is not one of the two causes, and it carries the user's images.
 * - One API response with parallel calls is several assistant rows sharing
 *   one engine id. A row without a handle gets a new id, which splits the
 *   response, and the engine then reports a sibling's result as missing. So a
 *   run of consecutive assistant rows with any row lacking a handle is merged
 *   into one message, and likewise the run of tool-result rows after it.
 *
 * What a message loses with its handle is what the engine kept beside its
 * text and tool blocks: hidden reasoning, and images inside tool results.
 * The newest response (the trailing assistant rows, up to a dropped call's
 * note) keeps its handles when compaction left all of it whole, so a tool
 * loop in progress keeps its reasoning; if a row of it was rebuilt, the
 * whole response is merged like any other.
 */
export function resumeSafeMessages(messages: readonly SessionMessage[]): SessionMessage[] {
  // The newest response: the trailing run of assistant rows, less any
  // leading dropped-call notes (rows with neither a handle nor a tool call).
  // Rows of an older response can only sit in this run when their results
  // were removed, i.e. as such notes; a note after a call of the run belongs
  // to the newest response itself, and mergeRuns then merges it whole.
  let newestRun = messages.length;
  let last = messages.length - 1;
  while (last >= 0 && messages[last]!.role !== 'assistant') last -= 1;
  if (last >= 0) {
    newestRun = last;
    while (newestRun > 0 && messages[newestRun - 1]!.role === 'assistant') newestRun -= 1;
    const isNote = (m: SessionMessage) => m.handle === undefined && m.toolUses.length === 0;
    while (newestRun < last && isNote(messages[newestRun]!)) newestRun += 1;
  }
  const stripped = messages.map((message, index) => {
    if (message.handle === undefined) return message;
    const carriesResults = (message.toolResults?.length ?? 0) > 0;
    const olderAssistant = message.role === 'assistant' && index < newestRun;
    if (!carriesResults && !olderAssistant) return message;
    const copy: SessionMessage = { role: message.role, text: message.text, toolUses: message.toolUses };
    if (message.toolResults) copy.toolResults = message.toolResults;
    return copy;
  });
  // A row that carried nothing but its handle (an interrupted response's
  // thinking) would go back as an empty message, which the API refuses.
  const isEmpty = (m: SessionMessage) =>
    m.handle === undefined && m.text.length === 0 && m.toolUses.length === 0 && (m.toolResults?.length ?? 0) === 0;
  const before = stripped.slice(0, newestRun).filter((m) => !isEmpty(m));
  return [...mergeRuns(before), ...mergeRuns(stripped.slice(newestRun).filter((m) => !isEmpty(m)))];
}

/**
 * Merges each run of consecutive assistant rows, and each run of consecutive
 * text-less tool-result rows, that has a row without a handle. A result row
 * with text of its own stays apart, so its tool_result blocks still come
 * first in its message.
 */
function mergeRuns(messages: readonly SessionMessage[]): SessionMessage[] {
  const isAssistant = (m: SessionMessage) => m.role === 'assistant';
  const isResultRow = (m: SessionMessage) =>
    m.role === 'user' && (m.toolResults?.length ?? 0) > 0 && m.text.length === 0;
  const out: SessionMessage[] = [];
  let index = 0;
  while (index < messages.length) {
    const message = messages[index]!;
    const sameRun = isAssistant(message) ? isAssistant : isResultRow(message) ? isResultRow : undefined;
    let end = index + 1;
    if (sameRun) while (end < messages.length && sameRun(messages[end]!)) end += 1;
    const run = messages.slice(index, end);
    if (run.length > 1 && run.some((m) => m.handle === undefined)) out.push(mergeRun(run));
    else out.push(...run);
    index = end;
  }
  return out;
}

/** What the prose-trim rescue did: the keep cut-off it settled on, or none when no cut-off reached the minimum or the stage failed. */
export type ProseOutcome = { cutoff?: number; stats: ProseStats };

/** What the notification-trim rescue did: the rung it settled on, or none when no rung reached the minimum or the stage failed. */
export type NotificationOutcome = { rung?: Rung; stats: NotificationStats };

/** What the injected-content rescue did: the rung it settled on, or none when no rung reached the minimum or the stage failed. */
export type InjectOutcome = { rung?: Rung; stats: InjectStats };

export type SessionCompaction = {
  result: CompactResult;
  messages: SessionMessage[];
  /** Set only when `proseTrim` is on and tool compaction fell short. */
  prose?: ProseOutcome;
  /** Set only when `notificationTrim` is on, an archive writer is given and tool compaction fell short. */
  notifications?: NotificationOutcome;
  /** Set only when `injectTrim` is on, an archive writer is given and tool compaction fell short. */
  injected?: InjectOutcome;
  /** Set only when `exchangeDrop` is on, an archive writer is given and every earlier stage fell short. */
  exchanges?: ExchangeOutcome;
  /** Set when step trim ran: as the last stage, or as reply cleanup on a compaction that already met the minimum. */
  steps?: StepsOutcome;
};

/** What step trim did: the pass and cut-off it settled on, or neither when it fell short or failed. */
export type StepsOutcome = { pass?: StepOutcome['pass']; cutoff?: number; stats: StepStats };

/** What the exchange drop did: the cut-off it settled on, or none when no cut-off reached the minimum or the stage failed. */
export type ExchangeOutcome = { cutoff?: number; stats: ExchangeStats };

const ARCHIVE_DIR = '.claude/jev-compaction';
/** Where archives went before the plugin was renamed (0.18.0 and earlier); old markers still cite it. */
const OLD_ARCHIVE_DIR = '.claude/fast-jev-compaction';

export type ArchiveWriter = {
  /** Names the file a dropped call's or result's full text will be written to. */
  cite: (toolUseId: string, text: string) => string | undefined;
  /** Names the file an image's or document's base64 copy will be written to, with its INDEX.md line (see `placeMedia`). */
  media: MediaArchive;
  /** Writes the files named so far. */
  flush: () => Promise<void>;
  /** The path `cite` returns for an archive id. */
  pathOf: (id: string) => string;
};

/**
 * The archive directory, absolute: under the git toplevel of the session's
 * directory, else under the session root. A relative path would resolve under
 * whatever directory the session runs in and scatter archives across a repo.
 * Falls back to the relative directory when the engine cannot answer.
 */
export async function archiveDir($: {
  session: { cwd: () => Promise<string>; root: () => Promise<string> };
  process: {
    run: (argv: readonly string[], init?: { cwd?: string }) => Promise<{ exitCode: number; stdout: string }>;
  };
}): Promise<string> {
  try {
    const cwd = await $.session.cwd();
    const git = await $.process.run(['git', 'rev-parse', '--show-toplevel'], { cwd });
    const top = git.exitCode === 0 ? git.stdout.trim() : '';
    return `${top || (await $.session.root())}/${ARCHIVE_DIR}`;
  } catch {
    return ARCHIVE_DIR;
  }
}

/**
 * Saves what compaction drops to `dir` (default `.claude/jev-compaction/`), so a detail
 * buried in a dropped call or result can still be read back later. The text is
 * written with the same redaction the Jev request applies, so a credential in
 * it is replaced by `[REDACTED]` rather than stored, and the rest is kept.
 */
export function archiveWriter($: {
  fs: {
    exists: (path: string) => Promise<boolean>;
    write: (path: string, text: string) => Promise<void>;
    read: (path: string) => Promise<unknown>;
  };
}, dir: string = ARCHIVE_DIR, apiKey?: string): ArchiveWriter {
  const pending = new Map<string, string>();
  // Base64 is written as it is: redaction could only corrupt it. Its INDEX.md line is kept apart.
  const media = new Map<string, { data: string; line: string }>();
  const pathOf = (id: string) => `${dir}/${id}.txt`;
  return {
    pathOf,
    cite(toolUseId, text) {
      if (!text) return undefined;
      const path = pathOf(toolUseId);
      pending.set(path, redactText(text, apiKey));
      return path;
    },
    media(name, data, line) {
      const path = `${dir}/${name}`;
      media.set(path, { data, line: redactText(line, apiKey) });
      return path;
    },
    async flush() {
      if (pending.size === 0 && media.size === 0) return;
      const ignorePath = `${dir}/.gitignore`;
      if (!(await $.fs.exists(ignorePath))) await $.fs.write(ignorePath, '*\n');
      for (const [path, text] of pending) await $.fs.write(path, text);
      for (const [path, { data }] of media) await $.fs.write(path, data);
      await writeIndex($, `${dir}/INDEX.md`, new Map([...pending, ...[...media].map(([path, { line }]) => [path, line] as const)]));
      pending.clear();
      media.clear();
    },
  };
}

/** Past this size `log.jsonl` keeps only its newer half. */
const LOG_MAX_CHARS = 2_000_000;

/**
 * Appends JSON lines to the archive's `log.jsonl`: one per compaction (figures
 * and per-call decisions) and one per archive file read back, the record
 * `eval/calibrate.ts` fits per-tool thresholds from. Best effort: a failure
 * changes nothing else.
 */
async function appendLog(
  $: { fs: { write: (path: string, text: string) => Promise<void>; read: (path: string) => Promise<unknown> } },
  path: string,
  entries: readonly Record<string, unknown>[],
): Promise<void> {
  try {
    let previous = '';
    try {
      const existing = await $.fs.read(path);
      if (typeof existing === 'string') previous = existing;
    } catch {
      previous = '';
    }
    if (previous.length > LOG_MAX_CHARS) previous = previous.slice(previous.indexOf('\n', previous.length - LOG_MAX_CHARS / 2) + 1);
    await $.fs.write(path, `${previous}${entries.map((entry) => JSON.stringify(entry)).join('\n')}\n`);
  } catch {
    // The log is a record for calibration; losing a line loses nothing else.
  }
}

/** The archive files (`name.txt`) a tool call's input names, for the read-back record. */
export function archiveFilesNamed(input: Record<string, unknown>): string[] {
  const text = ['file_path', 'path', 'command', 'pattern']
    .map((key) => (typeof input[key] === 'string' ? (input[key] as string) : ''))
    .join(' ');
  if (!text.includes(ARCHIVE_DIR) && !text.includes(OLD_ARCHIVE_DIR)) return [];
  const files = [...text.matchAll(/\.claude\/(?:fast-)?jev-compaction\/([A-Za-z0-9_.-]+\.txt)/g)].map((m) => m[1]!);
  return files.length > 0 ? [...new Set(files)] : ['*'];
}

/** Records the archive files a Read, Grep or Bash call read back (see `compactionLog`). */
async function noteReadBack(
  $: Parameters<Hook<'tool.call'>>[0],
  configured: HookConfig,
  tool: string,
  input: Record<string, unknown>,
): Promise<void> {
  if (!configured.compactionLog || !configured.archiveResults) return;
  const files = archiveFilesNamed(input);
  if (files.length === 0) return;
  try {
    const at = await $.clock.now();
    await appendLog($, `${await archiveDir($)}/log.jsonl`, [{ time: new Date(at).toISOString(), readBack: files, via: tool }]);
  } catch {
    // A missing record changes nothing for the tool call.
  }
}

/** Largest INDEX.md the plugin keeps adding to; past it the index stops growing. */
const INDEX_MAX_CHARS = 1_000_000;

/**
 * Adds one line per newly archived file to the archive's INDEX.md (file name and
 * the start of its text), so a detail can be found with one grep instead of
 * opening every file. Best effort: the archive files themselves are written
 * first and stay the source of truth.
 */
async function writeIndex(
  $: { fs: { write: (path: string, text: string) => Promise<void>; read: (path: string) => Promise<unknown> } },
  path: string,
  files: ReadonlyMap<string, string>,
): Promise<void> {
  try {
    let index = '';
    try {
      const existing = await $.fs.read(path);
      if (typeof existing === 'string') index = existing;
    } catch {
      index = '';
    }
    if (index.length > INDEX_MAX_CHARS) return;
    if (!index) index = '# jev-compaction archive\n\nOne line per archived item: its file, then how it starts.\n\n';
    const lines: string[] = [];
    for (const [file, text] of files) {
      const name = file.slice(file.lastIndexOf('/') + 1);
      if (index.includes(`- ${name}:`)) continue;
      const start = text.replace(/\s+/g, ' ').trim();
      lines.push(`- ${name}: ${start.length > 140 ? `${headOf(start, 139)}…` : start}`);
    }
    if (lines.length > 0) await $.fs.write(path, `${index}${lines.join('\n')}\n`);
  } catch {
    // The index is a convenience; a failed write leaves the archive as it is.
  }
}

/** Runs the library over a session transcript; throws when the key is missing or Jev fails. */
export async function compactSession(
  messages: readonly SessionMessage[],
  config: HookConfig,
  fetchFn: HookFetch,
  archive?: ArchiveWriter,
  askerOptions: JevAskerOptions = {},
): Promise<SessionCompaction> {
  if (!config.apiKey) throw new Error('TYPESAFE_API_KEY is not configured');
  const asker = jevAsker(fetchFn, config.apiKey, config.model, askerOptions);
  let result = await compact(messages, asker, {
    ...config,
    ...(archive ? { archive: archive.cite } : {}),
  });
  if (config.inputTrim) {
    const options = resolveOptions(config);
    const cut = elideInputs(
      result.messages,
      callActions(messages, result.decisions, options.preserveRecentMessages),
      { ...options, ...(archive ? { archive: archive.cite } : {}) },
    );
    if (cut.fields > 0) {
      result = {
        ...result,
        messages: cut.messages,
        stats: { ...result.stats, charsAfter: cut.messages.reduce((sum, m) => sum + messageChars(m), 0), inputsElided: cut.fields },
      };
    }
  }
  if (archive) await archive.flush();
  const tool = { result, messages: toSessionMessages(messages, result.messages) };
  const trimNotifications =
    (config.notificationTrim ?? HOOK_DEFAULTS.notificationTrim) && archive !== undefined;
  const trimInjected = (config.injectTrim ?? HOOK_DEFAULTS.injectTrim) && archive !== undefined;
  const finish = (out: SessionCompaction): SessionCompaction =>
    (config.stripHandles ?? HOOK_DEFAULTS.stripHandles)
      ? { ...out, messages: resumeSafeMessages(out.messages) }
      : out;
  const dropping = (config.exchangeDrop ?? HOOK_DEFAULTS.exchangeDrop) && archive !== undefined;
  const stepping = (config.stepTrim ?? HOOK_DEFAULTS.stepTrim) && archive !== undefined;
  if (reductionRatio(result) >= config.minReductionRatio) {
    if (!(config.replyTrim ?? HOOK_DEFAULTS.replyTrim) || !archive) return finish(tool);
    return finish({ ...tool, ...(await trimSteps(messages, result, asker, config, archive, 'mild')) });
  }
  const trimming = config.proseTrim || trimNotifications || trimInjected;
  if (!trimming && !dropping && !stepping) return finish(tool);
  const { best, ...rescue }: Partial<SessionCompaction> & { best?: CompactResult } = !trimming
    ? {}
    : (trimNotifications || trimInjected) && archive
      ? await rescueWithTrim(messages, result, asker, config, archive)
      : await rescueWithProse(messages, result, asker, config, archive);
  if (archive) await archive.flush();
  if (rescue.result || !archive || (!dropping && !stepping)) return finish({ ...tool, ...rescue });
  // Each later stage builds on the best result so far; what an earlier stage archived stays archived.
  let base = best ?? result;
  let out: SessionCompaction = { ...tool, ...rescue };
  if (dropping) {
    const dropped = await dropExchanges(messages, base, asker, config, archive);
    const tried = dropped.exchanges.stats.eligible > 0 || dropped.exchanges.stats.failed !== undefined;
    if (tried) out = { ...out, exchanges: dropped.exchanges };
    if (dropped.result) {
      base = dropped.result;
      if (reductionRatio(base) >= config.minReductionRatio) return finish({ ...out, result: base, messages: toSessionMessages(messages, base.messages) });
    }
  }
  if (stepping) {
    const stepped = await trimSteps(messages, base, asker, config, archive, 'ladder');
    if (stepped.result) return finish({ ...out, ...stepped });
    out = { ...out, steps: stepped.steps };
  }
  return finish(base === result && !best ? out : { ...out, result: base, messages: toSessionMessages(messages, base.messages) });
}

/**
 * Step trim on `result`. `ladder`: the last stage, archiving steps until the minimum is met (see
 * `stepLadder`). `mild`: reply cleanup on a compaction that already met it (see `mildSteps`).
 * Returns only `steps` when nothing is archived or the stage failed.
 */
async function trimSteps(
  messages: readonly SessionMessage[],
  result: CompactResult,
  asker: JevAsker,
  config: HookConfig,
  archive: ArchiveWriter,
  mode: 'ladder' | 'mild',
): Promise<Partial<SessionCompaction> & { steps: StepsOutcome }> {
  const scores = await scoreSteps(result.messages, asker, resolveOptions(config), mode === 'mild' ? (s) => !s.current : undefined);
  if (scores.stats.failed !== undefined || scores.steps.length === 0) return { steps: { stats: scores.stats } };
  const picked =
    mode === 'mild'
      ? mildSteps(scores, result, (id) => archive.pathOf(id))
      : stepLadder(scores, result, config.minReductionRatio, (id) => archive.pathOf(id));
  if (!picked) return { steps: { stats: scores.stats } };
  for (const item of picked.applied.archived) {
    if (archive.cite(item.id, item.text) !== item.path) throw new Error(`archive path for ${item.id} is not ${item.path}`);
  }
  await archive.flush();
  return {
    result: picked.result,
    messages: toSessionMessages(messages, picked.result.messages),
    steps: { pass: picked.pass, ...(picked.cutoff !== undefined ? { cutoff: picked.cutoff } : {}), stats: picked.applied.stats },
  };
}

/**
 * Drops old finished exchanges Jev no longer needs, each archived whole first, on the best result
 * so far. When no cut-off reaches the minimum it still returns the loosest cut-off's result, for
 * step trim to build on; returns only `exchanges` when nothing drops or the stage fails.
 */
async function dropExchanges(
  messages: readonly SessionMessage[],
  result: CompactResult,
  asker: JevAsker,
  config: HookConfig,
  archive: ArchiveWriter,
): Promise<Partial<SessionCompaction> & { exchanges: ExchangeOutcome }> {
  const scores = await scoreExchanges(result.messages, asker, resolveOptions(config));
  if (scores.stats.failed !== undefined || scores.exchanges.length === 0) return { exchanges: { stats: scores.stats } };
  let ladder = exchangeLadder(scores, result, config.minReductionRatio, (id) => archive.pathOf(id));
  if (!ladder) {
    // Nothing reaches the minimum: keep the loosest cut-off's drops for step trim to build on.
    const cutoff = PROSE_KEEP_LADDER[PROSE_KEEP_LADDER.length - 1] as number;
    const applied = applyExchanges(scores, cutoff, (id) => archive.pathOf(id));
    if (applied.stats.dropped === 0) return { exchanges: { stats: scores.stats } };
    const charsAfter = applied.messages.reduce((sum, m) => sum + messageChars(m), 0);
    ladder = { cutoff, result: { ...result, messages: applied.messages, stats: { ...result.stats, charsAfter } }, applied };
  }
  for (const item of ladder.applied.archived) {
    if (archive.cite(item.id, item.text) !== item.path) throw new Error(`archive path for ${item.id} is not ${item.path}`);
  }
  await archive.flush();
  return {
    result: ladder.result,
    messages: toSessionMessages(messages, ladder.result.messages),
    exchanges: { cutoff: ladder.cutoff, stats: ladder.applied.stats },
  };
}

/**
 * Scores the prose of tool compaction's result with Jev once, then walks `PROSE_KEEP_LADDER` offline
 * on those scores and takes the first cut-off whose result reaches `minReductionRatio` against the
 * original messages. Returns only `prose` when none does or the stage fails, so the caller keeps
 * tool compaction's result.
 */
async function rescueWithProse(
  messages: readonly SessionMessage[],
  result: CompactResult,
  asker: JevAsker,
  config: HookConfig,
  archive?: ArchiveWriter,
): Promise<Partial<SessionCompaction> & { prose: ProseOutcome }> {
  const options = resolveOptions(config);
  let scores;
  try {
    scores = await scoreProse(
      result.messages,
      callActions(messages, result.decisions, options.preserveRecentMessages),
      asker,
      archive ? { ...options, archive: archive.cite } : options,
    );
  } catch (error) {
    return { prose: { stats: failedProseStats(error) } };
  }
  if (scores.stats.failed !== undefined) return { prose: { stats: scores.stats } };
  const rescue = ladderRescue(scores, result, config.minReductionRatio, archive && (config.replyTrim ?? HOOK_DEFAULTS.replyTrim) ? (id) => archive.pathOf(id) : undefined);
  if (rescue) {
    for (const item of rescue.archived) {
      if (archive?.cite(item.id, item.text) !== item.path) throw new Error(`archive path for ${item.id} is not ${item.path}`);
    }
    return {
      result: rescue.result,
      messages: toSessionMessages(messages, rescue.result.messages),
      prose: { cutoff: rescue.cutoff, stats: rescue.stats },
    };
  }
  return { prose: { stats: scores.stats } };
}

/**
 * The rescue with notification trim on: scores the prose (when `proseTrim` is on) and the
 * `<task-notification>` reports once each, then walks the prose ladder and a final `all` rung
 * offline, applying both stages at each rung, and takes the first whose result reaches
 * `minReductionRatio` against the original messages. Only the chosen rung's archive files are
 * handed to the writer. A stage that fails is skipped and the other still runs.
 */
async function rescueWithTrim(
  messages: readonly SessionMessage[],
  result: CompactResult,
  asker: JevAsker,
  config: HookConfig,
  archive: ArchiveWriter,
): Promise<Partial<SessionCompaction> & { best?: CompactResult }> {
  const { out, rescue, best } = await trimRescue({
    messages,
    result,
    asker,
    options: { ...resolveOptions(config), imagesInContext: config.imagesInContext ?? true },
    proseTrim: config.proseTrim,
    notificationTrim: config.notificationTrim ?? HOOK_DEFAULTS.notificationTrim,
    injectTrim: config.injectTrim ?? HOOK_DEFAULTS.injectTrim,
    replyTrim: config.replyTrim ?? HOOK_DEFAULTS.replyTrim,
    minRatio: config.minReductionRatio,
    archive,
    pathOf: archive.pathOf,
  });
  const chosen = rescue ?? best;
  if (!chosen) return out;
  const { prose, notified, injected } = chosen.applied;
  const lastProse = PROSE_KEEP_LADDER[PROSE_KEEP_LADDER.length - 1] as number;
  return {
    ...(rescue ? { result: rescue.result, messages: toSessionMessages(messages, rescue.result.messages) } : { best: chosen.result }),
    ...(prose ? { prose: { cutoff: chosen.rung === 'all' ? lastProse : chosen.rung, stats: prose.stats } } : out.prose ? { prose: out.prose } : {}),
    ...(notified ? { notifications: { rung: chosen.rung, stats: notified.stats } } : out.notifications ? { notifications: out.notifications } : {}),
    ...(injected ? { injected: { rung: chosen.rung, stats: injected.stats } } : out.injected ? { injected: out.injected } : {}),
  };
}

function percent(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}

export function summarize(
  result: CompactResult,
  prose?: ProseOutcome,
  notifications?: NotificationOutcome,
  injected?: InjectOutcome,
  exchanges?: ExchangeOutcome,
  steps?: StepsOutcome,
): string {
  const { stats } = result;
  const parts = [
    stats.kept > 0 ? `${stats.kept} kept` : '',
    stats.resultsDropped > 0 ? `${stats.resultsDropped} results truncated` : '',
    stats.callsDropped > 0 ? `${stats.callsDropped} call_dropped` : '',
    stats.pinned > 0 ? `${stats.pinned} pinned` : '',
    stats.small > 0 ? `${stats.small} small kept` : '',
    stats.compacted > 0 ? `${stats.compacted} already compacted` : '',
    stats.duplicates > 0 ? `${stats.duplicates} repeat calls truncated` : '',
    stats.blocksDeduped > 0 ? `${stats.blocksDeduped} repeat blocks removed` : '',
  ].filter(Boolean);
  return `${percent(reductionRatio(result))} reduction; ${
    parts.join(', ') || 'no tool calls'
  }; state ~${stats.stateTokens} tokens (${stats.stateStage}) in ${stats.requests} request(s)${
    prose?.cutoff === undefined || prose.stats.proseCharsSaved + prose.stats.inputCharsSaved <= 0
      ? ''
      : `; prose trim at ${prose.cutoff}: ${prose.stats.dropped} dropped, ${prose.stats.inputsElided} inputs elided, ${
          prose.stats.proseCharsSaved + prose.stats.inputCharsSaved
        } chars`
  }${
    notifications?.rung === undefined
      ? ''
      : `; notification trim at ${notifications.rung}: ${notifications.stats.folded} folded, ${notifications.stats.stripped} stripped, ${notifications.stats.charsSaved} chars`
  }${
    injected?.rung === undefined
      ? ''
      : `; injected-content fold at ${injected.rung}: ${injected.stats.folded} folded, ${injected.stats.charsSaved} chars`
  }${
    exchanges?.cutoff === undefined
      ? ''
      : `; exchange drop at ${exchanges.cutoff}: ${exchanges.stats.dropped} archived, ${exchanges.stats.charsSaved} chars`
  }${
    steps?.pass === undefined
      ? ''
      : `; step trim (${steps.pass === 'fill' ? 'lowest-scored first' : `at ${steps.cutoff}`}): ${steps.stats.archived} archived, ${steps.stats.charsSaved} chars`
  }`;
}

const UI_LOG_MAX_CHARS = 4096;
const TOAST_MS = 6_000;

export function decisionLog(result: CompactResult): string {
  return result.decisions
    .filter((d) => d.reason !== 'pinned' && d.reason !== 'small' && d.reason !== 'compacted')
    .map((d) =>
      d.reason === 'duplicate'
        ? `${d.id}:${d.tool}:${d.action}/duplicate`
        : `${d.id}:${d.tool}:${d.action}/call=${d.keepCall.toFixed(2)}/result=${d.keepResult.toFixed(2)}`,
    )
    .join(' ');
}

export function decisionLogLines(
  result: CompactResult,
  maxChars: number = UI_LOG_MAX_CHARS,
): string[] {
  const entries = decisionLog(result).split(' ').filter(Boolean);
  if (entries.length === 0) return ['decisions: (none)'];
  const chunks: string[] = [];
  let current = '';
  for (const entry of entries) {
    const next = current ? `${current} ${entry}` : entry;
    if (current && next.length > maxChars - 24) {
      chunks.push(current);
      current = entry;
    } else current = next;
  }
  chunks.push(current);
  return chunks.map((chunk, index) =>
    chunks.length === 1
      ? `decisions: ${chunk}`
      : `decisions (${index + 1}/${chunks.length}): ${chunk}`,
  );
}

function settingsEnv(settings: Readonly<Record<string, unknown>>, name: string): string | undefined {
  const env = settings['env'];
  if (!env || typeof env !== 'object') return undefined;
  const value = (env as Record<string, unknown>)[name];
  return typeof value === 'string' && value ? value : undefined;
}

/**
 * The `apiKey` option, then TYPESAFE_API_KEY from the environment and then
 * the settings' `env` block; an empty string counts as unset at every step.
 */
export async function getApiKey(
  $: {
    env: { get: (name: string) => Promise<string | undefined> };
    settings: { read: () => Promise<Readonly<Record<string, unknown>>> };
  },
  config: HookConfig,
): Promise<string | undefined> {
  let key = config.apiKey;
  if (!key) {
    const settings = await $.settings.read();
    // $.env.get needs a literal name at each call site so the plugin's env
    // reads can be listed statically.
    key = (await $.env.get('TYPESAFE_API_KEY')) || settingsEnv(settings, 'TYPESAFE_API_KEY');
  }
  return key || undefined;
}

/**
 * `TYPESAFE_BASE_URL` from the environment, then the settings' `env` block: an
 * http(s) URL that replaces TypeSafe's endpoint, for a staging host or a local
 * test server. Undefined (TypeSafe's own endpoint) when unset or not http(s).
 */
export async function getBaseUrl($: {
  env: { get: (name: string) => Promise<string | undefined> };
  settings: { read: () => Promise<Readonly<Record<string, unknown>>> };
}): Promise<string | undefined> {
  try {
    // The loader requires `$.env.get('<literal>')` spelled out at the call site.
    const url = (await $.env.get('TYPESAFE_BASE_URL')) || settingsEnv(await $.settings.read(), 'TYPESAFE_BASE_URL');
    return url && /^https?:\/\/\S+$/.test(url) ? url : undefined;
  } catch {
    // An engine (or a test double) without env or settings uses TypeSafe's own endpoint.
    return undefined;
  }
}

/** `$.clock.sleep` as a `Sleep`: the hook runtime's only timer. */
function clockSleep($: {
  clock: { sleep: (ms: number, options?: { signal?: AbortSignal }) => Promise<void> };
}): Sleep {
  return (ms, signal) => $.clock.sleep(ms, signal ? { signal } : undefined);
}

type UiLog = { ui: { log: (text: string, options?: { to?: 'transcript' | 'debug' }) => void } };

/**
 * Diagnostics go to the debug log only: `$.ui.log` defaults to the transcript,
 * where every compaction would add a dozen dim rows to the chat.
 */
/** The debug lines and the toast of one compaction, kept for `/jev`. */
type Report = {
  lines: string[];
  toast?: string;
  /** The compaction's figures and per-call decisions, for the archive's `log.jsonl`. */
  ledger?: Record<string, unknown>;
};

function log($: UiLog, text: string, report?: Report): void {
  report?.lines.push(text);
  try {
    $.ui.log(text, { to: 'debug' });
  } catch {
    // Diagnostics are best-effort; the compaction result must still be returned.
  }
}

/** Estimated tokens of the compactable history: text, tool inputs and results. */
export function messageTokens(messages: readonly Message[]): number {
  let total = 0;
  for (const message of messages) {
    total += estimateTokens(message.text);
    for (const tool of message.toolUses) total += estimateTokens(JSON.stringify(tool.input ?? {}));
    for (const result of message.toolResults ?? []) total += estimateTokens(result.text);
  }
  return total;
}

/**
 * The message-level cut that frees `minRatio` of the real context. The system
 * prompt, tool schemas and attachments count toward the engine's trigger but
 * are never compacted, so a cut measured on messages alone overstates the
 * headroom it buys and the next compaction fires minutes later. Without a
 * context reading, or when it is not larger than the history, `minRatio` stands.
 */
export function messageTarget(contextTokens: number | undefined, historyTokens: number, minRatio: number): number {
  if (!contextTokens || historyTokens <= 0 || contextTokens <= historyTokens) return minRatio;
  return (minRatio * contextTokens) / historyTokens;
}

/** An interrupted dispatch is vetoed quietly: no summary, nothing replaced. */
/** Share of the original history the kept recent turns may take in a hybrid rescue. */
export const HYBRID_RECENT_SHARE = 0.3;
/** Share of the original history reserved for the summary of the older part. */
const HYBRID_SUMMARY_ALLOWANCE = 0.1;

/**
 * Where to split `messages` for a hybrid rescue: the earliest user turn (a
 * user message with text and no tool results, outside any open tool run, so
 * no tool call is separated from its result) whose tail, from there to the end, is at most `share` of
 * `historyTokens`. 0 when no turn after the first qualifies.
 */
export function hybridSplit(messages: readonly Message[], historyTokens: number, share: number): number {
  // A host notice can sit inside an open parallel run (call A and B, result A,
  // notice, result B); splitting there would orphan result B, so only rows
  // with no call still waiting for its result qualify.
  const answered = new Set(messages.flatMap((m) => (m.toolResults ?? []).map((r) => r.tool_use_id)));
  const open = new Set<string>();
  const closed: boolean[] = [];
  for (const m of messages) {
    closed.push(open.size === 0);
    for (const use of m.toolUses) if (answered.has(use.tool_use_id)) open.add(use.tool_use_id);
    for (const result of m.toolResults ?? []) open.delete(result.tool_use_id);
  }
  const limit = share * historyTokens;
  let tail = 0;
  let split = 0;
  for (let i = messages.length - 1; i > 0; i -= 1) {
    tail += messageTokens([messages[i]!]);
    if (tail > limit) break;
    const m = messages[i]!;
    if (closed[i] && m.role === 'user' && m.text.trim() !== '' && (m.toolResults?.length ?? 0) === 0) split = i;
  }
  return split;
}

/**
 * The index in `original` of the prompt row at `index` in `trimmed`: the row
 * with the same engine handle, else the same-text prompt row at the same rank
 * among the prompt rows with that text. Undefined when none matches.
 */
export function originalIndex(
  original: readonly SessionMessage[],
  trimmed: readonly SessionMessage[],
  index: number,
): number | undefined {
  const row = trimmed[index];
  if (!row) return undefined;
  if (row.handle !== undefined) {
    const at = original.findIndex((m) => m.handle === row.handle);
    if (at > 0) return at;
  }
  // A prompt that gained media placeholders (see `placeMedia`) still matches its original.
  const bare = (text: string) => splitMediaNotes(text).rest;
  const isPrompt = (m: SessionMessage) =>
    m.role === 'user' && (m.toolResults?.length ?? 0) === 0 && bare(m.text) === bare(row.text);
  const rank = trimmed.slice(0, index).filter(isPrompt).length;
  let seen = 0;
  for (let i = 0; i < original.length; i++) {
    if (!isPrompt(original[i]!)) continue;
    if (seen === rank) return i > 0 ? i : undefined;
    seen += 1;
  }
  return undefined;
}

export type ContextBreakdown = {
  userText: number;
  assistantText: number;
  thinking: number;
  toolInputs: number;
  toolResults: number;
  images: number;
};

/**
 * Estimated tokens per kind of content in the Messages API form of the conversation, the form
 * the model reads (`$.session.messages({ as: 'api' })`); images are counted, not sized.
 */
export function contextBreakdown(
  messages: readonly { role: string; content: readonly unknown[] }[],
): ContextBreakdown {
  const out: ContextBreakdown = { userText: 0, assistantText: 0, thinking: 0, toolInputs: 0, toolResults: 0, images: 0 };
  const json = (value: unknown): string => {
    try {
      return JSON.stringify(value) ?? '';
    } catch {
      return '';
    }
  };
  for (const message of messages) {
    for (const raw of message.content) {
      const block = (raw ?? {}) as Record<string, unknown>;
      switch (block['type']) {
        case 'text': {
          const tokens = estimateTokens(String(block['text'] ?? ''));
          if (message.role === 'assistant') out.assistantText += tokens;
          else out.userText += tokens;
          break;
        }
        case 'thinking':
          out.thinking += estimateTokens(String(block['thinking'] ?? ''));
          break;
        case 'tool_use':
          out.toolInputs += estimateTokens(json(block['input']));
          break;
        case 'tool_result': {
          const content = block['content'];
          if (typeof content === 'string') out.toolResults += estimateTokens(content);
          else if (Array.isArray(content)) {
            for (const part of content as Record<string, unknown>[]) {
              if (part?.['type'] === 'image') out.images++;
              else out.toolResults += estimateTokens(typeof part?.['text'] === 'string' ? part['text'] : json(part));
            }
          }
          break;
        }
        case 'image':
          out.images++;
          break;
        default:
          break;
      }
    }
  }
  return out;
}

export function formatBreakdown(b: ContextBreakdown): string {
  return `user text ~${b.userText}, assistant text ~${b.assistantText}, thinking ~${b.thinking}, tool inputs ~${b.toolInputs}, tool results ~${b.toolResults}, images ${b.images}`;
}

function interrupted($: UiLog, report?: Report): { skip: string } {
  log($, 'compaction interrupted; nothing changed', report);
  return { skip: 'jev-compaction: interrupted' };
}

/** An error's message for a toast: its first line, at most 100 characters. */
function shortError(error: unknown): string {
  const text = (error instanceof Error ? error.message : String(error)).split('\n')[0]!.replace(/^(?:fast-)?jev-compaction: /, '');
  return text.length > 100 ? `${text.slice(0, 99)}…` : text;
}

/** One toast in the chat, short; the full detail goes to the debug log. */
function notify(
  $: {
    ui: UiLog['ui'] & {
      toast: (text: string, options?: { timeoutMs?: number }) => void;
    };
  },
  toast: string,
  detail: string = toast,
  report?: Report,
): void {
  log($, detail, report);
  if (report) report.toast = toast;
  try {
    $.ui.toast(toast, { timeoutMs: TOAST_MS });
  } catch {
    // A disconnected UI must not prevent the native fallback.
  }
}

/**
 * The `session.compact` hook proper; `register` wraps it to keep a report for
 * `/jev`. Top-level, since the hooks loader follows `$` only into functions
 * declared at the top of the file.
 */
async function compactHook(
  $: Parameters<Hook<'session.compact'>>[0],
  event: Parameters<Hook<'session.compact'>>[1],
  next: Parameters<Hook<'session.compact'>>[2],
  configured: HookConfig,
  once: { inertNoticeLogged: boolean },
  report: Report,
  pending: MidTurn[] = [],
): Promise<Awaited<ReturnType<Hook<'session.compact'>>>> {
  // A subagent's or a fork's own transcript is compacted like the main one;
  // only the context reading, which is the main conversation's, is left out.
  // `precompute` computes nothing yet and installs no result, so there is
  // nothing useful for Jev to do here.
  if (event.trigger === 'precompute') {
    return { skip: 'jev-compaction does not handle speculative compactions' };
  }
  try {
    const config = { ...configured, apiKey: await getApiKey($, configured) };
    // The last response's input tokens: the size the engine compacts at.
    // The history's share of it comes from the engine's own count of the
    // `Messages` row (Claude's tokenizer), not from the plugin's estimate,
    // which is calibrated to Jev's and can read half of Claude's count.
    let contextTokens: number | undefined;
    let messagesInContext: number | undefined;
    try {
      if (!event.agentId) {
        const usage = await $.session.usage({ breakdown: 'summary' });
        contextTokens = usage.context.tokens;
        messagesInContext = usage.context.breakdown?.categories.find(
          (row) => row.kind === 'used' && row.name === 'Messages',
        )?.tokens;
      }
    } catch {
      contextTokens = undefined;
    }
    // Messages the user sent mid-turn live only in the request that carried them
    // (see src/midturn.ts): put back any the history lacks before anything is cut.
    prunePresent(pending, event.messages);
    const { messages: history, restored } = restoreMidTurn(event.messages, pending);
    if (restored > 0) log($, `kept ${restored} message${restored > 1 ? 's' : ''} you sent mid-turn`, report);
    const historyTokens = messageTokens(history);
    const historyInContext = messagesInContext && messagesInContext > 0 ? messagesInContext : historyTokens;
    const target = messageTarget(contextTokens, historyInContext, configured.minReductionRatio);
    if (contextTokens !== undefined) {
      log(
        $,
        `context ${contextTokens} tokens, history ${messagesInContext ? `${messagesInContext} tokens (engine count)` : `~${historyTokens} tokens (estimate)`}, history cut needed ${percent(target)}`,
        report,
      );
    }
    // Unknown (a subagent, or an engine without the API form) counts as having images.
    let imagesInContext = true;
    let api: readonly { role: string; content: readonly unknown[] }[] | undefined;
    if (!event.agentId) try {
      api = await $.session.messages({ as: 'api' });
      const breakdown = contextBreakdown(api);
      imagesInContext = breakdown.images > 0;
      log($, `history as the model sees it: ${formatBreakdown(breakdown)}`, report);
    } catch {
      // Diagnostics only: an engine without the API form still compacts.
    }
    config.imagesInContext = imagesInContext;
    if (target >= 1 && target > configured.minReductionRatio) {
      const reason = `history is too small a share of the context to free ${percent(configured.minReductionRatio)}`;
      if (!mayUseBuiltin(event.trigger, config.builtinFallback)) {
        notify(
          $,
          'Not compacted: the history is too small a share of the context for Jev to free enough',
          `skipped (${reason}, builtin fallback disabled for ${event.trigger ?? 'this'} trigger)`,
          report,
        );
        return { skip: 'jev-compaction: below minimum reduction, builtin fallback disabled' };
      }
      notify(
        $,
        "Using Claude's summary: the history is too small a share of the context for Jev",
        `fallback to built-in summary (${reason})`,
        report,
      );
      return next(event);
    }
    config.minReductionRatio = target;
    if (event.instructions?.trim()) {
      const goal = config.goal || goalFromMessages(history);
      config.goal = `${goal}\n\nCompaction instructions: ${event.instructions}`;
    }
    if (config.notificationTrim && !config.archiveResults && !once.inertNoticeLogged) {
      once.inertNoticeLogged = true;
      log($, 'notification trim skipped: archiveResults is off', report);
    }
    const archive = config.archiveResults ? archiveWriter($, await archiveDir($), config.apiKey) : undefined;
    // Media leave with their rows' handles; without stripHandles the handles, and the media, stay.
    const media =
      api && (config.stripHandles ?? HOOK_DEFAULTS.stripHandles)
        ? placeMedia(history, findMedia(api), archive?.media)
        : { messages: history, images: 0, documents: 0, kept: 0 };
    if (media.images + media.documents + media.kept > 0) {
      log($, `media: ${media.images} images and ${media.documents} documents removed with a placeholder, ${media.kept} kept in the current prompt`, report);
    }
    const removedMedia = mediaNote(media.images, media.documents);
    const { result, messages, prose, notifications, injected, exchanges, steps } = await compactSession(
      media.messages,
      config,
      async (url, init) => {
        const response = await $.http.fetch(url, init);
        return { status: response.status, ok: response.ok, text: response.text, headers: response.headers };
      },
      archive,
      { signal: next.signal, sleep: clockSleep($), remainingMs: () => next.budget.remainingMs, baseUrl: await getBaseUrl($) },
    );
    if (next.signal.aborted) return interrupted($, report);
    for (const line of decisionLogLines(result)) log($, line, report);
    report.ledger = {
      messagesBefore: history.length,
      messagesAfter: messages.length,
      historyTokens,
      tokensAfter: messageTokens(messages),
      contextTokens,
      historyInContext,
      reduction: Math.round(reductionRatio(result) * 1000) / 1000,
      // Plugin marker rows left in the history, and what each stage that ran settled on.
      markersAfter: messages.filter((m) => hasMarker(m.text)).length,
      stages: {
        tool: Math.round(reductionRatio(result) * 1000) / 1000,
        ...(prose ? { prose: prose.cutoff ?? null } : {}),
        ...(notifications ? { notifications: notifications.rung ?? null } : {}),
        ...(injected ? { injected: injected.rung ?? null } : {}),
        ...(exchanges ? { exchanges: exchanges.cutoff ?? null, exchangesDropped: exchanges.stats.dropped } : {}),
        ...(steps ? { steps: steps.pass ?? null, stepsArchived: steps.stats.archived } : {}),
      },
      requests: result.stats.requests,
      failedRequests: result.stats.failedRequests,
      stateStage: result.stats.stateStage,
      model: config.model,
      options: { stateWording: config.stateWording ?? 'delete', resultChecks: config.resultChecks ?? false },
      // [tool, action, reason, keepCall, keepResult, archive file]
      calls: result.decisions.map((d) => [
        d.tool,
        d.action,
        d.reason,
        Math.round(d.keepCall * 100) / 100,
        Math.round(d.keepResult * 100) / 100,
        // Only a cut (dropped or truncated) result has its output in the archive for good.
        d.action === 'keep' ? '' : `${result.toolUseIds?.[d.id] ?? d.id}.txt`,
      ]),
    };
    if (reductionRatio(result) < config.minReductionRatio) {
      const tried = `${prose ? ', prose trim tried' : ''}${notifications ? ', notification trim tried' : ''}${injected ? ', injected-content fold tried' : ''}${exchanges ? ', exchange drop tried' : ''}${steps ? ', step trim tried' : ''}${
        target > configured.minReductionRatio ? `, history needs ${percent(target)} to free ${percent(configured.minReductionRatio)} of context` : ''
      }`;
      // The cut Jev did make: below the minimum, but where no summary may run
      // (or the summary fails) it still beats leaving the session as it is.
      const tokensAfter = messageTokens(messages);
      const partial = tokensAfter < historyTokens ? { messages, tokensBefore: historyTokens, tokensAfter } : undefined;
      // What the cut frees of the whole context, the share the minimum is about.
      const freed = percent(
        (1 - tokensAfter / Math.max(1, historyTokens)) *
          (contextTokens && contextTokens > historyInContext ? historyInContext / contextTokens : 1),
      );
      if (!mayUseBuiltin(event.trigger, config.builtinFallback)) {
        if (partial) {
          notify(
            $,
            `Compacted with Jev: ${event.messages.length} → ${messages.length} messages, frees ${freed} of the context (below the ${percent(configured.minReductionRatio)} target)${removedMedia ? `, ${removedMedia}` : ''}`,
            `kept the cut below the minimum (${percent(configured.minReductionRatio)} minimum${tried}, builtin fallback disabled for ${event.trigger ?? 'this'} trigger: ${summarize(result, prose, notifications, injected, exchanges, steps)})`,
            report,
          );
          return partial;
        }
        notify(
          $,
          'Not compacted: Jev found nothing it could remove',
          `skipped (below ${percent(configured.minReductionRatio)} minimum${tried}, builtin fallback disabled for ${event.trigger ?? 'this'} trigger: ${summarize(result)})`,
          report,
        );
        return { skip: 'jev-compaction: below minimum reduction, builtin fallback disabled' };
      }
      const summary = async (e: typeof event): ReturnType<typeof next> => {
        try {
          return await next(e);
        } catch (error) {
          if (next.signal.aborted || !partial) throw error;
          notify(
            $,
            `Claude's summary failed; kept Jev's cut (frees ${freed} of the context)`,
            `built-in summary failed, kept the Jev cut (${summarize(result)})`,
            report,
          );
          return partial;
        }
      };
      const share = Math.min(HYBRID_RECENT_SHARE, 1 - target - HYBRID_SUMMARY_ALLOWANCE);
      const split = config.hybridSummary && share > 0 ? hybridSplit(messages, historyTokens, share) : 0;
      if (split > 0) {
        // hybridOriginals: the summarizer reads the older history as it was,
        // not with truncation notes in place of the text.
        const original = config.hybridOriginals ? originalIndex(history, messages, split) : undefined;
        const older = original !== undefined ? history.slice(0, original) : messages.slice(0, split);
        if (config.hybridOriginals) log($, `hybrid: summarizing ${original !== undefined ? 'the original' : 'the trimmed (no matching prompt)'} older messages`, report);
        const summarized = await summary({ ...event, messages: older });
        if (summarized === partial) return partial;
        if (next.signal.aborted) return interrupted($, report);
        if (summarized.skip !== undefined) return summarized;
        const recent = messages.slice(split);
        log($, `hybrid: summarized ${split} older messages, kept ${recent.length} recent (~${messageTokens(recent)} tokens)`, report);
        notify(
          $,
          `Compacted: summarized ${split} older messages, kept ${recent.length} recent ones verbatim`,
          `summarized ${split} older messages, kept ${recent.length} recent (trim alone below ${percent(configured.minReductionRatio)} minimum${tried}: ${summarize(result)})`,
          report,
        );
        return { messages: [...summarized.messages, ...recent] };
      }
      notify(
        $,
        `Using Claude's summary: Jev's cut frees ${freed} of the context, below the ${percent(configured.minReductionRatio)} minimum`,
        `fallback to built-in summary (below ${percent(configured.minReductionRatio)} minimum${tried}: ${summarize(result)})`,
        report,
      );
      return summary(event);
    }
    const tokensAfter = messageTokens(messages);
    log($, `kept history ~${tokensAfter} tokens of ~${historyTokens}`, report);
    notify(
      $,
      `Compacted with Jev: ${event.messages.length} → ${messages.length} messages, ${
        contextTokens && contextTokens > historyInContext
          ? `frees ${percent((1 - tokensAfter / Math.max(1, historyTokens)) * (historyInContext / contextTokens))} of the context`
          : `${percent(1 - tokensAfter / Math.max(1, historyTokens))} smaller`
      }, no summary${removedMedia ? `, ${removedMedia}` : ''}`,
      `kept ${messages.length}/${event.messages.length} messages, no summary (${summarize(result, prose, notifications, injected, exchanges, steps)})`,
      report,
    );
    return { messages, tokensBefore: historyTokens, tokensAfter };
  } catch (error) {
    if (next.signal.aborted) return interrupted($, report);
    if (!mayUseBuiltin(event.trigger, configured.builtinFallback)) {
      notify(
        $,
        `Not compacted: Jev request failed (${shortError(error)})`,
        `skipped (builtin fallback disabled for ${event.trigger ?? 'this'} trigger: ${error instanceof Error ? error.message : String(error)})`,
        report,
      );
      return { skip: 'jev-compaction: request failed, builtin fallback disabled' };
    }
    notify(
      $,
      `Using Claude's summary: Jev request failed (${shortError(error)})`,
      `fallback to built-in summary (${error instanceof Error ? error.message : String(error)})`,
      report,
    );
    return next(event);
  }
}

export const register: Register = (on: On, options: PluginOptions) => {
  const configured = resolveHookConfig(options);
  const once = { inertNoticeLogged: false };
  let lastReport: { at: number; trigger: string; agentId?: string; lines: string[]; toast?: string } | undefined;

  // `/jev`: what the last compaction did and the settings it ran with, without
  // turning on the debug log. Registered at each (re)load of the plugin.
  on('session.start', async ($, e, next) => {
    try {
      await $.command.register({
        name: 'jev',
        description: 'jev-compaction: the last compaction and the current settings',
        immediate: true,
      });
    } catch {
      // A host without plugin commands still compacts.
    }
    return next(e);
  });

  on('command.run', { command: 'jev' }, async ($) => {
    const key = await getApiKey($, configured).catch(() => undefined);
    const baseUrl = await getBaseUrl($);
    const settings = `Settings: TypeSafe key ${key ? 'set' : 'missing (set TYPESAFE_API_KEY)'} · endpoint ${baseUrl ?? 'TypeSafe'} · model ${configured.model} · builtinFallback ${configured.builtinFallback} · minimum ${percent(configured.minReductionRatio)} of the context`;
    if (!lastReport) return { text: `No compaction yet in this session.\n${settings}` };
    const minutes = lastReport.at > 0 ? Math.max(0, Math.round(((await $.clock.now()) - lastReport.at) / 60_000)) : undefined;
    const lines = lastReport.lines.filter((line) => !line.startsWith('decisions'));
    return {
      text: [
        `Last compaction${minutes === undefined ? '' : minutes === 0 ? ' just now' : ` ${minutes} min ago`} (${lastReport.trigger}${lastReport.agentId ? ', subagent' : ''})`,
        lastReport.toast ?? '(no outcome recorded)',
        ...lines.map((line) => `  ${line}`),
        settings,
      ].join('\n'),
    };
  });

  // Records each message the user sends mid-turn, for the next compaction to keep
  // (see src/midturn.ts). Only reads: the attachment goes on unchanged.
  const midTurn = new Map<string, MidTurn[]>();
  const pendingFor = (agentId: string | undefined): MidTurn[] => {
    const key = agentId ?? 'main';
    if (!midTurn.has(key)) midTurn.set(key, []);
    return midTurn.get(key) as MidTurn[];
  };
  on('prompt.attachment', { type: 'queued_command' }, async ($, e, next) => {
    if (e.text.includes(MID_TURN_LEAD)) {
      try {
        const tail = await $.session.messages(e.agentId ? { agentId: e.agentId } : {});
        if (Array.isArray(tail)) recordMidTurn(pendingFor(e.agentId), e.text, tail);
      } catch {
        // A record only: a session that cannot be read leaves the message as the engine keeps it.
      }
    }
    return next(e);
  });

  on('session.compact', async ($, event, next) => {
    const current: Report = { lines: [] };
    if (event.trigger === 'precompute') return compactHook($, event, next, configured, once, current);
    try {
      return await compactHook($, event, next, configured, once, current, pendingFor(event.agentId));
    } finally {
      if (configured.compactionLog && configured.archiveResults && current.ledger) {
        try {
          const at = await $.clock.now();
          await appendLog($, `${await archiveDir($)}/log.jsonl`, [
            { time: new Date(at).toISOString(), trigger: event.trigger ?? 'unknown', subagent: event.agentId !== undefined, outcome: current.toast ?? '', ...current.ledger },
          ]);
        } catch {
          // The ledger is a record only; the compaction's result stands.
        }
      }
      // Only a record for /jev: a clock that fails must never replace the result.
      let at = 0;
      try {
        at = await $.clock.now();
      } catch {
        at = 0;
      }
      lastReport = {
        at,
        trigger: event.trigger ?? 'unknown',
        ...(event.agentId ? { agentId: event.agentId } : {}),
        ...current,
      };
    }
  });


  // Opt-in (see HOOK_DEFAULTS.jevPruneBashOutput): trims a Bash command's
  // stdout through Jev before the model ever sees it, the same idea as
  // compaction applied at tool.call time to one result. A short-circuit on
  // length runs before anything else, so a plugin left enabled costs nothing
  // on the ordinary, short-output case.
  // compactionLog: notes each archive file the assistant reads back (Read, Grep
  // or a shell command over the archive), so calibration can tell which tools'
  // dropped output turned out to be needed. Only these tools are hooked: every
  // hooked call pays a hop to the plugin's worker.
  on('tool.call', { tool: 'Read' }, async ($, e, next) => {
    const result = await next(e);
    await noteReadBack($, configured, e.tool, e as unknown as Record<string, unknown>);
    return result;
  });
  on('tool.call', { tool: 'Grep' }, async ($, e, next) => {
    const result = await next(e);
    await noteReadBack($, configured, e.tool, e as unknown as Record<string, unknown>);
    return result;
  });

  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    const result = await next(e);
    await noteReadBack($, configured, e.tool, e as unknown as Record<string, unknown>);
    if (!configured.jevPruneBashOutput || result.deny || !result.result) return result;
    const output = result.result as {
      stdout?: string;
      isImage?: boolean;
      backgroundTaskId?: string;
      persistedOutputPath?: string;
      rawOutputPath?: string;
    };
    // Never touch an image result, a still-running background command, or an
    // output the engine already moved to disk for being too large — trimming
    // those would be either meaningless or redundant.
    if (
      output.isImage ||
      output.backgroundTaskId ||
      output.persistedOutputPath ||
      output.rawOutputPath
    ) {
      return result;
    }
    const stdout = output.stdout ?? '';
    if (stdout.length < configured.jevPruneMinChars) return result;
    try {
      const apiKey = await getApiKey($, configured);
      if (!apiKey) return result;
      const asker = jevAsker(
        async (url, init) => {
          const response = await $.http.fetch(url, init);
          return { status: response.status, ok: response.ok, text: response.text, headers: response.headers };
        },
        apiKey,
        configured.model,
        { signal: next.signal, sleep: clockSleep($), remainingMs: () => next.budget.remainingMs, baseUrl: await getBaseUrl($) },
      );
      const trimmed = await trimOutput(
        { command: typeof e.command === 'string' ? e.command : '', goal: configured.goal ?? '', output: stdout },
        asker,
        { minChars: configured.jevPruneMinChars },
      );
      if (!trimmed.trimmed) return result;
      return { result: { ...output, stdout: trimmed.output }, context: result.context };
    } catch (error) {
      log($, `bash output trim skipped (${error instanceof Error ? error.message : String(error)})`);
      return result;
    }
  });
};

export { resolveOptions };
