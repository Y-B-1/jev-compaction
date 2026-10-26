import { headOf, messageChars, reductionRatio } from './compact.js';
import { goalIndices, isPromptRow } from './inject.js';
import { type NotificationOptions, askNoul } from './notifications.js';
import { PROSE_KEEP_LADDER } from './prose.js';
import { sha256Hex } from './sha256.js';
import { MARKER, isPinned, ownText, resultPreview } from './state.js';
import type { CompactResult, JevAsker, JevQuestions, Message, NoulQuestion } from './types.js';

/**
 * Whole-exchange drop: the last trim stage before the built-in summary. An exchange is a prompt
 * row and every row after it up to the next prompt. When Jev says an old exchange's work is no
 * longer needed, its work rows (tool calls, results, replies) are archived as one file and replaced
 * by one assistant row: a marker naming the file and the head of the exchange's last reply. The
 * prompt row stays verbatim. Tool calls and their results always leave together.
 */

const REPLY_HEAD_CHARS = 400;

export type Exchange = {
  key: string;
  /** Index of the prompt row. */
  prompt: number;
  /** Indices `[from, to)` of the work rows. */
  work: [number, number];
  tools: Map<string, number>;
};

export type ExchangeStats = {
  eligible: number;
  scored: number;
  dropped: number;
  charsSaved: number;
  requests: number;
  failedRequests: number;
  stateTokens: number;
  failed?: string;
};

export type ExchangeScores = {
  messages: readonly Message[];
  exchanges: Exchange[];
  noul: ReadonlyMap<string, number | undefined>;
  stats: ExchangeStats;
};

export type ExchangeApplied = {
  messages: Message[];
  archived: { id: string; text: string; path: string }[];
  stats: ExchangeStats;
};

/**
 * The exchanges that may be dropped: the prompt is not one of the goal's prompts, every row is
 * outside the pinned recent window, there are at least two work rows, no work row holds typed text
 * (`TYPED_WORK`), and every tool call in the
 * work rows has its result there too (and the other way round).
 */
/**
 * Text the user typed inside a work row: a `!` shell command, a slash command's non-empty
 * arguments, or a message sent while a turn was running (delivered as a system reminder).
 */
export const TYPED_WORK = /<bash-input>|<command-args>(?!\s*<\/command-args>)|The user sent a new message while you were working:/;

export function findExchanges(messages: readonly Message[], preserveRecentMessages: number): Exchange[] {
  const goal = goalIndices(messages);
  const prompts: number[] = [];
  messages.forEach((m, i) => {
    if (isPromptRow(m)) prompts.push(i);
  });
  const out: Exchange[] = [];
  prompts.forEach((p, n) => {
    const to = prompts[n + 1] ?? messages.length;
    const from = p + 1;
    if (goal.has(p) || to - from < 2) return;
    for (let i = p; i < to; i++) if (isPinned(i, messages.length, preserveRecentMessages)) return;
    const uses = new Set<string>();
    const results = new Set<string>();
    const tools = new Map<string, number>();
    for (let i = from; i < to; i++) {
      const m = messages[i] as Message;
      if (m.role === 'user' && [m.text, ...(m.toolResults ?? []).map((r) => r.text)].some((t) => TYPED_WORK.test(t))) return;
      for (const use of m.toolUses) {
        uses.add(use.tool_use_id);
        tools.set(use.tool, (tools.get(use.tool) ?? 0) + 1);
      }
      for (const result of m.toolResults ?? []) results.add(result.tool_use_id);
    }
    if (uses.size !== results.size || [...uses].some((id) => !results.has(id))) return;
    out.push({ key: `x${out.length + 1}`, prompt: p, work: [from, to], tools });
  });
  return out;
}

function lastReply(messages: readonly Message[], [from, to]: [number, number]): string {
  for (let i = to - 1; i >= from; i--) {
    const m = messages[i] as Message;
    const text = m.role === 'assistant' ? ownText(m.text) : '';
    if (text.length > 0) return text;
  }
  return '';
}

function toolList(tools: ReadonlyMap<string, number>): string {
  return [...tools].map(([name, n]) => (n > 1 ? `${name} x${n}` : name)).join(', ');
}

function question(e: Exchange, messages: readonly Message[], previewChars: number): JevQuestions[string] {
  const prompt = resultPreview((messages[e.prompt] as Message).text, previewChars);
  const reply = resultPreview(lastReply(messages, e.work), previewChars);
  const q: NoulQuestion = {
    type: 'noul',
    instructions: `The work in \`history\` entries ${e.work[0]} to ${e.work[1] - 1}, done for the request in entry ${e.prompt}, should stay in \`history\` in full: the assistant still needs its details. Request: ${prompt || '(none)'}. Tools: ${toolList(e.tools) || 'none'}. Last reply: ${reply || '(none)'}`,
    criteria: {
      true: 'The current work builds on these details: files, values, decisions or errors it still refers to, or an open thread it continues',
      false: 'The exchange is finished and the current work no longer refers to it: a closed side task, an earlier approach that was replaced, a question already answered',
    },
  };
  return q;
}

const emptyStats = (): ExchangeStats => ({
  eligible: 0,
  scored: 0,
  dropped: 0,
  charsSaved: 0,
  requests: 0,
  failedRequests: 0,
  stateTokens: 0,
});

/** Asks Jev once per eligible exchange whether its work must stay. Nothing is changed yet. */
export async function scoreExchanges(
  messages: readonly Message[],
  asker: JevAsker,
  options: NotificationOptions,
): Promise<ExchangeScores> {
  const stats = emptyStats();
  try {
    const exchanges = findExchanges(messages, options.preserveRecentMessages);
    stats.eligible = exchanges.length;
    if (exchanges.length === 0) return { messages, exchanges, noul: new Map(), stats };
    const questions = exchanges.map((e) => [e.key, question(e, messages, options.resultPreviewChars)] as const);
    const asked = await askNoul(messages, questions, asker, options);
    stats.requests = asked.requests;
    stats.failedRequests = asked.failedRequests;
    stats.stateTokens = asked.stateTokens;
    stats.scored = [...asked.noul.values()].filter((v) => v !== undefined).length;
    if (asked.failed !== undefined) return { messages, exchanges: [], noul: new Map(), stats: { ...stats, failed: asked.failed } };
    return { messages, exchanges, noul: asked.noul, stats };
  } catch (error) {
    return { messages, exchanges: [], noul: new Map(), stats: { ...stats, failed: error instanceof Error ? error.message : String(error) } };
  }
}

/** The work rows as one plain-text file: each row's role, text, tool calls with inputs, and results. */
export function renderExchange(messages: readonly Message[], e: Exchange): string {
  const parts: string[] = [`request (entry ${e.prompt}):\n${(messages[e.prompt] as Message).text}`];
  for (let i = e.work[0]; i < e.work[1]; i++) {
    const m = messages[i] as Message;
    if (m.text.length > 0) parts.push(`${m.role}:\n${m.text}`);
    for (const use of m.toolUses) parts.push(`tool call ${use.tool} (${use.tool_use_id}):\n${JSON.stringify(use.input, null, 2)}`);
    for (const result of m.toolResults ?? []) {
      parts.push(`tool result (${result.tool_use_id})${result.isError ? ' error' : ''}:\n${result.text}`);
    }
  }
  return parts.join('\n\n');
}

/**
 * Drops each exchange whose noul is below `keepAt`; an exchange with no answer stays. Before an
 * exchange changes, `cite` is given its rendered text and must return the archive path the marker
 * names; without a path the exchange stays. Untouched rows come back as the same objects.
 */
export function applyExchanges(
  scores: ExchangeScores,
  keepAt: number,
  cite: (id: string, text: string) => string | undefined,
): ExchangeApplied {
  const stats: ExchangeStats = { ...scores.stats, dropped: 0, charsSaved: 0 };
  const archived: ExchangeApplied['archived'] = [];
  const { messages } = scores;
  if (stats.failed !== undefined) return { messages: [...messages], archived, stats };
  // Dropped exchanges in a row (each prompt right after the previous one's work) share one
  // marker and one file: their prompts stay, one after another, and one row follows them.
  const runs: Exchange[][] = [];
  for (const e of scores.exchanges) {
    const n = scores.noul.get(e.key);
    if (n === undefined || n >= keepAt) continue;
    const run = runs.at(-1);
    if (run && (run.at(-1) as Exchange).work[1] === e.prompt) run.push(e);
    else runs.push([e]);
  }
  const replace = new Map<number, { to: number; row?: Message }>();
  for (const run of runs) {
    const text = run.map((e) => renderExchange(messages, e)).join('\n\n');
    const id = `exchange-${sha256Hex(text).slice(0, 12)}`;
    const path = cite(id, text);
    if (!path) continue;
    let chars = 0;
    let calls = 0;
    let rows = 0;
    const tools = new Map<string, number>();
    for (const e of run) {
      rows += e.work[1] - e.work[0];
      for (let i = e.work[0]; i < e.work[1]; i++) {
        chars += messageChars(messages[i] as Message);
        calls += (messages[i] as Message).toolUses.length;
      }
      for (const [tool, k] of e.tools) tools.set(tool, (tools.get(tool) ?? 0) + k);
    }
    const reply = lastReply(messages, (run.at(-1) as Exchange).work);
    const head = reply.length > REPLY_HEAD_CHARS ? `${headOf(reply, REPLY_HEAD_CHARS)}…` : reply;
    const what = run.length > 1 ? `the work of the ${run.length} exchanges above` : 'this exchange';
    const marker = `${MARKER}archived ${what}: ${rows} rows, ${calls} tool calls${calls > 0 ? ` (${toolList(tools)})` : ''}, ${chars} chars; full text: ${path} (Read it if needed)]`;
    const row: Message = { role: 'assistant', text: head ? `${marker}\nLast reply: ${head}` : marker, toolUses: [] };
    if (messageChars(row) >= chars) continue;
    run.forEach((e, k) => replace.set(e.work[0], { to: e.work[1], ...(k === run.length - 1 ? { row } : {}) }));
    archived.push({ id, text, path });
    stats.dropped += run.length;
    stats.charsSaved += chars - messageChars(row);
  }
  const out: Message[] = [];
  for (let i = 0; i < messages.length; i++) {
    const hit = replace.get(i);
    if (hit) {
      if (hit.row) out.push(hit.row);
      i = hit.to - 1;
    } else out.push(messages[i] as Message);
  }
  return { messages: out, archived, stats };
}

/**
 * Walks `PROSE_KEEP_LADDER` on one set of scores and takes the first cut-off whose result reaches
 * `minRatio` against the original `charsBefore` of `result`. Undefined when none does.
 */
export function exchangeLadder(
  scores: ExchangeScores,
  result: CompactResult,
  minRatio: number,
  pathOf: (id: string) => string,
): { cutoff: number; result: CompactResult; applied: ExchangeApplied } | undefined {
  for (const cutoff of PROSE_KEEP_LADDER) {
    const applied = applyExchanges(scores, cutoff, (id) => pathOf(id));
    const charsAfter = applied.messages.reduce((sum, m) => sum + messageChars(m), 0);
    const candidate = { ...result, messages: applied.messages, stats: { ...result.stats, charsAfter } };
    if (reductionRatio(candidate) >= minRatio) return { cutoff, result: candidate, applied };
  }
  return undefined;
}
