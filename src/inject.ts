import { type NotificationOptions, type Rung, askNoul, fold } from './notifications.js';
import { PROSE_KEEP_LADDER } from './prose.js';
import { sha256Hex } from './sha256.js';
import { type InjectKind, MARKER, goalText, hasMarker, injectedSpans, isPinned, isPromptRow, resultPreview } from './state.js';
import type { JevAsker, JevQuestions, Message, NoulQuestion } from './types.js';

export { type InjectKind, LONG_USER_CHARS, goalText, injectedSpans, isPromptRow } from './state.js';

/**
 * Injected-content trim: folds the large blocks the host puts into user rows — skill bodies,
 * browser element selections, pasted content and other long user rows — to a head, a marker that
 * names the archived original, and a salvage block. Text the user typed around a block stays
 * verbatim, and the last three typed prompts (the goal) are never touched. Two parts, like
 * notification trim: `scoreInjected` asks Jev once, offline `applyInjected` applies a rung.
 */

const FOLD_MARGIN = 120;
/** At the `all` rung a `long` or `paste` block — likely typed or chosen by the user — still needs Jev's go-ahead at the last prose cut-off. */
const USER_ALL_CUTOFF = PROSE_KEEP_LADDER[PROSE_KEEP_LADDER.length - 1] as number;

/** Indices of the last three prompt rows, the ones the goal is read from. */
export function goalIndices(messages: readonly Message[]): Set<number> {
  const out: number[] = [];
  for (let i = messages.length - 1; i >= 0 && out.length < 3; i--) {
    const message = messages[i] as Message;
    if (isPromptRow(message) && goalText(message.text).trim().length > 0) out.push(i);
  }
  return new Set(out);
}

type Block = { index: number; key: string; kind: InjectKind; span: [number, number] };

export type InjectStats = {
  /** Blocks the stage may fold. */
  eligible: number;
  /** Blocks Jev gave a valid answer for. */
  scored: number;
  /** Blocks folded at this rung. */
  folded: number;
  charsSaved: number;
  requests: number;
  failedRequests: number;
  stateTokens: number;
  failed?: string;
};

/** Jev's scores for the injected blocks of one message list. Reapply them at any rung with `applyInjected`. */
export type InjectScores = {
  messages: readonly Message[];
  blocks: readonly Block[];
  noul: ReadonlyMap<string, number | undefined>;
  headChars: number;
  salvageChars: number;
  stats: InjectStats;
};

export type InjectApplied = {
  messages: Message[];
  /** Original row to its rewritten copy, for composing with another stage by identity. */
  replaced: ReadonlyMap<Message, Message>;
  /** The originals this rung cites; hand them to the real archive writer once the rung is chosen. */
  archived: { id: string; text: string; path: string }[];
  stats: InjectStats;
};

/** Finds the blocks: unpinned prompt rows outside the goal, holding no marker of ours, with a span worth folding. */
export function findBlocks(
  messages: readonly Message[],
  preserveRecentMessages: number,
  headChars: number,
  imagesInContext = false,
): Block[] {
  const goal = goalIndices(messages);
  const blocks: Block[] = [];
  messages.forEach((message, index) => {
    if (!isPromptRow(message) || goal.has(index) || isPinned(index, messages.length, preserveRecentMessages)) return;
    if (hasMarker(message.text)) return;
    const engineHeld = (message as { handle?: string }).handle !== undefined;
    for (const { kind, span } of injectedSpans(message.text)) {
      if (span[1] - span[0] <= headChars + FOLD_MARGIN) continue;
      if (imagesInContext && engineHeld && (kind === 'long' || kind === 'paste')) continue;
      blocks.push({ index, key: `i${blocks.length + 1}`, kind, span });
    }
  });
  return blocks;
}

const KIND_WORDS: Record<InjectKind, string> = {
  skill: 'skill instructions loaded into the conversation',
  element: 'page element the user selected in the browser (its markup and styles)',
  paste: 'content the user pasted',
  long: 'long user message',
};

function question(block: Block, text: string, previewChars: number): JevQuestions[string] {
  const body = text.slice(block.span[0], block.span[1]);
  const preview = resultPreview(body, previewChars);
  const q: NoulQuestion = {
    type: 'noul',
    instructions: `The ${KIND_WORDS[block.kind]} in \`history\` entry ${block.index} (${body.length} chars) should stay in \`history\` verbatim: the assistant still needs its exact contents${
      preview ? `. Preview: ${preview}` : ''
    }`,
    criteria: {
      true: 'The exact contents are still in use: instructions the assistant is still following, markup or text the current work refers to, data that cannot be read again',
      false:
        'The contents are stale or already acted on: instructions for a finished task, an element from an earlier design pass, pasted text the assistant already used or restated',
    },
  };
  return q;
}

const emptyStats = (): InjectStats => ({
  eligible: 0,
  scored: 0,
  folded: 0,
  charsSaved: 0,
  requests: 0,
  failedRequests: 0,
  stateTokens: 0,
});

/** Asks Jev, once per block, whether it must stay verbatim. Nothing is changed yet; a total failure sets `stats.failed`. */
export async function scoreInjected(
  messages: readonly Message[],
  asker: JevAsker,
  options: NotificationOptions,
): Promise<InjectScores> {
  const stats = emptyStats();
  const base = { messages, headChars: options.truncateHeadChars, salvageChars: options.salvageChars };
  try {
    const blocks = findBlocks(messages, options.preserveRecentMessages, options.truncateHeadChars, options.imagesInContext);
    stats.eligible = blocks.length;
    if (blocks.length === 0) return { ...base, blocks, noul: new Map(), stats };
    const questions = blocks.map((b) => [b.key, question(b, (messages[b.index] as Message).text, options.resultPreviewChars)] as const);
    const asked = await askNoul(messages, questions, asker, options);
    stats.requests = asked.requests;
    stats.failedRequests = asked.failedRequests;
    stats.stateTokens = asked.stateTokens;
    stats.scored = [...asked.noul.values()].filter((v) => v !== undefined).length;
    if (asked.failed !== undefined) return { ...base, blocks: [], noul: new Map(), stats: { ...stats, failed: asked.failed } };
    return { ...base, blocks, noul: asked.noul, stats };
  } catch (error) {
    return { ...base, blocks: [], noul: new Map(), stats: { ...stats, failed: error instanceof Error ? error.message : String(error) } };
  }
}

const WHERE = (path: string): string => `full text: ${path} (Read or grep it if needed)]`;

/**
 * Applies `scores` at a rung: a block whose noul is below the rung is folded. At `all`, every
 * `skill` and `element` block folds, while `long` and `paste` blocks still need a noul below the
 * last prose cut-off. Below `all`, a missing answer keeps the block. Before a row changes, `cite` is given its
 * whole original and must return the path the markers will name; without a path the row stays.
 */
export function applyInjected(
  scores: InjectScores,
  rung: Rung,
  cite?: (id: string, text: string) => string | undefined,
): InjectApplied {
  const stats: InjectStats = { ...scores.stats, folded: 0, charsSaved: 0 };
  const replaced = new Map<Message, Message>();
  const archived: InjectApplied['archived'] = [];
  if (stats.failed !== undefined || !cite) return { messages: [...scores.messages], replaced, archived, stats };
  const byRow = new Map<number, Block[]>();
  const folds = (block: Block): boolean => {
    const n = scores.noul.get(block.key);
    const userish = block.kind === 'long' || block.kind === 'paste';
    if (rung !== 'all') return n !== undefined && n < rung;
    if (!userish) return true;
    return n !== undefined && n < USER_ALL_CUTOFF;
  };
  for (const block of scores.blocks) {
    if (!folds(block)) continue;
    byRow.set(block.index, [...(byRow.get(block.index) ?? []), block]);
  }
  for (const [index, blocks] of byRow) {
    const message = scores.messages[index] as Message;
    const id = `inject-${sha256Hex(message.text).slice(0, 12)}`;
    const path = cite(id, message.text);
    if (!path) continue;
    let text = message.text;
    let folded = 0;
    for (const block of [...blocks].sort((a, b) => b.span[0] - a.span[0])) {
      const body = text.slice(block.span[0], block.span[1]);
      const out = fold(body, scores.headChars, scores.salvageChars, (n) => `${MARKER}folded ${n} chars of this ${block.kind === 'element' ? 'element selection' : block.kind === 'long' ? 'message' : block.kind}; ${WHERE(path)}`);
      if (out.length >= body.length) continue;
      text = text.slice(0, block.span[0]) + out + text.slice(block.span[1]);
      folded++;
    }
    if (folded === 0) continue;
    replaced.set(message, { ...message, text });
    archived.push({ id, text: message.text, path });
    stats.folded += folded;
    stats.charsSaved += message.text.length - text.length;
  }
  return { messages: scores.messages.map((m) => replaced.get(m) ?? m), replaced, archived, stats };
}
