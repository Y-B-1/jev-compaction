import type { Message } from './types.js';

/**
 * Messages the user sends while Claude works. Claude Code delivers each one as a
 * `queued_command` attachment that only the next model request carries: no user or
 * assistant message holds it, so a compaction that hands back its own message list
 * loses it, and the model no longer knows it was said. The hook records each one with
 * the tool results it arrived after, and before compacting puts it back, word for
 * word, as text on that result row, where the request had it.
 */
export const MID_TURN_LEAD = 'The user sent a new message while you were working:';

/** How many recorded messages a session keeps; the oldest go first. */
const MAX_PENDING = 50;

/** A mid-turn message as the model read it, and the `tool_use_id`s of the result row it arrived after. */
export type MidTurn = { text: string; after: readonly string[] };

type Row = Message & { handle?: string };

const isResultRow = (m: Row) => m.role === 'user' && (m.toolResults?.length ?? 0) > 0;

/** The words the user typed, from an attachment's text; '' when it is not a mid-turn message. */
export function midTurnBody(text: string): string {
  const at = text.indexOf(MID_TURN_LEAD);
  if (at === -1) return '';
  const rest = text.slice(at + MID_TURN_LEAD.length);
  const end = rest.indexOf('\n\nThis is how Claude Code surfaces');
  return (end === -1 ? rest : rest.slice(0, end)).trim();
}

/** Adds one mid-turn message to `pending`, anchored to the newest result row of `tail` (the session when it arrived). */
export function recordMidTurn(pending: MidTurn[], text: string, tail: readonly Row[]): void {
  if (!midTurnBody(text) || pending.some((p) => p.text === text)) return;
  const last = [...tail].reverse().find(isResultRow);
  pending.push({ text, after: last ? (last.toolResults ?? []).map((r) => r.tool_use_id) : [] });
  if (pending.length > MAX_PENDING) pending.splice(0, pending.length - MAX_PENDING);
}

/**
 * `messages` with each pending message that none of them holds put back as text on the
 * last result row of the group it arrived after, or on the newest result row when that
 * group is gone; never on a prompt the user typed. The changed row goes back without its
 * handle, so the engine saves it from its text and tool blocks.
 */
export function restoreMidTurn<M extends Row>(messages: readonly M[], pending: readonly MidTurn[]): { messages: M[]; restored: number } {
  const out = [...messages];
  let restored = 0;
  for (const p of pending) {
    const body = midTurnBody(p.text);
    if (!body || out.some((m) => m.text.includes(body))) continue;
    let at = out.findIndex((m) => isResultRow(m) && (m.toolResults ?? []).some((r) => p.after.includes(r.tool_use_id)));
    if (at !== -1) while (at + 1 < out.length && isResultRow(out[at + 1] as M)) at += 1;
    else for (let i = out.length - 1; i >= 0 && at === -1; i--) if (isResultRow(out[i] as M)) at = i;
    if (at === -1) continue;
    const { handle: _handle, ...row } = out[at] as M;
    out[at] = { ...row, text: row.text ? `${row.text}\n\n${p.text}` : p.text } as M;
    restored += 1;
  }
  return { messages: out, restored };
}

/** Drops from `pending` each message `messages` already holds. */
export function prunePresent(pending: MidTurn[], messages: readonly Row[]): void {
  for (let i = pending.length - 1; i >= 0; i--) {
    const body = midTurnBody((pending[i] as MidTurn).text);
    if (!body || messages.some((m) => m.text.includes(body))) pending.splice(i, 1);
  }
}
