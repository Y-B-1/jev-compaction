// Suggests per-tool keep thresholds from the plugin's own record (compactionLog): each compaction's per-call
// scores and decisions, and each archive file the assistant later read back. A dropped output that was read
// back was needed after all; a tool whose drops are often read back is cut too eagerly at its threshold.
// `npx tsx eval/calibrate.ts [log.jsonl ...]` (default: ./.claude/jev-compaction/log.jsonl). Prints a
// table and a `keepThresholdByTool` value; reads only the log, never transcript text.
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

/** [tool, action, reason, keepCall, keepResult, archive file] as the hook writes it. */
type Call = [string, string, string, number, number, string];
type Entry = { time?: string; calls?: Call[]; readBack?: string[] };

export interface ToolRow {
  tool: string;
  /** Results Jev scored and the plugin cut (dropped or truncated), each with its archive file. */
  dropped: number;
  /** Of those, read back later. */
  readBack: number;
  /** The highest keepResult among the drops that were read back. */
  maxReadBackScore: number;
  /** The threshold that would have kept every read-back drop, or undefined when too few drops or none read back. */
  suggested?: number;
}

/** Fewer drops than this per tool give no suggestion. */
export const MIN_DROPS = 10;
/** Suggestions never go above this: past it Jev's "unsure" answers would all stay. */
export const MAX_THRESHOLD = 0.45;

export function calibrate(lines: readonly string[], current = 0.15): { rows: ToolRow[]; unattributed: number; option: string } {
  const entries: Entry[] = [];
  for (const line of lines) {
    try {
      if (line.trim()) entries.push(JSON.parse(line) as Entry);
    } catch {
      // A torn line from a concurrent write is skipped.
    }
  }
  const drops = new Map<string, { tool: string; score: number; at: number }>();
  const readAt = new Map<string, number>();
  let unattributed = 0;
  entries.forEach((entry, at) => {
    for (const [tool, action, reason, , keepResult, file] of entry.calls ?? []) {
      if (file && (action !== 'keep' || reason !== 'kept') && reason !== 'duplicate' && reason !== 'compacted') {
        drops.set(file, { tool, score: keepResult, at });
      }
    }
    for (const file of entry.readBack ?? []) {
      if (file === '*') unattributed += 1;
      else if (!readAt.has(file)) readAt.set(file, at);
    }
  });
  const byTool = new Map<string, ToolRow>();
  for (const [file, drop] of drops) {
    const row = byTool.get(drop.tool) ?? { tool: drop.tool, dropped: 0, readBack: 0, maxReadBackScore: 0 };
    row.dropped += 1;
    const read = readAt.get(file);
    if (read !== undefined && read > drop.at) {
      row.readBack += 1;
      row.maxReadBackScore = Math.max(row.maxReadBackScore, drop.score);
    }
    byTool.set(drop.tool, row);
  }
  const rows = [...byTool.values()].sort((a, b) => b.dropped - a.dropped);
  for (const row of rows) {
    if (row.dropped < MIN_DROPS || row.readBack === 0) continue;
    const needed = Math.ceil((row.maxReadBackScore + 0.0001) * 20) / 20;
    if (needed > current) row.suggested = Math.min(MAX_THRESHOLD, needed);
  }
  const option = rows
    .filter((row) => row.suggested !== undefined)
    .map((row) => `${row.tool}=${row.suggested}`)
    .join(', ');
  return { rows, unattributed, option };
}

export function render(result: ReturnType<typeof calibrate>): string {
  const pct = (n: number, d: number) => (d === 0 ? '-' : `${Math.round((n / d) * 100)}%`);
  return [
    '| tool | cut | read back | rate | highest score read back | suggested threshold |',
    '| --- | --- | --- | --- | --- | --- |',
    ...result.rows.map(
      (r) => `| ${r.tool} | ${r.dropped} | ${r.readBack} | ${pct(r.readBack, r.dropped)} | ${r.readBack ? r.maxReadBackScore : '-'} | ${r.suggested ?? (r.dropped < MIN_DROPS ? 'too few cuts' : 'keep current')} |`,
    ),
    '',
    `Reads of the archive that name no file (a grep over the folder): ${result.unattributed}.`,
    result.option ? `keepThresholdByTool: ${result.option}` : 'No change suggested.',
  ].join('\n');
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const paths = process.argv.slice(2);
  const files = paths.length > 0 ? paths : ['.claude/jev-compaction/log.jsonl'];
  const lines = files.filter((p) => existsSync(p)).flatMap((p) => readFileSync(p, 'utf8').split('\n'));
  if (lines.length === 0) {
    console.log(`no log found (${files.join(', ')}); turn on compactionLog and archiveResults, then compact a few sessions`);
    process.exitCode = 1;
  } else console.log(render(calibrate(lines)));
}
