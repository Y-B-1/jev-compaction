import { describe, expect, it } from 'vitest';
import { compactSession } from '../hooks/fast-jev.ts';
import type { SessionMessage } from '../src/types.js';

const message = (role: 'user' | 'assistant', text: string, extra: Partial<SessionMessage> = {}): SessionMessage => ({
  role,
  text,
  toolUses: [],
  ...extra,
});
const call = (id: string, tool: string, input: Record<string, unknown>, text: string): SessionMessage =>
  message('assistant', '', { toolUses: [{ tool_use_id: id, tool, input, text }], handle: `h-${id}` });
const result = (id: string, text: string): SessionMessage =>
  message('user', '', { toolResults: [{ tool_use_id: id, text, isError: false }], handle: `r-${id}` });

const BIG = 'export const value = 1;\n'.repeat(2000); // 48,000 chars, dropped by tool compaction
const FLUFF = 'Let me take a careful look at how this module is laid out before changing anything.';
const KEPT = 'The cache key lives in src/cache/key.ts and must not change.';
const REPLY = `${Array(20).fill(FLUFF).join(' ')} ${KEPT}`;

function transcript(big = BIG): SessionMessage[] {
  return [
    message('user', 'Fix the stale cache.', { handle: 'u-0' }),
    call('rd1', 'Read', { file_path: 'src/big.ts' }, big),
    result('rd1', big),
    message('assistant', REPLY, { handle: 'a-1' }),
    message('user', 'first recent prompt', { handle: 'u-2' }),
    message('assistant', 'ok', { handle: 'a-3' }),
    message('user', 'second recent prompt', { handle: 'u-4' }),
    message('assistant', 'ok', { handle: 'a-5' }),
  ];
}

const isStep = (key: string) => /^s\d+$/.test(key);
function fetchWith(step: number | 'fail', requests: string[][] = []) {
  return async (_url: string, init?: { body?: string }) => {
    const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
    const keys = Object.keys(questions);
    requests.push(keys);
    if (step === 'fail' && keys.some(isStep)) return { status: 500, ok: false, text: 'down' };
    const answers = Object.fromEntries(keys.map((k) => [k, { type: 'noul', noul: isStep(k) ? step : 0.01 }]));
    return { status: 200, ok: true, text: JSON.stringify({ answers }) };
  };
}
function archive() {
  const files = new Map<string, string>();
  const pathOf = (id: string) => `arch/${id}.txt`;
  return { files, pathOf, cite: (id: string, text: string) => (files.set(id, text), pathOf(id)), flush: async () => {} };
}
const config = (extra: Record<string, unknown> = {}) => ({
  apiKey: 'k',
  preserveRecentMessages: 2,
  minReductionRatio: 0.25,
  archiveResults: true,
  proseTrim: true,
  ...extra,
});
const replyOf = (messages: readonly SessionMessage[]) => messages.find((m) => m.role === 'assistant' && m.text.includes(KEPT));
const stepRequests = (requests: string[][]) => requests.filter((keys) => keys.some(isStep));

describe('replyTrim: old replies folded whole on every compaction', () => {
  it('folds an old reply Jev scores done into one archive pointer after tool compaction met the minimum', async () => {
    const a = archive();
    const requests: string[][] = [];
    // A smaller file, so the reply is more than MILD_MIN_SHARE of the context.
    const out = await compactSession(transcript(BIG.slice(0, 8_000)), config() as never, fetchWith(0.01, requests), a);
    expect(replyOf(out.messages)).toBeUndefined();
    // The dropped Read call's note row folds into the same marker as the reply beside it.
    const marker = out.messages.find((m) => m.text.includes('archived 2 rows of earlier work'))!;
    expect(out.messages.filter((m) => m.text.includes('[jev-compaction '))).toHaveLength(1);
    expect(marker.text).toMatch(/full text: arch\/step-[0-9a-f]{12}\.txt \(Read it if needed\)\]\nNote: Let me take/);
    expect([...a.files.values()].some((t) => t.includes(REPLY))).toBe(true);
    expect([...a.files.values()].some((t) => t.includes(REPLY) && t.includes('removed the Read call'))).toBe(true);
    expect(out.steps).toMatchObject({ pass: 'trim', cutoff: 0.05, stats: { archived: 1 } });
    expect(stepRequests(requests)).toHaveLength(1);
    // The prompts and the recent rows are never scored or changed.
    expect(out.messages.slice(-4).map((m) => m.text)).toEqual(['first recent prompt', 'ok', 'second recent prompt', 'ok']);
    expect(out.messages[0]!.text).toBe('Fix the stale cache.');
  });

  it('keeps a reply Jev scores at or above the cut-off', async () => {
    const out = await compactSession(transcript(), config() as never, fetchWith(0.05), archive());
    expect(replyOf(out.messages)!.text).toContain(REPLY);
    expect(out.steps?.pass).toBeUndefined();
  });

  it('changes nothing and asks nothing when replyTrim is off or there is no archive', async () => {
    for (const [extra, withArchive] of [
      [{ replyTrim: false }, true],
      [{ archiveResults: false }, false],
    ] as const) {
      const requests: string[][] = [];
      const out = await compactSession(transcript(), config(extra) as never, fetchWith(0.01, requests), withArchive ? archive() : undefined);
      expect(replyOf(out.messages)!.text).toContain(REPLY);
      expect(stepRequests(requests)).toHaveLength(0);
    }
  });

  it('keeps tool compaction’s result when the step request fails', async () => {
    const a = archive();
    const out = await compactSession(transcript(), config() as never, fetchWith('fail'), a);
    expect(replyOf(out.messages)!.text).toContain(REPLY);
    expect(out.steps?.stats.failed).toBeDefined();
    expect([...a.files.keys()].some((id) => id.includes('step-'))).toBe(false);
  });
});

describe('replyTrim floor', () => {
  it('leaves an old reply alone when folding it frees less than 5% of the context', async () => {
    const out = await compactSession(transcript(), config() as never, fetchWith(0.01), archive());
    expect(replyOf(out.messages)!.text).toContain(REPLY);
    expect(out.messages.some((m) => m.text.includes('[jev-compaction archived'))).toBe(false);
  });
});
