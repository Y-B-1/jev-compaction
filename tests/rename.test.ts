import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { archiveFilesNamed, register } from '../hooks/fast-jev.js';
import { wasCapped, wasTruncated } from '../src/compact.js';
import { MARKER, hasMarker, ownText } from '../src/state.js';

const json = (path: string) => JSON.parse(readFileSync(new URL(`../${path}`, import.meta.url), 'utf8'));

describe('the jev-compaction name', () => {
  it('names the plugin, its marketplace and its package jev-compaction', () => {
    expect(json('.claude-plugin/plugin.json').name).toBe('jev-compaction');
    const market = json('.claude-plugin/marketplace.json');
    expect(market.name).toBe('jev-compaction');
    expect(market.plugins.map((p: { name: string }) => p.name)).toEqual(['jev-compaction']);
    expect(json('package.json').name).toBe('jev-compaction');
  });

  it('writes new notes under the new name', () => {
    expect(MARKER).toBe('[jev-compaction ');
  });

  it('still recognizes notes written under the old name', () => {
    for (const name of ['jev-compaction', 'fast-jev-compaction']) {
      expect(wasTruncated(`head\n[${name} truncated 120 chars]`)).toBe(true);
      expect(wasCapped(`head\n[${name} kept the head and tail of 4000 chars, 2000 omitted]\ntail`)).toBe(true);
      expect(hasMarker(`[${name} archived 2 rows of earlier work; full text: x.txt (Read it if needed)]`)).toBe(true);
      expect(ownText(`[${name} archived 2 rows of earlier work]\nNote: old\nThe real reply.`)).toBe('The real reply.');
    }
    expect(hasMarker('A reply that names jev-compaction in passing.')).toBe(false);
  });

  it('counts archive files read back from the new and the old folder', () => {
    expect(archiveFilesNamed({ file_path: '/w/.claude/jev-compaction/step-abc.txt' })).toEqual(['step-abc.txt']);
    expect(archiveFilesNamed({ file_path: '/w/.claude/fast-jev-compaction/step-def.txt' })).toEqual(['step-def.txt']);
    expect(archiveFilesNamed({ file_path: '/w/src/app.ts' })).toEqual([]);
  });

  it('archives into .claude/jev-compaction', async () => {
    const hooks: { event: string; hook: (...args: any[]) => Promise<any> }[] = [];
    (register as any)(
      (event: string, a: unknown, b?: unknown) => hooks.push({ event, hook: (typeof b === 'function' ? b : a) as any }),
      { apiKey: 'k', preserveRecentMessages: 2, minReductionRatio: 0 },
    );
    const files = new Map<string, string>();
    const big = 'output line\n'.repeat(300);
    const $ = {
      clock: { sleep: () => new Promise(() => {}), now: async () => Date.UTC(2026, 9, 26) },
      http: {
        fetch: async (_url: string, init: { body: string }) => {
          const { questions } = JSON.parse(init.body) as { questions: Record<string, unknown> };
          const answers = Object.fromEntries(Object.keys(questions).map((k) => [k, { type: 'noul', noul: k.startsWith('result_') ? 0.01 : 0.9 }]));
          return { status: 200, ok: true, text: JSON.stringify({ answers }) };
        },
      },
      ui: { log: () => {}, toast: () => {} },
      fs: {
        exists: async (p: string) => files.has(p),
        write: async (p: string, t: string) => void files.set(p, t),
        read: async (p: string) => {
          if (!files.has(p)) throw new Error('missing');
          return files.get(p)!;
        },
      },
      session: { cwd: async () => '/w', root: async () => '/w' },
      process: { run: async () => ({ exitCode: 1, stdout: '' }) },
    };
    const next = Object.assign(async (e: unknown) => e, { signal: new AbortController().signal, budget: { ms: 1e4, remainingMs: 1e4 } });
    const messages = [
      { role: 'user', text: 'Fix it.', toolUses: [] },
      { role: 'assistant', text: '', toolUses: [{ tool_use_id: 'toolu_A', tool: 'Bash', input: { command: 'make' }, text: big }] },
      { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'toolu_A', text: big, isError: false }] },
      { role: 'assistant', text: 'ok', toolUses: [] },
      { role: 'user', text: 'go', toolUses: [] },
    ];
    const compact = hooks.find((h) => h.event === 'session.compact')!.hook;
    const out = await compact($, { trigger: 'manual', messages }, next);
    expect(files.has('/w/.claude/jev-compaction/toolu_A.txt')).toBe(true);
    expect([...files.keys()].some((p) => p.includes('fast-jev-compaction'))).toBe(false);
    expect(JSON.stringify(out.messages)).toContain('[jev-compaction truncated');
  });
});
