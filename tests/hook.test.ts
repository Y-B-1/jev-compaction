import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  archiveDir,
  archiveWriter,
  compactSession,
  decisionLog,
  decisionLogLines,
  getApiKey,
  jevAsker,
  messageTarget,
  messageTokens,
  register,
  resolveHookConfig as resolveHookConfigLib,
  resumeSafeMessages,
  summarize,
  toSessionMessages,
} from '../hooks/fast-jev.ts';
import { sha256Hex } from '../src/sha256.js';
import { applyDecisions, collectToolCalls, decideCall, type Message } from '../src/index.js';

type SessionMessage = Message & { handle?: string };

// These fixtures use short results and check handles directly; they predate
// the small-result floor, duplicate removal, the keep-signal guard and the
// resume-safe handle policy, which have their own tests.
const LEGACY = { minResultChars: 0, dedupeRepeats: false, minKeepSignalCalls: 0, stripHandles: false };
const resolveHookConfig = (options: Record<string, unknown>) =>
  resolveHookConfigLib({ ...LEGACY, ...options });

/** The engine's `$.clock`: a sleep that never elapses, only aborts. */
const NOTICE = /^(fallback to built-in summary|skipped \(|kept \d+\/\d+ messages|summarized \d+ older|built-in summary failed|kept the cut)/;
const clock = {
  calls: 0,
  sleep: (_ms: number, options?: { signal?: AbortSignal }) =>
    ++clock.calls &&
    new Promise<void>((_resolve, reject) =>
      options?.signal?.addEventListener('abort', () => reject(options.signal?.reason), { once: true }),
    ),
};

function message(role: Message['role'], text: string, extra: Partial<SessionMessage> = {}): SessionMessage {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): SessionMessage {
  return message('assistant', '', {
    toolUses: [{ tool_use_id: id, tool, input, text }],
    handle: `h-${id}`,
  });
}

function result(id: string, text: string, isError = false): SessionMessage {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }], handle: `r-${id}` });
}

const fileA = 'export const a = 1;\n'.repeat(50);

function transcript(): SessionMessage[] {
  return [
    message('user', 'Fix the failing test.', { handle: 'h-0' }),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    call('tool-2', 'Bash', { command: 'npm test' }, 'FAIL'),
    result('tool-2', 'FAIL b.test.ts: expected 2 to be 3', true),
    message('assistant', 'Fixing now.', { handle: 'h-5' }),
    message('user', 'go ahead', { handle: 'h-6' }),
  ];
}

function jevFetch(answer: (name: string) => number, bodies: string[] = []) {
  return async (_url: string, init?: { body?: string }) => {
    bodies.push(init?.body ?? '');
    const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
    const answers = Object.fromEntries(
      Object.keys(questions).map((key) => [key, { type: 'noul', noul: answer(key) }]),
    );
    return { status: 200, ok: true, text: JSON.stringify({ answers }) };
  };
}

describe('hook config', () => {
  it('reads userConfig values and falls back to defaults', () => {
    expect(resolveHookConfigLib({})).toEqual({
      archiveResults: true,
      minReductionRatio: 0.25,
      model: 'jev-latest',
      builtinFallback: 'auto',
      proseTrim: true,
      jevPruneBashOutput: false,
      jevPruneMinChars: 4_000,
      stripHandles: true,
      inputTrim: true,
      hybridSummary: true,
      hybridOriginals: false,
      compactionLog: true,
      injectTrim: true,
      replyTrim: true,
      exchangeDrop: true,
    });
    expect(
      resolveHookConfigLib({ apiKey: 'k', keepThreshold: 0.3, maxStateTokens: 1000, model: 'jev-x', goal: 'g' }),
    ).toEqual({
      apiKey: 'k',
      keepThreshold: 0.3,
      maxStateTokens: 1000,
      model: 'jev-x',
      goal: 'g',
      archiveResults: true,
      minReductionRatio: 0.25,
      builtinFallback: 'auto',
      proseTrim: true,
      jevPruneBashOutput: false,
      jevPruneMinChars: 4_000,
      stripHandles: true,
      inputTrim: true,
      hybridSummary: true,
      hybridOriginals: false,
      compactionLog: true,
      injectTrim: true,
      replyTrim: true,
      exchangeDrop: true,
    });
    expect(resolveHookConfig({ injectTrim: false, replyTrim: false, exchangeDrop: 'no' })).toMatchObject({
      injectTrim: false,
      replyTrim: false,
      exchangeDrop: true,
    });
  });

  it('reads jevPruneBashOutput from userConfig', () => {
    expect(resolveHookConfig({}).jevPruneBashOutput).toBe(false);
    expect(resolveHookConfig({ jevPruneBashOutput: true }).jevPruneBashOutput).toBe(true);
    expect(resolveHookConfig({ jevPruneBashOutput: 'yes' }).jevPruneBashOutput).toBe(false);
    expect(resolveHookConfig({ jevPruneMinChars: 500 }).jevPruneMinChars).toBe(500);
  });

  it('reads builtinFallback from userConfig, ignoring an invalid value', () => {
    expect(resolveHookConfig({ builtinFallback: 'always' }).builtinFallback).toBe('always');
    expect(resolveHookConfig({ builtinFallback: 'never' }).builtinFallback).toBe('never');
    expect(resolveHookConfig({ builtinFallback: 'sometimes' }).builtinFallback).toBe('auto');
  });

  it('reads inputTrim and maxKeptResultChars from userConfig', () => {
    expect(resolveHookConfig({ inputTrim: false }).inputTrim).toBe(false);
    expect(resolveHookConfig({ inputTrim: 'no' }).inputTrim).toBe(true);
    expect(resolveHookConfig({ maxKeptResultChars: 0 }).maxKeptResultChars).toBe(0);
  });
});

describe('session message mapping', () => {
  it('returns the engine objects for untouched messages and handle-less copies for rebuilt ones', () => {
    const messages = transcript();
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    messages[1]!.toolUses[0]!.text = 'x'.repeat(2000);
    messages[2]!.toolResults![0]!.text = 'x'.repeat(2000);
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out).toHaveLength(messages.length);
    expect(out[0]).toBe(messages[0]);
    expect(out[1]?.handle).toBeUndefined();
    expect(out[1]?.toolUses[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[jev-compaction truncated 1700 chars`),
    );
    expect(out[2]?.handle).toBeUndefined();
    expect(out[2]?.toolResults?.[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[jev-compaction truncated 1700 chars`),
    );
    expect(out[2]?.toolResults?.[0]).toMatchObject({ tool_use_id: 'tool-1', isError: false });
    expect(out[3]).toBe(messages[3]);
    expect(out[4]).toBe(messages[4]);
  });

  it('preserves short dropped-result messages and their handles', () => {
    const messages = transcript();
    messages[1]!.toolUses[0]!.text = 'y'.repeat(100);
    messages[2]!.toolResults![0]!.text = 'y'.repeat(100);
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out[1]).toBe(messages[1]);
    expect(out[2]).toBe(messages[2]);
  });
});

describe('compactSession', () => {
  it('runs the library over the engine fetch and reports the outcome', async () => {
    const bodies: string[] = [];
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k', model: 'jev-x' };
    const { result: output, messages } = await compactSession(
      transcript(),
      config,
      jevFetch((name) => (name === 'call_t2' || name === 'result_t2' ? 0.9 : 0.01), bodies),
    );
    expect(bodies).toHaveLength(1);
    expect(JSON.parse(bodies[0]!).model).toBe('jev-x');
    expect(output.decisions.map((d) => d.action)).toEqual(['drop_call', 'keep']);
    expect(messages.map((m) => m.handle)).toEqual(['h-0', undefined, 'h-tool-2', 'r-tool-2', 'h-5', 'h-6']);
    expect(summarize(output)).toMatch(/^\d+% reduction; 1 kept, 1 call_dropped; state ~\d+ tokens \(full\) in 1 request\(s\)$/);
    expect(decisionLog(output)).toBe('t1:Read:drop_call/call=0.01/result=0.01 t2:Bash:keep/call=0.90/result=0.90');
    expect(decisionLogLines(output)).toEqual([`decisions: ${decisionLog(output)}`]);
  });

  it('splits a long decision log into ui.log lines under the host limit', async () => {
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k' };
    const { result: output } = await compactSession(transcript(), config, jevFetch(() => 0.01));
    const lines = decisionLogLines(output, 60);
    expect(lines).toEqual([
      'decisions (1/2): t1:Read:drop_call/call=0.01/result=0.01',
      'decisions (2/2): t2:Bash:drop_call/call=0.01/result=0.01',
    ]);
    expect(lines.every((line) => line.length <= 60)).toBe(true);
    expect(decisionLogLines({ ...output, decisions: [] })).toEqual(['decisions: (none)']);
  });

  it('throws on a missing key and on failed requests so the hook falls back', async () => {
    const config = resolveHookConfig({ preserveRecentMessages: 1 });
    await expect(compactSession(transcript(), config, jevFetch(() => 0))).rejects.toThrow(/TYPESAFE_API_KEY/);
    await expect(
      compactSession(transcript(), { ...config, apiKey: 'k' }, async () => ({ status: 500, ok: false, text: 'x' })),
    ).rejects.toThrow(/500/);
  });

  it('redacts secrets in tool results out of the whole request body, keep questions included', async () => {
    const bodies: string[] = [];
    const secret = 'sk-abcdefghijklmnopqrstuvwxyz123456';
    const messages = transcript();
    messages.splice(
      5,
      0,
      call('tool-3', 'Bash', { command: 'cat .env' }, 'env'),
      result('tool-3', `API_KEY=${secret} DB_PASSWORD=hunter2hunter2\n${'filler line\n'.repeat(200)}`),
    );
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k' };
    await compactSession(messages, config, jevFetch(() => 0.9, bodies));
    expect(bodies.length).toBeGreaterThan(0);
    const all = bodies.join('\n');
    expect(all).toContain('result_t3');
    expect(all).not.toContain(secret);
    expect(all).not.toContain('hunter2hunter2');
  });

  it('gives up on a Jev call that never answers instead of stalling the compaction', async () => {
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k' };
    const never = () => new Promise<never>(() => {});
    await expect(
      compactSession(transcript(), config, never, undefined, { timeoutMs: 30 }),
    ).rejects.toThrow(/timed out after 30 ms/);
    const controller = new AbortController();
    const pending = compactSession(transcript(), config, never, undefined, { signal: controller.signal });
    controller.abort(new Error('dispatch ended'));
    await expect(pending).rejects.toThrow(/dispatch ended/);
  });

  it('stops calling Jev for the rest of the compaction once a call times out', async () => {
    let calls = 0;
    const never = () => {
      calls++;
      return new Promise<never>(() => {});
    };
    const asker = jevAsker(never, 'k', 'jev-latest', { timeoutMs: 30 });
    await expect(asker.ask({}, {})).rejects.toThrow(/timed out after 30 ms/);
    await expect(asker.ask({}, {})).rejects.toThrow(/^Jev unavailable: .*timed out/);
    expect(calls).toBe(1);
  });

  it('bounds a Jev call with the injected sleep when the runtime has no AbortSignal.timeout or .any', async () => {
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k' };
    const never = () => new Promise<never>(() => {});
    const { timeout, any } = AbortSignal;
    // The hook runtime (JavaScriptCore) has neither, nor setTimeout.
    Object.assign(AbortSignal, { timeout: undefined, any: undefined });
    const sleeps: number[] = [];
    // Elapses at once for the 30 ms deadline; otherwise waits for the signal.
    const sleep = (ms: number, signal?: AbortSignal) => {
      sleeps.push(ms);
      if (ms === 30) return Promise.resolve();
      return new Promise<void>((_resolve, reject) =>
        signal?.addEventListener('abort', () => reject(signal.reason), { once: true }),
      );
    };
    try {
      await expect(
        compactSession(transcript(), config, never, undefined, { timeoutMs: 30, sleep }),
      ).rejects.toThrow(/timed out after 30 ms/);
      expect(sleeps).toContain(30);
      const controller = new AbortController();
      const aborted = compactSession(transcript(), config, never, undefined, { signal: controller.signal, sleep });
      controller.abort(new Error('dispatch ended'));
      await expect(aborted).rejects.toThrow(/dispatch ended/);
    } finally {
      Object.assign(AbortSignal, { timeout, any });
    }
  });

  it('skips a retry whose wait the hook budget cannot afford', async () => {
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k' };
    let calls = 0;
    const busy = async () => (++calls, { status: 429, ok: false, text: 'slow down', headers: { 'retry-after': '15' } });
    const sleeps: number[] = [];
    const sleep = async (ms: number) => void sleeps.push(ms);
    await expect(
      compactSession(transcript(), config, busy, undefined, { sleep, remainingMs: () => 9_000 }),
    ).rejects.toThrow(/429/);
    expect(calls).toBe(1);
    expect(sleeps).not.toContain(15_000);
    calls = 0;
    await expect(
      compactSession(transcript(), config, busy, undefined, { sleep, remainingMs: () => 60_000 }),
    ).rejects.toThrow(/429/);
    expect(calls).toBe(2);
    expect(sleeps).toContain(15_000);
  });

  it('reports a failed deadline timer as itself, not as an abort', async () => {
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k' };
    const never = () => new Promise<never>(() => {});
    const sleep = async () => {
      throw new Error('clock.sleep refused');
    };
    await expect(compactSession(transcript(), config, never, undefined, { sleep })).rejects.toThrow(
      /clock.sleep refused/,
    );
  });

  it('retries a 429 or 5xx once, and not a 4xx', async () => {
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k' };
    const ok = jevFetch(() => 0.01);
    let calls = 0;
    const flaky = async (url: string, init?: { body?: string }) =>
      ++calls === 1 ? { status: 503, ok: false, text: 'busy', headers: { 'retry-after': '0' } } : ok(url, init);
    const { result } = await compactSession(transcript(), config, flaky);
    expect(calls).toBe(2);
    expect(result.decisions).toHaveLength(2);
    calls = 0;
    const bad = async () => (++calls, { status: 400, ok: false, text: 'bad request' });
    await expect(compactSession(transcript(), config, bad)).rejects.toThrow(/400/);
    expect(calls).toBe(1);
  });

  it('archives a dropped call and cites the file in the note left behind, when an archive writer is given', async () => {
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k' };
    const files = new Map<string, string>();
    const writer = archiveWriter({
      fs: {
        exists: async (path: string) => files.has(path),
        write: async (path: string, text: string) => void files.set(path, text),
      },
    });
    const { result } = await compactSession(transcript(), config, jevFetch(() => 0.01), writer);
    const dropped = result.messages.find((m) => m.text.includes('removed the Read call'));
    expect(dropped?.text).toContain('.claude/jev-compaction/tool-1.txt');
    expect(files.get('.claude/jev-compaction/tool-1.txt')).toBe(fileA);
    expect(files.get('.claude/jev-compaction/.gitignore')).toBe('*\n');
  });
});

describe('archiveWriter', () => {
  const fsStub = () => {
    const files = new Map<string, string>();
    return {
      files,
      fs: {
        exists: async (path: string) => files.has(path),
        write: async (path: string, text: string) => void files.set(path, text),
      },
    };
  };

  it('names a file per cited call, writes them on flush, and adds a .gitignore', async () => {
    const stub = fsStub();
    const writer = archiveWriter(stub);
    const path = writer.cite('toolu_1', 'a'.repeat(5000));
    expect(path).toBe('.claude/jev-compaction/toolu_1.txt');
    expect(stub.files.size).toBe(0);
    await writer.flush();
    expect(stub.files.get('.claude/jev-compaction/.gitignore')).toBe('*\n');
    expect(stub.files.get(path!)?.length).toBe(5000);
  });

  it('archives text with credentials redacted, rather than skipping it', async () => {
    const stub = fsStub();
    const writer = archiveWriter(stub);
    const path = writer.cite('toolu_2', 'ok\nAWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI7K7MDENG\ndone');
    await writer.flush();
    expect(stub.files.get(path!)).toBe('ok\n[REDACTED]\ndone');
  });

  it('archives text with the configured API key redacted even when it matches no secret pattern', async () => {
    const stub = fsStub();
    const writer = archiveWriter(stub, undefined, 'plainkeyvalue123');
    const path = writer.cite('toolu_5', 'pasted plainkeyvalue123 here');
    await writer.flush();
    expect(stub.files.get(path!)).not.toContain('plainkeyvalue123');
  });

  it('archives git output and paths verbatim', async () => {
    const stub = fsStub();
    const writer = archiveWriter(stub);
    const text = 'commit 3bfad06e1c2d4f5a6b7c8d9e0f1a2b3c4d5e6f7a\n/Users/yusri/Documents/Claude/Orchestra2/notes';
    const path = writer.cite('toolu_3', text);
    await writer.flush();
    expect(stub.files.get(path!)).toBe(text);
  });

  it('names and writes files under the directory it is given', async () => {
    const stub = fsStub();
    const writer = archiveWriter(stub, '/repo/.claude/jev-compaction');
    const path = writer.cite('toolu_4', 'text');
    expect(path).toBe('/repo/.claude/jev-compaction/toolu_4.txt');
    expect(writer.pathOf('toolu_4')).toBe(path);
    await writer.flush();
    expect(stub.files.get('/repo/.claude/jev-compaction/.gitignore')).toBe('*\n');
    expect(stub.files.get(path!)).toBe('text');
  });

  it('writes nothing when nothing was cited', async () => {
    const stub = fsStub();
    await archiveWriter(stub).flush();
    expect(stub.files.size).toBe(0);
  });
});

describe('archiveDir', () => {
  const engine = (git: { exitCode: number; stdout: string } | Error) => {
    const runs: { argv: readonly string[]; cwd?: string }[] = [];
    return {
      runs,
      $: {
        session: { cwd: async () => '/repo/plugins/sub', root: async () => '/started' },
        process: {
          run: async (argv: readonly string[], init?: { cwd?: string }) => {
            runs.push({ argv, cwd: init?.cwd });
            if (git instanceof Error) throw git;
            return { ...git, stderr: '' };
          },
        },
      },
    };
  };

  it('anchors the archive at the git toplevel of the session directory, not the directory itself', async () => {
    const e = engine({ exitCode: 0, stdout: '/repo\n' });
    expect(await archiveDir(e.$)).toBe('/repo/.claude/jev-compaction');
    expect(e.runs).toEqual([{ argv: ['git', 'rev-parse', '--show-toplevel'], cwd: '/repo/plugins/sub' }]);
  });

  it('falls back to the session root outside a git repository', async () => {
    expect(await archiveDir(engine({ exitCode: 128, stdout: '' }).$)).toBe('/started/.claude/jev-compaction');
  });

  it('falls back to the relative directory when the engine cannot answer', async () => {
    expect(await archiveDir(engine(new Error('no git')).$)).toBe('.claude/jev-compaction');
    expect(await archiveDir({} as never)).toBe('.claude/jev-compaction');
  });
});

describe('getApiKey', () => {
  const engine = (env: Record<string, string> = {}, settingsEnv: Record<string, string> = {}) => ({
    env: { get: async (name: string) => env[name] },
    settings: { read: async () => ({ env: settingsEnv }) },
  });

  it('prefers config.apiKey, then TYPESAFE_API_KEY from the environment, then the settings env block', async () => {
    expect(
      await getApiKey(engine({ TYPESAFE_API_KEY: 'typesafe' }), { ...resolveHookConfig({}), apiKey: 'configured' }),
    ).toBe('configured');
    expect(
      await getApiKey(engine({ TYPESAFE_API_KEY: 'from-env' }, { TYPESAFE_API_KEY: 'from-settings' }), resolveHookConfig({})),
    ).toBe('from-env');
    expect(await getApiKey(engine({}, { TYPESAFE_API_KEY: 'from-settings' }), resolveHookConfig({}))).toBe(
      'from-settings',
    );
    expect(await getApiKey(engine(), resolveHookConfig({}))).toBeUndefined();
  });

  it('treats an empty-string TYPESAFE_API_KEY in the environment as unset', async () => {
    expect(
      await getApiKey(engine({ TYPESAFE_API_KEY: '' }, { TYPESAFE_API_KEY: 'from-settings' }), resolveHookConfig({})),
    ).toBe('from-settings');
  });

});

describe('register', () => {
  type Hook = (...args: any[]) => Promise<unknown>;
  function registered(options: Record<string, unknown> = {}) {
    const hooks: Record<string, Hook> = {};
    (register as any)((event: string, matcherOrHook: unknown, maybeHook?: Hook) => {
      hooks[event] = typeof maybeHook === 'function' ? maybeHook : (matcherOrHook as Hook);
    }, { apiKey: 'k', ...LEGACY, ...options });
    return hooks;
  }
  function engine(fetchFn: ReturnType<typeof jevFetch>) {
    const logs: string[] = [];
    const toasts: string[] = [];
    const $ = {
      clock,
      http: {
        fetch: async (url: string, init?: { body?: string }) => fetchFn(url, init),
      },
      ui: {
        log: (t: string) => logs.push(t),
        toast: (t: string) => toasts.push(t),
      },
      session: {
        usage: async () => ({ context: { percent: 100 } }),
        compact: async () => ({ messages: [] }),
      },
    };
    return { $, logs, toasts, get notices() { return logs.filter((l) => NOTICE.test(l)); } };
  }
  function next(signal: AbortSignal = new AbortController().signal) {
    return Object.assign(async (e: unknown) => e, { signal, budget: { ms: 10_000, remainingMs: 10_000 } });
  }

  it('/jev reports the last compaction and the settings, without the per-call decisions', async () => {
    const hooks = registered({ preserveRecentMessages: 1, minReductionRatio: 0, archiveResults: false });
    const e = engine(jevFetch((name) => (name.startsWith('result_') ? 0.01 : 0.9)));
    const $ = { ...e.$, env: { get: async () => undefined }, settings: { read: async () => ({}) } };
    const before = await hooks['command.run']!($, { command: 'jev', args: '' }, next());
    expect(before.text).toMatch(/^No compaction yet in this session\.\nSettings: TypeSafe key set · endpoint TypeSafe · model jev-latest/);
    await hooks['session.compact']!($, { trigger: 'manual', messages: transcript() }, next());
    const after = await hooks['command.run']!($, { command: 'jev', args: '' }, next());
    expect(after.text).toMatch(/^Last compaction \(manual\)\nCompacted with Jev: /);
    expect(after.text).toContain('  kept ');
    expect(after.text).not.toContain('decisions');
    expect(after.text).toMatch(/\nSettings: /);
  });

  it('registers /jev when the plugin loads', async () => {
    const hooks = registered();
    const registeredCommands: unknown[] = [];
    const $ = { command: { register: async (spec: unknown) => void registeredCommands.push(spec) } };
    await hooks['session.start']!($, { cwd: '/' }, Object.assign(async (e: unknown) => e, {}));
    expect(registeredCommands).toEqual([expect.objectContaining({ name: 'jev', immediate: true })]);
  });

  it('installs no turn.complete hook, so compaction only runs when Claude Code compacts', () => {
    expect(Object.keys(registered()).sort()).toEqual(['command.run', 'session.compact', 'session.start', 'tool.call']);
  });

  it('skips a 429 retry the hook budget from next.budget cannot afford', async () => {
    const hooks = registered({ preserveRecentMessages: 1, minReductionRatio: 0 });
    const run = async (remainingMs: number) => {
      let calls = 0;
      const e = engine(async () => (++calls, {
        status: 429,
        ok: false,
        text: 'slow down',
        headers: { 'retry-after': '0' },
      }));
      const n = Object.assign(async (x: unknown) => x, {
        signal: new AbortController().signal,
        budget: { ms: 10_000, remainingMs },
      });
      await hooks['session.compact']!(e.$, { trigger: 'manual', messages: transcript() }, n);
      return calls;
    };
    const roomy = await run(10_000);
    expect(roomy).toBeGreaterThan(1);
    // Same 429, but too little budget left to afford a retry: each Jev request is attempted once.
    expect(await run(1_000)).toBeLessThan(roomy);
  });

  it('vetoes an interrupted dispatch instead of falling back to a summary', async () => {
    const hooks = registered({ preserveRecentMessages: 1, minReductionRatio: 0 });
    const controller = new AbortController();
    const e = engine(async () => {
      controller.abort();
      throw new Error('gateway down');
    });
    const out = await hooks['session.compact']!(
      e.$,
      { trigger: 'manual', messages: transcript() },
      next(controller.signal),
    );
    expect(out).toEqual({ skip: 'jev-compaction: interrupted' });
    expect(e.toasts).toEqual([]);
  });

  it('compacts a subagent transcript with Jev too, without reading the main context size', async () => {
    const hooks = registered({ preserveRecentMessages: 1, minReductionRatio: 0 });
    const e = engine(async (_url: string, init?: { body?: string }) => {
      const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
      const answers = Object.fromEntries(Object.keys(questions).map((k) => [k, { noul: 0.9 }]));
      return { status: 200, ok: true, text: JSON.stringify({ answers }) };
    });
    const usage = e.$.session.usage;
    let usageRead = false;
    e.$.session.usage = async () => ((usageRead = true), usage());
    const out = await hooks['session.compact']!(
      e.$,
      { trigger: 'plugin', agentId: 'sub-1', messages: transcript() },
      next(),
    );
    expect(Array.isArray(out.messages)).toBe(true);
    expect(out).not.toHaveProperty('trigger');
    expect(out.tokensAfter).toBeLessThanOrEqual(out.tokensBefore);
    expect(usageRead).toBe(false);
  });

  it('skips a speculative precompute instead of running Jev on it', async () => {
    const hooks = registered({ preserveRecentMessages: 1, minReductionRatio: 0 });
    const e = engine(async () => ({ status: 200, ok: true, text: '{"answers":{}}' }));
    const out = await hooks['session.compact']!(
      e.$,
      { trigger: 'precompute', messages: transcript() },
      next(),
    );
    expect(out).toEqual({
      skip: 'jev-compaction does not handle speculative compactions',
    });
  });

  it('reports a real failure as a fallback for the engine\'s own auto trigger', async () => {
    const hooks = registered({ preserveRecentMessages: 1, minReductionRatio: 0 });
    const e = engine(async () => ({ status: 500, ok: false, text: 'down' }));
    const out = await hooks['session.compact']!(
      e.$,
      { trigger: 'auto', messages: transcript() },
      next(),
    );
    expect(out).toEqual({ trigger: 'auto', messages: transcript() });
    expect(e.toasts).toHaveLength(1);
    expect(e.notices[0]).toMatch(/fallback to built-in summary/);
    expect(e.toasts).toEqual([expect.stringMatching(/^Using Claude's summary: Jev request failed \(.{1,100}\)$/)]);
  });

  it('skips instead of falling back to a summary for a manual /compact', async () => {
    const hooks = registered({ preserveRecentMessages: 1, minReductionRatio: 0 });
    const e = engine(async () => ({ status: 500, ok: false, text: 'down' }));
    const out = await hooks['session.compact']!(
      e.$,
      { trigger: 'manual', messages: transcript() },
      next(),
    );
    expect(out).toEqual({ skip: 'jev-compaction: request failed, builtin fallback disabled' });
    expect(e.toasts).toHaveLength(1);
    expect(e.notices[0]).toMatch(/builtin fallback disabled/);
  });

  it('skips a below-minimum reduction instead of falling back for a manual /compact', async () => {
    const hooks = registered({ preserveRecentMessages: 1, minReductionRatio: 1 });
    const e = engine(jevFetch(() => 1));
    // The runtime has no setTimeout: each Jev call must be timed by $.clock.
    const before = clock.calls;
    const out = await hooks['session.compact']!(
      e.$,
      { trigger: 'manual', messages: transcript() },
      next(),
    );
    expect(out).toEqual({
      skip: 'jev-compaction: below minimum reduction, builtin fallback disabled',
    });
    expect(clock.calls).toBeGreaterThan(before);
  });

  it('always falls back when builtinFallback is set to always', async () => {
    const hooks = registered({
      preserveRecentMessages: 1,
      minReductionRatio: 0,
      builtinFallback: 'always',
    });
    const e = engine(async () => ({ status: 500, ok: false, text: 'down' }));
    const out = await hooks['session.compact']!(
      e.$,
      { trigger: 'manual', messages: transcript() },
      next(),
    );
    expect(out).toEqual({ trigger: 'manual', messages: transcript() });
  });

  it('never falls back when builtinFallback is set to never, even for the auto trigger', async () => {
    const hooks = registered({
      preserveRecentMessages: 1,
      minReductionRatio: 0,
      builtinFallback: 'never',
    });
    const e = engine(async () => ({ status: 500, ok: false, text: 'down' }));
    const out = await hooks['session.compact']!(
      e.$,
      { trigger: 'auto', messages: transcript() },
      next(),
    );
    expect(out).toEqual({ skip: 'jev-compaction: request failed, builtin fallback disabled' });
  });

  it('passes /compact instructions through as extra goal context', async () => {
    const hooks = registered({ preserveRecentMessages: 1, minReductionRatio: 0, goal: 'base goal' });
    let sentGoal: string | undefined;
    const e = engine(async (url, init) => {
      sentGoal = JSON.parse(String((init as RequestInit).body)).state.goal;
      return { status: 200, ok: true, text: '{"answers":{}}' };
    });
    await hooks['session.compact']!(
      e.$,
      { trigger: 'manual', messages: transcript(), instructions: 'focus on the auth bug' },
      next(),
    );
    expect(sentGoal).toContain('base goal');
    expect(sentGoal).toContain('focus on the auth bug');
  });
});

describe('register: tool.call (Bash output trimming)', () => {
  type Hook = (...args: any[]) => Promise<unknown>;
  function registeredMatchers(options: Record<string, unknown> = {}) {
    const hooks: Record<string, { matcher: unknown; hook: Hook }> = {};
    (register as any)(
      (event: string, matcherOrHook: unknown, maybeHook?: Hook) => {
        hooks[event] =
          typeof maybeHook === 'function'
            ? { matcher: matcherOrHook, hook: maybeHook }
            : { matcher: undefined, hook: matcherOrHook as Hook };
      },
      { apiKey: 'k', ...options },
    );
    return hooks;
  }
  function bashEngine(fetchFn: (url: string, init?: { body?: string }) => Promise<unknown>) {
    return {
      env: { get: async () => undefined },
      settings: { read: async () => ({}) },
      ui: { log: () => {}, toast: () => {} },
      http: { fetch: fetchFn as any },
      clock,
    };
  }

  it('matches only the Bash tool', () => {
    const { matcher } = registeredMatchers()['tool.call']!;
    expect(matcher).toEqual({ tool: 'Bash' });
  });

  it('leaves output untouched when jevPruneBashOutput is off', async () => {
    const { hook } = registeredMatchers({ jevPruneBashOutput: false })['tool.call']!;
    let asked = false;
    const $ = bashEngine(async () => {
      asked = true;
      return { status: 200, ok: true, text: '{"answers":{}}' };
    });
    const long = 'line\n'.repeat(2000);
    const out = (await hook($, { command: 'npm test' }, next(async () => ({ result: { stdout: long } })))) as {
      result: { stdout: string };
    };
    expect(out.result.stdout).toBe(long);
    expect(asked).toBe(false);
  });

  it('trims long Bash stdout through Jev when opted in', async () => {
    const { hook } = registeredMatchers({ jevPruneBashOutput: true, jevPruneMinChars: 10 })['tool.call']!;
    const $ = bashEngine(async () => {
      // 60 lines / 20 per chunk = 3 chunks (c1, c2, c3). The first and last
      // are always kept regardless of score, so drop the middle one to make
      // trimming observable and deterministic.
      const answers = {
        c1: { type: 'noul', noul: 1 },
        c2: { type: 'noul', noul: 0 },
        c3: { type: 'noul', noul: 1 },
      };
      return { status: 200, ok: true, text: JSON.stringify({ answers }) };
    });
    const long = Array.from({ length: 60 }, (_, i) => `line ${i}`).join('\n');
    const before = clock.calls;
    const out = (await hook(
      $,
      { command: 'npm test' },
      next(async () => ({ result: { stdout: long } })),
    )) as { result: { stdout: string } };
    expect(out.result.stdout.length).toBeLessThan(long.length);
    expect(out.result.stdout).toContain('jev-compaction trimmed');
    expect(clock.calls).toBeGreaterThan(before);
  });

  it('leaves a short Bash result untouched even when opted in', async () => {
    const { hook } = registeredMatchers({ jevPruneBashOutput: true, jevPruneMinChars: 4000 })['tool.call']!;
    let asked = false;
    const $ = bashEngine(async () => {
      asked = true;
      return { status: 200, ok: true, text: '{"answers":{}}' };
    });
    const out = (await hook(
      $,
      { command: 'echo hi' },
      next(async () => ({ result: { stdout: 'hi' } })),
    )) as { result: { stdout: string } };
    expect(out.result.stdout).toBe('hi');
    expect(asked).toBe(false);
  });

  it('skips an image result, a background task, or output already persisted to disk', async () => {
    const { hook } = registeredMatchers({ jevPruneBashOutput: true, jevPruneMinChars: 1 })['tool.call']!;
    let asked = false;
    const $ = bashEngine(async () => {
      asked = true;
      return { status: 200, ok: true, text: '{"answers":{}}' };
    });
    for (const result of [
      { stdout: 'x'.repeat(50), isImage: true },
      { stdout: 'x'.repeat(50), backgroundTaskId: 'bg-1' },
      { stdout: 'x'.repeat(50), persistedOutputPath: '/tmp/out.txt' },
    ]) {
      const out = (await hook($, { command: 'x' }, next(async () => ({ result })))) as {
        result: typeof result;
      };
      expect(out.result).toBe(result);
    }
    expect(asked).toBe(false);
  });

  function next(fn: () => Promise<unknown>) {
    return Object.assign(fn, { signal: new AbortController().signal, budget: { ms: 10_000, remainingMs: 10_000 } });
  }
});

describe('proseTrim rescue', () => {
  const FLUFF = 'Let me have a good look at the layout of this module now.';
  const isProse = (key: string) => /^c\d+$/.test(key);

  function proseTranscript(): SessionMessage[] {
    return [
      message('user', 'Fix the stale cache.', { handle: 'p-0' }),
      message('assistant', Array(10).fill(FLUFF).join(' '), { handle: 'p-1' }),
      message('user', 'thanks', { handle: 'p-2' }),
      message('assistant', 'Done.', { handle: 'p-3' }),
      message('user', 'ok', { handle: 'p-4' }),
    ];
  }

  /** Prose parts (`c<n>`) get `prose` (or a 500 for 'fail'); every other question gets `tool` (default 0.9). Each request is logged. */
  function fetchWith(prose: number | 'fail', requests: string[][] = [], tool = 0.9) {
    return async (_url: string, init?: { body?: string }) => {
      const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
      const keys = Object.keys(questions);
      requests.push(keys);
      if (prose === 'fail' && keys.some(isProse)) return { status: 500, ok: false, text: 'down' };
      const answers = Object.fromEntries(
        keys.map((k) => [k, { type: 'noul', noul: isProse(k) ? prose : tool }]),
      );
      return { status: 200, ok: true, text: JSON.stringify({ answers }) };
    };
  }
  const proseRequests = (requests: string[][]) => requests.filter((keys) => keys.some(isProse));

  const cfg = (extra: Record<string, unknown> = {}) => ({
    ...resolveHookConfig({ preserveRecentMessages: 2, archiveResults: false, ...extra }),
    apiKey: 'k',
  });
  const strip = (r: Awaited<ReturnType<typeof compactSession>>) => ({
    messages: r.messages,
    decisions: r.result.decisions,
    stats: { ...r.result.stats, ms: 0 },
    prose: r.prose,
  });

  it('reads proseTrim as a boolean, on unless set to false, ignoring other values', () => {
    expect(resolveHookConfig({}).proseTrim).toBe(true);
    expect(resolveHookConfig({ proseTrim: false }).proseTrim).toBe(false);
    expect(resolveHookConfig({ proseTrim: 'trim' }).proseTrim).toBe(true);
  });

  it('declares proseTrim in the manifest as a boolean defaulting to true', () => {
    const manifest = JSON.parse(readFileSync(new URL('../.claude-plugin/plugin.json', import.meta.url), 'utf8'));
    expect(manifest.userConfig.proseTrim).toMatchObject({ type: 'boolean', default: true });
  });

  it('makes no prose request when tool compaction reaches the minimum', async () => {
    const requests: string[][] = [];
    const out = await compactSession(transcript(), cfg({ proseTrim: true }), fetchWith(0.01, requests, 0.01));
    const plain = await compactSession(transcript(), cfg(), fetchWith(0.01, [], 0.01));
    expect(reductionRatioOf(out)).toBeGreaterThanOrEqual(0.25);
    expect(proseRequests(requests)).toHaveLength(0);
    expect(out.prose).toBeUndefined();
    expect(strip(out)).toEqual(strip(plain));
  });

  it('uses the first cut-off that reaches the minimum', async () => {
    // noul 0.1: the 0.05 cut-off keeps every part, 0.15 drops them all.
    const requests: string[][] = [];
    const out = await compactSession(proseTranscript(), cfg({ proseTrim: true }), fetchWith(0.1, requests));
    expect(out.prose?.cutoff).toBe(0.15);
    expect(out.prose?.stats.dropped).toBe(10);
    expect(reductionRatioOf(out)).toBeGreaterThanOrEqual(0.25);
    expect(out.messages.find((m) => m.handle === 'p-1')).toBeUndefined();
    expect(out.messages.map((m) => m.handle)).toContain('p-0');
    expect(summarize(out.result, out.prose)).toContain('; prose trim at 0.15: 10 dropped');
  });

  it('walks the whole ladder on one round of prose requests', async () => {
    const requests: string[][] = [];
    const out = await compactSession(proseTranscript(), cfg({ proseTrim: true }), fetchWith(0.9, requests));
    expect(out.prose?.cutoff).toBeUndefined();
    const asked = proseRequests(requests);
    expect(asked).toHaveLength(1);
    expect(asked[0]).toHaveLength(10);
    expect(new Set(asked[0]).size).toBe(10);
    const rescued = await (async () => {
      const seen: string[][] = [];
      const r = await compactSession(proseTranscript(), cfg({ proseTrim: true }), fetchWith(0.3, seen));
      return { r, seen };
    })();
    // noul 0.3 passes 0.05 and 0.15 but not 0.35: the third cut-off, still one round.
    expect(rescued.r.prose?.cutoff).toBe(0.35);
    expect(proseRequests(rescued.seen)).toHaveLength(1);
  });

  it('keeps the tool result and records the attempt when no cut-off reaches the minimum', async () => {
    const plain = await compactSession(proseTranscript(), cfg({ proseTrim: false, notificationTrim: false }), fetchWith(0.9));
    const out = await compactSession(proseTranscript(), cfg({ proseTrim: true }), fetchWith(0.9));
    expect(out.prose).toMatchObject({ stats: { requests: 1 } });
    expect(out.prose?.cutoff).toBeUndefined();
    expect(strip({ ...out, prose: undefined })).toEqual(strip(plain));
  });

  it('keeps the tool result when the prose stage fails completely', async () => {
    const plain = await compactSession(proseTranscript(), cfg({ proseTrim: false, notificationTrim: false }), fetchWith(0.9));
    const out = await compactSession(proseTranscript(), cfg({ proseTrim: true }), fetchWith('fail'));
    expect(out.prose?.stats.failed).toBeTruthy();
    expect(out.prose?.cutoff).toBeUndefined();
    expect(strip({ ...out, prose: undefined })).toEqual(strip(plain));
  });

  it('proseTrim false keeps the 0.6.1 result, with no prose request', async () => {
    const seenOff: string[][] = [];
    const seenOmitted: string[][] = [];
    const off = await compactSession(proseTranscript(), cfg({ proseTrim: false }), fetchWith(0.01, seenOff));
    const omitted = await compactSession(proseTranscript(), cfg({ proseTrim: false, notificationTrim: false }), fetchWith(0.01, seenOmitted));
    expect(strip(off)).toEqual(strip(omitted));
    expect(off.prose).toBeUndefined();
    expect(seenOff).toEqual(seenOmitted);
    expect(proseRequests(seenOff)).toHaveLength(0);
  });

  it('with no options, tool compaction below the minimum runs the trim rescue', async () => {
    const requests: string[][] = [];
    const out = await compactSession(proseTranscript(), cfg(), fetchWith(0.1, requests));
    expect(proseRequests(requests)).toHaveLength(1);
    expect(out.prose?.cutoff).toBe(0.15);
    expect(reductionRatioOf(out)).toBeGreaterThanOrEqual(0.25);
  });

  it('with both trims false, nothing but tool compaction runs', async () => {
    const requests: string[][] = [];
    const out = await compactSession(proseTranscript(), cfg({ proseTrim: false, notificationTrim: false }), fetchWith(0.1, requests));
    expect(proseRequests(requests)).toHaveLength(0);
    expect(out.prose).toBeUndefined();
    expect(out.notifications).toBeUndefined();
  });

  describe('through the hook', () => {
    const hook = (
      options: Record<string, unknown>,
      fetchFn = fetchWith(0.1),
      usageTokens?: number,
      builtin: (e: unknown) => Promise<unknown> = async (e) => e,
      messagesInContext?: number,
    ) => {
      const hooks: Record<string, (...args: any[]) => Promise<any>> = {};
      (register as any)(
        (event: string, a: unknown, b?: any) => {
          hooks[event] = typeof b === 'function' ? b : (a as any);
        },
        { apiKey: 'k', preserveRecentMessages: 2, archiveResults: false, ...options },
      );
      const toasts: string[] = [];
      const notices: string[] = [];
      const $ = {
        clock,
        http: { fetch: async (url: string, init?: { body?: string }) => fetchFn(url, init) },
        ui: { log: (t: string) => void (NOTICE.test(t) && notices.push(t)), toast: (t: string) => toasts.push(t) },
        ...(usageTokens === undefined
          ? {}
          : {
              session: {
                usage: async () => ({
                  context: {
                    tokens: usageTokens,
                    window: 200_000,
                    ...(messagesInContext === undefined
                      ? {}
                      : { breakdown: { categories: [{ name: 'Messages', kind: 'used', tokens: messagesInContext }] } }),
                  },
                }),
              },
            }),
      };
      const next = Object.assign(builtin, {
        signal: new AbortController().signal,
        budget: { ms: 10_000, remainingMs: 10_000 },
      });
      return { run: (event: Record<string, unknown>) => hooks['session.compact']!($, event, next), toasts, notices };
    };

    describe('measured against the real context', () => {
      const history = () => messageTokens(proseTranscript());

      it('compacts as before when the context is no larger than the history', async () => {
        const out = await hook({ proseTrim: true }, fetchWith(0.1), history()).run({ trigger: 'auto', messages: proseTranscript() });
        expect(Object.keys(out)).toEqual(['messages', 'tokensBefore', 'tokensAfter']);
      });

      it('falls back without asking Jev when no history cut can free the minimum', async () => {
        const requests: string[][] = [];
        const event = { trigger: 'auto', messages: proseTranscript() };
        const on = hook({ proseTrim: true }, fetchWith(0.1, requests), history() * 10);
        expect(await on.run(event)).toBe(event);
        expect(requests).toHaveLength(0);
        expect(on.notices[0]).toBe('fallback to built-in summary (history is too small a share of the context to free 25%)');
      });

      it("measures the history's share with the engine's Messages count, not the plugin's estimate", async () => {
        const requests: string[][] = [];
        const event = { trigger: 'auto', messages: proseTranscript() };
        // By the estimate the history is a quarter of the context, too small to free 25%
        // even if all of it went; by the engine's count it is three quarters.
        const on = hook({ proseTrim: true }, fetchWith(0.1, requests), history() * 4, undefined, history() * 3);
        await on.run(event);
        expect(requests.length).toBeGreaterThan(0);
        expect(on.notices[0]).not.toMatch(/too small a share/);
      });

      it('falls back when the history cut frees too little of the context', async () => {
        const event = { trigger: 'auto', messages: proseTranscript() };
        const on = hook({ proseTrim: true }, fetchWith(0.1), Math.round(history() * 3.9));
        expect(await on.run(event)).toBe(event);
        expect(on.notices[0]).toMatch(/^fallback to built-in summary \(below 25% minimum, prose trim tried, history needs 9\d% to free 25% of context: /);
        expect(on.toasts).toEqual([expect.stringMatching(/^Using Claude's summary: Jev's cut frees \d+% of the context, below the 25% minimum$/)]);
      });

      it('rethrows a failed built-in summary when its own cut freed nothing', async () => {
        const failing = async () => {
          throw new Error('summary failed');
        };
        const event = { trigger: 'auto', messages: proseTranscript() };
        const on = hook({ proseTrim: true }, fetchWith(0.1), Math.round(history() * 3.9), failing);
        await expect(on.run(event)).rejects.toThrow('summary failed');
      });

    });

    it('replaces history and names the cut-off', async () => {
      const event = { trigger: 'auto', messages: proseTranscript() };
      const on = hook({ proseTrim: true });
      const out = await on.run(event);
      expect(Object.keys(out)).toEqual(['messages', 'tokensBefore', 'tokensAfter']);
      expect(out.messages.length).toBe(4);
      expect(on.notices.join('\n')).toContain('prose trim at 0.15');
      expect(on.notices.join('\n')).toContain('no summary');
      expect(on.toasts).toEqual([expect.stringMatching(/^Compacted with Jev: 5 → 4 messages, \d+% smaller, no summary$/)]);
    });

    describe('archiving cut code bodies', () => {
      const body = 'const value = compute();\n'.repeat(80);
      const withWrite = (): SessionMessage[] => [
        message('user', 'Fix the stale cache.', { handle: 'w-0' }),
        call('wr1', 'Write', { file_path: 'src/app.ts', content: body }, 'ok'),
        result('wr1', 'ok'),
        message('user', 'thanks', { handle: 'w-3' }),
        message('assistant', 'Done.', { handle: 'w-4' }),
        message('user', 'ok', { handle: 'w-5' }),
      ];
      const run = async (archiveResults: boolean) => {
        const files = new Map<string, string>();
        const hooks: Record<string, (...args: any[]) => Promise<any>> = {};
        (register as any)(
          (event: string, a: unknown, b?: any) => {
            hooks[event] = typeof b === 'function' ? b : (a as any);
          },
          { apiKey: 'k', preserveRecentMessages: 2, archiveResults, proseTrim: true },
        );
        const $ = {
          clock,
          http: { fetch: async (url: string, init?: { body?: string }) => fetchWith(0.9)(url, init) },
          ui: { log: () => {}, toast: () => {} },
          fs: {
            exists: async (path: string) => files.has(path),
            write: async (path: string, text: string) => void files.set(path, text),
          },
        };
        const next = Object.assign(async (e: unknown) => e, {
          signal: new AbortController().signal,
          budget: { ms: 10_000, remainingMs: 10_000 },
        });
        const out = await hooks['session.compact']!($, { trigger: 'auto', messages: withWrite() }, next);
        return { out, files };
      };
      const written = (out: { messages: SessionMessage[] }) =>
        out.messages.flatMap((m) => m.toolUses).find((t) => t.tool === 'Write')?.input['content'] as string;

      it('archives the cut body and cites it when archiveResults is on', async () => {
        const { out, files } = await run(true);
        const cut = written(out);
        const path = /full output: (\S+?)\]/.exec(cut)?.[1];
        expect(path).toBeDefined();
        expect(files.get(path!)).toBe(body);
        expect(files.get('.claude/jev-compaction/.gitignore')).toBe('*\n');
      });

      it('writes nothing and keeps the old marker when archiveResults is off', async () => {
        const { out, files } = await run(false);
        expect(written(out)).toContain('Read the file if needed]');
        expect(files.size).toBe(0);
      });
    });

    it('falls back, saying trim was tried, when no cut-off reaches the minimum', async () => {
      const event = { trigger: 'auto', messages: proseTranscript() };
      const on = hook({ proseTrim: true, hybridSummary: false }, fetchWith(0.9));
      expect(await on.run(event)).toBe(event);
      expect(on.notices[0]).toMatch(/^fallback to built-in summary \(below 25% minimum, prose trim tried: /);
      const never = hook({ proseTrim: true, builtinFallback: 'never' }, fetchWith(0.9));
      expect(await never.run(event)).toEqual({ skip: 'jev-compaction: below minimum reduction, builtin fallback disabled' });
      expect(never.notices[0]).toContain('minimum, prose trim tried, builtin fallback disabled');
    });

    it('falls back when the prose stage fails, and says nothing about trim with proseTrim off', async () => {
      const event = { trigger: 'auto', messages: proseTranscript() };
      const failed = hook({ proseTrim: true, hybridSummary: false }, fetchWith('fail'));
      expect(await failed.run(event)).toBe(event);
      expect(failed.notices[0]).toContain('prose trim tried');
      const off = hook({ proseTrim: false, notificationTrim: false, hybridSummary: false }, fetchWith(0.1));
      expect(await off.run(event)).toBe(event);
      expect(off.notices[0]).not.toContain('prose');
    });
  });
});

function reductionRatioOf(r: { result: { stats: { charsBefore: number; charsAfter: number } } }): number {
  return (r.result.stats.charsBefore - r.result.stats.charsAfter) / r.result.stats.charsBefore;
}

describe('notificationTrim rescue', () => {
  const isNotif = (key: string) => /^n\d+$/.test(key);
  const isProse = (key: string) => /^c\d+$/.test(key);
  const REPORT = 'The worker examined the module and found the cache stale. '.repeat(25);
  const SECRET = 'sk-abcdefghijklmnopqrstuvwxyz0123456789';

  function notification(id: string, body = REPORT, summary = 'Worker finished the audit'): string {
    return [
      '<task-notification>',
      `<task-id>${id}</task-id>`,
      `<tool-use-id>toolu_${id}</tool-use-id>`,
      `<output-file>/tmp/tasks/${id}.output</output-file>`,
      '<status>completed</status>',
      `<summary>${summary}</summary>`,
      `<result>${body}\nerror: build failed in src/cache/store.ts at 4f3a9c2\n${body}</result>`,
      '<usage><total_tokens>81234</total_tokens><tool_uses>31</tool_uses></usage>',
      '<diagnostics>Output file is missing: /tmp/tasks/gone.output</diagnostics>',
      '</task-notification>',
    ].join('\n');
  }

  /** Message 0, a kept Read, three finished workers, then a short pinned tail. */
  function transcriptWithReports(extra: SessionMessage[] = []): SessionMessage[] {
    const note = (id: string) => [message('assistant', 'Dispatching a worker.'), message('user', notification(id))];
    return [
      message('user', 'Fix the stale cache.', { handle: 'n-0' }),
      call('rd1', 'Read', { file_path: 'src/a.ts' }, 'const a = 1;'),
      result('rd1', 'const a = 1;'),
      ...extra,
      ...note('w1'),
      ...note('w2'),
      ...note('w3'),
      message('assistant', 'Done.'),
      message('user', 'ok'),
    ];
  }

  /** Notification questions (`n<k>`) get `notif`, prose parts `prose`, tool questions 0.9; 'fail' answers notification requests with a 500. Each request is logged. */
  function fetchWith(notif: number | 'fail', requests: string[][] = [], prose = 0.9, bodies: string[] = []) {
    return async (_url: string, init?: { body?: string }) => {
      bodies.push(init?.body ?? '');
      const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
      const keys = Object.keys(questions);
      requests.push(keys);
      if (notif === 'fail' && keys.some(isNotif)) return { status: 500, ok: false, text: 'down' };
      const answers = Object.fromEntries(
        keys.map((k) => [k, { type: 'noul', noul: isNotif(k) ? notif : isProse(k) ? prose : 0.9 }]),
      );
      return { status: 200, ok: true, text: JSON.stringify({ answers }) };
    };
  }
  const notifRequests = (requests: string[][]) => requests.filter((keys) => keys.some(isNotif));

  const cfg = (extra: Record<string, unknown> = {}) => ({
    ...resolveHookConfig({ preserveRecentMessages: 2, archiveResults: false, ...extra }),
    apiKey: 'k',
  });
  const memoryArchive = () => {
    const files = new Map<string, string>();
    return {
      files,
      writer: archiveWriter({
        fs: { exists: async (p: string) => files.has(p), write: async (p: string, t: string) => void files.set(p, t) },
      }),
    };
  };
  const strip = (r: Awaited<ReturnType<typeof compactSession>>) => ({
    messages: r.messages,
    decisions: r.result.decisions,
    stats: { ...r.result.stats, ms: 0 },
  });
  const textOf = (r: Awaited<ReturnType<typeof compactSession>>) => r.messages.map((m) => m.text).join('\n');

  it('reads notificationTrim as a boolean, absent (default on) unless set to false', () => {
    expect(resolveHookConfig({}).notificationTrim).toBeUndefined();
    expect(resolveHookConfig({ notificationTrim: false }).notificationTrim).toBe(false);
    expect(resolveHookConfig({ notificationTrim: 'yes' }).notificationTrim).toBeUndefined();
  });

  it('declares notificationTrim in the manifest as a boolean defaulting to true', () => {
    const manifest = JSON.parse(readFileSync(new URL('../.claude-plugin/plugin.json', import.meta.url), 'utf8'));
    expect(manifest.userConfig.notificationTrim).toMatchObject({ type: 'boolean', default: true });
  });

  it('asks nothing extra when the option is off (0.6.1 behaviour)', async () => {
    const seenOff: string[][] = [];
    const seenOmitted: string[][] = [];
    const off = await compactSession(transcriptWithReports(), cfg({ proseTrim: false, notificationTrim: false }), fetchWith(0.01, seenOff));
    const omitted = await compactSession(transcriptWithReports(), cfg({ proseTrim: false, notificationTrim: false }), fetchWith(0.01, seenOmitted));
    expect(strip(off)).toEqual(strip(omitted));
    expect(off.notifications).toBeUndefined();
    expect(seenOff).toEqual(seenOmitted);
    expect(notifRequests(seenOff)).toHaveLength(0);
  });

  it('asks nothing extra when tool compaction already reaches the minimum', async () => {
    const big = 'x'.repeat(20_000);
    const messages = [
      message('user', 'Fix the stale cache.'),
      call('rd1', 'Read', { file_path: 'src/a.ts' }, big),
      result('rd1', big),
      message('assistant', 'Dispatching a worker.'),
      message('user', notification('w1')),
      message('assistant', 'Done.'),
      message('user', 'ok'),
    ];
    const requests: string[][] = [];
    const tool = (key: string) => key.startsWith('result_') || key.startsWith('call_');
    const fetchFn = async (_u: string, init?: { body?: string }) => {
      const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
      requests.push(Object.keys(questions));
      const answers = Object.fromEntries(Object.keys(questions).map((k) => [k, { type: 'noul', noul: tool(k) ? 0.01 : 0.9 }]));
      return { status: 200, ok: true, text: JSON.stringify({ answers }) };
    };
    const out = await compactSession(messages, cfg({ notificationTrim: true }), fetchFn, memoryArchive().writer);
    expect(reductionRatioOf(out)).toBeGreaterThanOrEqual(0.25);
    expect(notifRequests(requests)).toHaveLength(0);
    expect(out.notifications).toBeUndefined();
  });

  it('scores every report in exactly one request and folds at the first rung that reaches the minimum', async () => {
    const requests: string[][] = [];
    const { writer } = memoryArchive();
    const out = await compactSession(transcriptWithReports(), cfg({ notificationTrim: true, archiveResults: true }), fetchWith(0.1, requests), writer);
    expect(notifRequests(requests)).toEqual([['n1', 'n2', 'n3']]);
    expect(out.notifications?.rung).toBe(0.15);
    expect(out.notifications?.stats).toMatchObject({ folded: 3, stripped: 0, requests: 1 });
    expect(reductionRatioOf(out)).toBeGreaterThanOrEqual(0.25);
    expect(textOf(out)).toContain('[jev-compaction folded');
    expect(textOf(out)).not.toContain('<usage>');
    expect(summarize(out.result, out.prose, out.notifications)).toMatch(
      /; notification trim at 0\.15: 3 folded, 0 stripped, \d+ chars$/,
    );
  });

  it('keeps the archived originals of the same task id from two compaction runs in separate files', async () => {
    const { files, writer } = memoryArchive();
    const run = async (variant: string) => {
      const messages = [
        message('user', 'Fix the stale cache.', { handle: 'n-0' }),
        ...['w1', 'w2', 'w3'].flatMap((id) => [
          message('assistant', 'Dispatching a worker.'),
          message('user', notification(id, `${REPORT}${variant}`)),
        ]),
        message('assistant', 'Done.'),
        message('user', 'ok'),
      ];
      const out = await compactSession(messages, cfg({ notificationTrim: true, archiveResults: true }), fetchWith(0.9), writer);
      expect(out.notifications?.rung).toBe('all');
      return out;
    };
    const first = await run('FIRST-RUN-MARK');
    const second = await run('SECOND-RUN-MARK');
    const w1Files = [...files.keys()].filter((p) => p.includes('notif-w1'));
    expect(w1Files).toHaveLength(2);
    for (const [out, mark] of [[first, 'FIRST-RUN-MARK'], [second, 'SECOND-RUN-MARK']] as const) {
      const cited = /full notification: (\S*notif-w1\S*) \(Read/.exec(textOf(out))?.[1];
      expect(cited).toBeDefined();
      expect(files.get(cited!)).toContain(mark);
    }
    expect(w1Files.map((p) => files.get(p)).join('')).not.toBe('');
    expect(new Set(w1Files.map((p) => files.get(p)!.includes('FIRST-RUN-MARK'))).size).toBe(2);
  });

  it('falls to the all rung when Jev keeps every report', async () => {
    const requests: string[][] = [];
    const out = await compactSession(transcriptWithReports(), cfg({ notificationTrim: true, archiveResults: true }), fetchWith(0.9, requests), memoryArchive().writer);
    expect(notifRequests(requests)).toHaveLength(1);
    expect(out.notifications?.rung).toBe('all');
    expect(out.notifications?.stats.folded).toBe(3);
    expect(summarize(out.result, out.prose, out.notifications)).toContain('; notification trim at all: 3 folded');
  });

  it('keeps its best partial result for later stages when no rung reaches the minimum', async () => {
    const plain = await compactSession(transcriptWithReports(), cfg({ proseTrim: false, notificationTrim: false, stepTrim: false }), fetchWith(0.9));
    const out = await compactSession(
      transcriptWithReports(),
      cfg({ notificationTrim: true, archiveResults: true, minReductionRatio: 0.99, stepTrim: false }),
      fetchWith(0.9),
      memoryArchive().writer,
    );
    expect(out.notifications).toMatchObject({ rung: 'all', stats: { requests: 1, folded: 3 } });
    expect(reductionRatioOf(out)).toBeLessThan(0.99);
    expect(out.result.stats.charsAfter).toBeLessThan(plain.result.stats.charsAfter);
  });

  it('keeps the tool-compaction result when the notification request fails', async () => {
    const plain = await compactSession(transcriptWithReports(), cfg({ proseTrim: false, notificationTrim: false }), fetchWith(0.9));
    const out = await compactSession(
      transcriptWithReports(),
      cfg({ notificationTrim: true, archiveResults: true, stepTrim: false }),
      fetchWith('fail'),
      memoryArchive().writer,
    );
    expect(out.notifications?.stats.failed).toBeTruthy();
    expect(out.notifications?.rung).toBeUndefined();
    expect(strip(out)).toEqual(strip(plain));
  });

  it('is inert, and asks nothing, when there is no archive writer', async () => {
    const requests: string[][] = [];
    const out = await compactSession(transcriptWithReports(), cfg({ notificationTrim: true }), fetchWith(0.1, requests));
    expect(notifRequests(requests)).toHaveLength(0);
    expect(out.notifications).toBeUndefined();
  });

  it('writes exactly the archive files its output cites', async () => {
    const { files, writer } = memoryArchive();
    const out = await compactSession(transcriptWithReports(), cfg({ notificationTrim: true, archiveResults: true }), fetchWith(0.1), writer);
    const cited = [...textOf(out).matchAll(/full notification: (\S+) \(Read or grep it if needed\)/g)].map((m) => m[1]);
    const written = [...files.keys()].filter((p) => !p.endsWith('.gitignore') && !p.endsWith('INDEX.md'));
    expect(cited).toHaveLength(3);
    expect(written.sort()).toEqual([...cited].sort());
    const original = notification('w2');
    expect(files.get(cited.find((p) => /notif-w2-[0-9a-f]{8}\.txt$/.test(p ?? '')) ?? '')).toBe(original);
  });

  it('writes exactly the notification archives its partial result cites when no rung reaches the minimum', async () => {
    const { files, writer } = memoryArchive();
    const out = await compactSession(
      transcriptWithReports(),
      cfg({ notificationTrim: true, archiveResults: true, minReductionRatio: 0.99, stepTrim: false }),
      fetchWith(0.1),
      writer,
    );
    const cited = [...textOf(out).matchAll(/full notification: (\S+) \(Read or grep it if needed\)/g)].map((m) => m[1]);
    expect([...files.keys()].filter((p) => p.includes('notif-')).sort()).toEqual([...cited].sort());
  });

  it('redacts a key-like string in a report, its summary and its preview before the request leaves', async () => {
    const bodies: string[] = [];
    const leaky = [
      message('user', 'Fix the stale cache.'),
      message('assistant', 'Dispatching a worker.'),
      message('user', notification('w1', `${REPORT} key ${SECRET} ${REPORT}`, `Worker used ${SECRET}`)),
      message('assistant', 'Dispatching a worker.'),
      message('user', notification('w2')),
      message('assistant', 'Done.'),
      message('user', 'ok'),
    ];
    const { files, writer } = memoryArchive();
    await compactSession(leaky, cfg({ notificationTrim: true, archiveResults: true }), fetchWith(0.1, [], 0.9, bodies), writer);
    expect(bodies.some((b) => b.includes('"n1"'))).toBe(true);
    for (const body of bodies) expect(body).not.toContain(SECRET);
    for (const text of files.values()) expect(text).not.toContain(SECRET);
  });

  it('composes with proseTrim by identity: a message prose drops does not shift the folded reports', async () => {
    const FLUFF = 'Let me have a good look at the layout of this module now.';
    const extra = [message('assistant', Array(10).fill(FLUFF).join(' '))];
    const proseOnly = await compactSession(
      transcriptWithReports(extra),
      cfg({ proseTrim: true, notificationTrim: false, injectTrim: false, replyTrim: false, archiveResults: true, minReductionRatio: 0.01 }),
      fetchWith(0.1, [], 0.1),
      memoryArchive().writer,
    );
    const both = await compactSession(
      transcriptWithReports(extra),
      cfg({ proseTrim: true, notificationTrim: true, archiveResults: true, injectTrim: false, replyTrim: false }),
      fetchWith(0.1, [], 0.1),
      memoryArchive().writer,
    );
    expect(proseOnly.prose?.cutoff).toBe(0.15);
    expect(proseOnly.messages.some((m) => m.text.includes('of this reply; full text'))).toBe(false);
    expect(both.prose?.cutoff).toBe(0.15);
    expect(both.notifications?.rung).toBe(0.15);
    expect(both.messages).toHaveLength(proseOnly.messages.length);
    expect(both.messages.some((m) => m.text.startsWith(FLUFF))).toBe(false);
    expect(both.messages.filter((m) => m.text.includes('[jev-compaction folded'))).toHaveLength(3);
    expect(both.messages.filter((m) => m.text.startsWith('<task-notification>'))).toHaveLength(3);
    expect(both.messages.map((m) => m.role)).toEqual(proseOnly.messages.map((m) => m.role));
  });

  describe('through the hook', () => {
    const hook = (
      options: Record<string, unknown>,
      fetchFn = fetchWith(0.1),
      logs: string[] = [],
      builtin: (e: unknown) => Promise<unknown> = async (e) => e,
    ) => {
      const hooks: Record<string, (...args: any[]) => Promise<any>> = {};
      (register as any)(
        (event: string, a: unknown, b?: any) => {
          hooks[event] = typeof b === 'function' ? b : (a as any);
        },
        { apiKey: 'k', preserveRecentMessages: 2, ...options },
      );
      const toasts: string[] = [];
      const sinks: (string | undefined)[] = [];
      const files = new Map<string, string>();
      const $ = {
        clock,
        http: { fetch: async (url: string, init?: { body?: string }) => fetchFn(url, init) },
        ui: {
          log: (t: string, o?: { to?: string }) => {
            logs.push(t);
            sinks.push(o?.to);
          },
          toast: (t: string) => toasts.push(t),
        },
        fs: {
          exists: async (p: string) => files.has(p),
          write: async (p: string, t: string) => void files.set(p, t),
        },
      };
      const next = Object.assign(builtin, {
        signal: new AbortController().signal,
        budget: { ms: 10_000, remainingMs: 10_000 },
      });
      return { run: (event: Record<string, unknown>) => hooks['session.compact']!($, event, next), toasts, files, sinks, get notices() { return logs.filter((l) => NOTICE.test(l)); } };
    };

    it('replaces history and names the rung in the notice', async () => {
      const event = { trigger: 'auto', messages: transcriptWithReports() };
      const on = hook({ notificationTrim: true });
      const out = await on.run(event);
      expect(Object.keys(out)).toEqual(['messages', 'tokensBefore', 'tokensAfter']);
      expect(out.messages).toHaveLength(event.messages.length);
      expect(on.notices[0]).toMatch(
        /^kept \d+\/\d+ messages, no summary \(\d+% reduction; .*; notification trim at 0\.15: 3 folded, 0 stripped, \d+ chars\)$/,
      );
    });

    it('keeps diagnostics out of the transcript: every ui.log goes to the debug log', async () => {
      const on = hook({ notificationTrim: true });
      await on.run({ trigger: 'auto', messages: transcriptWithReports() });
      expect(on.sinks.length).toBeGreaterThan(0);
      expect(on.sinks.every((to) => to === 'debug')).toBe(true);
      expect(on.toasts).toHaveLength(1);
    });

    it('keeps its own shorter cut when the built-in summary then fails, calling it once', async () => {
      let calls = 0;
      const failing = async () => {
        calls++;
        throw new Error('summary failed');
      };
      for (const hybridSummary of [false, true]) {
        calls = 0;
        const event = { trigger: 'auto', messages: transcriptWithReports() };
        const on = hook({ notificationTrim: true, proseTrim: false, minReductionRatio: 0.99, hybridSummary }, fetchWith(0.1), [], failing);
        const out = await on.run(event);
        expect(calls).toBe(1);
        expect(Object.keys(out)).toEqual(['messages', 'tokensBefore', 'tokensAfter']);
        expect(out.tokensAfter).toBeLessThan(out.tokensBefore);
        expect(on.notices.at(-1)).toMatch(/^built-in summary failed, kept the Jev cut/);
      }
    });

    it('falls back, saying notification trim was tried, when no rung reaches the minimum', async () => {
      const event = { trigger: 'auto', messages: transcriptWithReports() };
      const on = hook({ notificationTrim: true, proseTrim: false, minReductionRatio: 0.99 }, fetchWith(0.9));
      expect(await on.run(event)).toBe(event);
      expect(on.notices[0]).toMatch(/^fallback to built-in summary \(below 99% minimum, notification trim tried, step trim tried: /);
      const never = hook({ notificationTrim: true, proseTrim: false, minReductionRatio: 0.99, builtinFallback: 'never' }, fetchWith(0.9));
      // No summary may run, so the cut Jev did make is kept rather than nothing.
      const kept = await never.run(event);
      expect(Object.keys(kept)).toEqual(['messages', 'tokensBefore', 'tokensAfter']);
      expect(kept.tokensAfter).toBeLessThan(kept.tokensBefore);
      expect(never.notices[0]).toContain('kept the cut below the minimum (99% minimum, notification trim tried, step trim tried, builtin fallback disabled');
      expect(never.toasts).toEqual([expect.stringMatching(/^Compacted with Jev: \d+ → \d+ messages, frees \d+% of the context \(below the 99% target\)$/)]);
      const both = hook({ notificationTrim: true, proseTrim: true, minReductionRatio: 0.99 }, fetchWith(0.9));
      expect(await both.run(event)).toBe(event);
      expect(both.notices[0]).toContain('minimum, prose trim tried, notification trim tried, step trim tried: ');
    });

    it('says nothing about notification trim when the option is off', async () => {
      const event = { trigger: 'auto', messages: transcriptWithReports() };
      const off = hook({ notificationTrim: false, minReductionRatio: 0.99 }, fetchWith(0.9));
      expect(await off.run(event)).toBe(event);
      expect(off.notices[0]).not.toContain('notification');
    });

    it('logs once that archiveResults off makes the option inert, and leaves the history alone', async () => {
      const logs: string[] = [];
      const event = { trigger: 'auto', messages: transcriptWithReports() };
      const requests: string[][] = [];
      const on = hook({ notificationTrim: true, archiveResults: false }, fetchWith(0.1, requests), logs);
      await on.run(event);
      await on.run(event);
      expect(logs.filter((l) => l === 'notification trim skipped: archiveResults is off')).toHaveLength(1);
      expect(notifRequests(requests)).toHaveLength(0);
    });
  });
});

describe('sha256Hex', () => {
  it('matches node:crypto, including across block boundaries and non-ASCII text', async () => {
    const { createHash } = await import('node:crypto');
    for (const text of ['', 'abc', 'a'.repeat(55), 'a'.repeat(56), 'a'.repeat(64), 'a'.repeat(1000), 'héllo 😀 wörld'.repeat(9)]) {
      expect(sha256Hex(text)).toBe(createHash('sha256').update(text).digest('hex'));
    }
  });
});

describe('messageTarget', () => {
  it('scales the minimum by the share of the context the history fills', () => {
    expect(messageTarget(160_000, 80_000, 0.25)).toBeCloseTo(0.5);
  });
  it('keeps the minimum without a reading or when the history fills the context', () => {
    expect(messageTarget(undefined, 80_000, 0.25)).toBe(0.25);
    expect(messageTarget(80_000, 80_000, 0.25)).toBe(0.25);
    expect(messageTarget(70_000, 80_000, 0.25)).toBe(0.25);
    expect(messageTarget(160_000, 0, 0.25)).toBe(0.25);
  });
});
