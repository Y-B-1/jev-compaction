import { describe, expect, it } from 'vitest';
import { resolveOptions } from '../src/compact.js';
import { MARKER, ownText } from '../src/state.js';
import { mildSteps, planSteps, scoreSteps, stepLadder } from '../src/steps.js';
import type { CompactResult, JevAsker, Message } from '../src/types.js';

const user = (text: string): Message => ({ role: 'user', text, toolUses: [] });
const reply = (text: string): Message => ({ role: 'assistant', text, toolUses: [] });
const call = (id: string, text = ''): Message => ({ role: 'assistant', text, toolUses: [{ tool_use_id: id, tool: 'Bash', input: { command: `run ${id}` } }] });
const result = (id: string, text: string): Message => ({ role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: id, text, isError: false }] });

const BIG = 'build output line that the task already acted on\n'.repeat(60);
const OLD = `${MARKER}archived 2 rows of earlier work, 1 tool call (Bash), 3000 chars; full text: arch/old.txt (Read it if needed)]\nNote: ${MARKER}archived 2 rows of earlier work, 1 tool call (Bash), 900 chars; full text: arch/older.txt (Read…`;

const options = resolveOptions({ goal: 'add tests', preserveRecentMessages: 2 });
const asker = (n: number): JevAsker =>
  ({ ask: async (_s: unknown, q: Record<string, unknown>) => ({ answers: Object.fromEntries(Object.keys(q).map((k) => [k, { noul: n }])) }) }) as unknown as JevAsker;
const toolResult = (messages: Message[], scale = 1): CompactResult => {
  const chars = messages.reduce((sum, m) => sum + m.text.length + (m.toolResults ?? []).reduce((a, r) => a + r.text.length, 0) + m.toolUses.length * 30, 0);
  return { messages, decisions: [], stats: { charsBefore: chars * scale, charsAfter: chars * scale } } as unknown as CompactResult;
};
const markers = (messages: readonly Message[]) => messages.filter((m) => m.text.includes(MARKER));
const count = (text: string, part: string) => text.split(part).length - 1;

/** Two runs of adjacent finished work before the last prompt, then the current task. */
function session(): Message[] {
  return [
    user('start'),
    user('build the parser'),
    call('a'), result('a', BIG),
    call('b'), result('b', BIG),
    reply(`Built the parser. ${'Details of the build. '.repeat(60)}`),
    user('now add the tests'),
    call('c'), result('c', BIG),
    call('d'), result('d', BIG),
    user('and run them'),
    call('e'), result('e', BIG),
    call('f', 'Running the final check.'), result('f', 'pass'),
  ];
}

describe('marker rows', () => {
  it('never quotes an earlier marker in a new note', async () => {
    const messages = session();
    messages[4] = call('b', OLD);
    const scores = await scoreSteps(messages, asker(0.01), options);
    for (const plan of planSteps(scores, (id) => `arch/${id}.txt`)) expect(count(plan.row.text, MARKER)).toBe(1);
    expect(ownText(`${OLD}\nThe real reply.`)).toBe('The real reply.');
    // A reply that only quotes the marker string keeps all its lines.
    const quoting = 'The row reads `[jev-compaction archived 2 rows]` and I will fix it.\nNote: the cache key must not change.\nDone.';
    expect(ownText(quoting)).toBe(quoting);
  });

  it('archives a run of adjacent steps as one marker row and one file', async () => {
    const messages = session();
    const scores = await scoreSteps(messages, asker(0.01), options);
    const out = stepLadder(scores, toolResult(messages), 0.2, (id) => `arch/${id}.txt`)!;
    expect(out.pass).toBe('trim');
    expect(markers(out.result.messages)).toHaveLength(2);
    expect(out.applied.archived).toHaveLength(2);
    expect(out.applied.archived[0]!.text).toContain('run a');
    expect(out.applied.archived[0]!.text).toContain('Built the parser');
    for (const row of markers(out.result.messages)) expect(count(row.text, 'full text: ')).toBe(1);
  });

  it('folds an earlier marker row between two archived steps into the run', async () => {
    const messages = session();
    messages.splice(4, 0, reply(OLD));
    const scores = await scoreSteps(messages, asker(0.01), options);
    const out = stepLadder(scores, toolResult(messages), 0.2, (id) => `arch/${id}.txt`)!;
    expect(markers(out.result.messages)).toHaveLength(2);
    expect(out.applied.archived[0]!.text).toContain('arch/old.txt');
  });

  it('skips the reply trim when it would free less than 5% of the context', async () => {
    const messages = session();
    const scores = await scoreSteps(messages, asker(0.01), options);
    expect(mildSteps(scores, toolResult(messages), (id) => id)).toBeDefined();
    expect(mildSteps(scores, toolResult(messages, 100), (id) => id)).toBeUndefined();
  });
});
