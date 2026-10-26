# Jev compaction plugin

`jev-compaction` (named fast-jev-compaction up to 0.18.0). A fork of [tamaratran/fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction) with context-relative minimums, a result cap, input trim and a hybrid rescue (built-in summary of the older history only). See the version history in `git log`.

Claude Code plugin for Jev-guided selective compaction. It replaces the
compaction summary with Jev decisions: every tool call and result is scored in
one fast request, stale ones are dropped or truncated, everything kept stays
verbatim. Also usable as an npm library.

### Where it runs

It needs Claude Code's `session.compact` function hook, the one place a plugin
can hand back its own history instead of a summary. So it runs wherever Claude
Code itself runs: the `claude` CLI, the IDE extensions, and apps that drive
Claude Code underneath, such as [Orca](https://github.com/stablyai/orca) (which
starts `claude` in its terminals and uses the Claude Agent SDK) and
[T3 Code](https://github.com/pingdotgg/t3code) (Claude Agent SDK, loading your
user settings). In those apps, put the two variables below in the `env` block of
`~/.claude/settings.json` rather than your shell, since the app starts Claude
Code with its own environment.

It does not run in Cursor or Codex. Cursor's `preCompact` hook only observes a
compaction (its only output is a message to the user), and stock Codex's
`PreCompact`/`PostCompact` hooks cannot return a history either; replacing it
there takes a patched Codex runtime, which this repository dropped in 0.14.0.

## What and why

Most context compaction asks an LLM to summarize old turns. A summary is
lossy: a file path, exact error, constraint, or command can disappear even when
it matters later. This library never rewrites anything. It only deletes tool
calls and tool results Jev says are no longer needed, and it asks Jev while
showing it the whole conversation. User and assistant text stays verbatim and
in order. The Claude Code plugin adds later stages for when that is not enough:
they move whole old steps or exchanges, assistant text included, to archive
files and leave one marker row per run naming the file. What you typed always
stays verbatim.

The repository is both an npm package (`src/`) and a Claude Code plugin
(`hooks/`, `.claude-plugin/`) that uses the package to replace Claude Code's
built-in compaction summary with the original messages.

## How it works

1. Every `tool_use` is paired with its `tool_result` by `tool_use_id`. Calls in
   the first message or in the newest `preserveRecentMessages` messages are
   pinned and never touched.
   Before anything is asked, exact repeats are settled for free
   (`dedupeRepeats`): an earlier copy of a repeated `<system-reminder>` or
   `<local-command-stdout>` block, or of a long repeated user message such as
   a skill loaded twice, becomes a one-line marker, and the result of an
   earlier call with the same tool and input as a later one is truncated (the
   latest copy always stays; a user prompt Claude Code still holds whole is
   never rewritten, since it may carry images the hook cannot see). Results under `minResultChars` (500) are kept
   without a question, and results an earlier compaction already truncated
   are kept as they are, so their archive is never overwritten.
2. The **state** sent to Jev is the whole conversation so far, oldest first,
   with every tool result replaced by a short note (`ok, 4213 chars (omitted)`).
   Tool inputs are included, texts are included, nothing is summarized.
3. The state is fitted into `maxStateTokens` (25k by default) in stages, each
   applied only if the previous one was not enough: tool inputs truncated to
   1000, then 200, then 60 characters; long texts abridged to head + tail,
   oldest non-pinned messages first; old non-pinned messages collapsed to a
   `[… N chars omitted …]` note; old tool calls reduced to one line each
   (`t12 Read file_path=src/a.ts → ok 480ch`); old call-less messages left
   out; runs of old call-only messages folded into one entry. If it still
   does not fit, compaction throws. Tokens are estimated without a tokenizer (a
   word per six letters, half a token per digit, ~one per other symbol),
   calibrated to land a little above the counts Jev reports.
4. For every non-pinned call Jev gets two `noul` questions: should the **call**
   stay (knowing it was made, with its input, still matters), and should the
   **result** stay verbatim (its contents are still needed and re-running the
   tool would not do).
5. Questions are split into as many requests as needed so state plus questions
   stays under `maxRequestTokens` (60k by default, under Jev's 64k per-request
   limit). Jev's second limit — state plus the single longest question under
   32k — is checked separately. The same full state is resent with every
   request, so fewer requests is strictly cheaper; requests run concurrently
   and their answers are merged.
6. Decisions per call. A Noul answer of 0.5 is Jev saying it is unsure, and
   distance from 0.5 is the confidence signal, so both thresholds are
   deliberately low: deleting context cannot be undone, and an uncertain
   answer should keep. The result is gated by `keepThreshold`, the call by its
   own, much lower `keepCallThreshold` — the two are independent Nouls on
   different scales, and a result runs to thousands of characters while its
   call is a few dozen, so losing the call for the same confidence as the
   result erases the record that the work happened for savings that round to
   zero. The decisions are:
   - `keepResult ≥ keepThreshold` → keep call and result;
   - else `keepCall ≥ keepCallThreshold` → keep the call, truncate the result
     to its first `truncateHeadChars` characters plus a one-line note;
   - else → remove the call together with its result.
7. The message list is rebuilt: a message that loses all its content is
   removed, untouched messages are returned as the same objects, and no result
   is ever left without its call. With `archiveResults` on (the default), the
   full text a call or result loses is first written to
   `.claude/jev-compaction/<tool_use_id>.txt` at the project root (the git
   toplevel of the session's directory, else the session root; a `.gitignore`
   there keeps the directory out of version control), and the marker left in its place
   cites the file so it can be `Read` or grepped back later. Text that looks
   like a credential is never archived.

8. The Claude Code hook hands the result back **resume-safe**
   (`stripHandles`, on by default): tool-result messages and every assistant
   message but the newest response go back without the engine's `handle`, and the
   rows of one parallel tool-call response (and its text-less result rows)
   are merged into one message. A kept message with its handle makes `--resume` restore
   the history the compaction removed (anthropics/claude-code#95328), and a
   handle-less row inside a parallel group splits it, after which Claude Code
   reports a sibling's result as missing. The cost: hidden reasoning of the
   older messages is dropped. Images and documents (PDFs) are removed too, as
   Claude Code's own compaction removes them: tool-result ones, and pasted ones
   before the current prompt (whose own stay). Each leaves a one-line
   placeholder: the file name, the model's first sentence about it then, the
   path to Read it again, and an archived base64 copy listed in `INDEX.md`
   (when `archiveResults` is on). Later cuts of the result keep the placeholder
   whole. The toast counts what was removed.

Jev failures, malformed answers, a missing key, a history that cannot be
fitted, or answers that drop every call they score (at least
`minKeepSignalCalls`, default 3 — Jev answering 0 to everything is a failure,
not a verdict) throw; the caller (or the Claude Code hook) decides what to fall back to.

## Install and usage

```sh
npm install jev-compaction
export TYPESAFE_API_KEY=...
```

```ts
import { compactMessages, reductionRatio, type Message } from 'jev-compaction';

const transcript: Message[] = [
  { role: 'user', text: 'Fix the failing test. Never edit src/generated.', toolUses: [] },
  {
    role: 'assistant',
    text: '',
    toolUses: [{ tool_use_id: 'toolu_1', tool: 'Read', input: { file_path: 'src/a.ts' } }],
  },
  { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'toolu_1', text: '…file…' }] },
  // …
];

const result = await compactMessages(transcript, { preserveRecentMessages: 4 });
console.log(result.messages, result.decisions, result.stats);
if (reductionRatio(result) < 0.25) {
  // not worth it: keep the original transcript, or summarize instead
}
```

`Message` is a subset of Claude Code's `SessionMessage`, so a session transcript
can be passed in as is.

To bring your own transport, implement `JevAsker` (one `ask(state, questions)`
method) and call `compact(messages, asker, options)`; `buildJevRequest` and
`parseJevResponse` give you the HTTP request body and response validation.
The building blocks (`collectToolCalls`, `fitState`, `batchCalls`,
`decideCall`, `applyDecisions`) are exported too.

`apiKey` defaults to `process.env.TYPESAFE_API_KEY`. Never commit the key or
put it in a source file.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `apiKey` | `TYPESAFE_API_KEY` | TypeSafe API key (`compactMessages`/`JevClient`) |
| `model` | `jev-latest` | Jev model name |
| `baseUrl` | `https://api.typesafe.ai/v1/systemone` | System One endpoint |
| `fetch` | native `fetch` | Injectable fetch implementation for tests |
| `goal` | last 3 user prompts | Ongoing task description included in the state |
| `keepThreshold` | `0.15` | Minimum Jev probability for a tool *result* to stay verbatim |
| `keepCallThreshold` | `0.05` | Minimum Jev probability for the tool *call* to stay; below it the call goes with its result |
| `preserveRecentMessages` | `6` | Newest messages never touched (the first is always kept) |
| `maxStateTokens` | `25000` | Estimated token ceiling for the state |
| `maxRequestTokens` | `60000` | Estimated ceiling for state plus one batch of questions (Jev allows 64k) |
| `truncateHeadChars` | `150` | Characters of a dropped tool result retained before its note |
| `archiveResults` | `true` | Write the full text of a dropped tool call or result to `.claude/jev-compaction/` and cite the file in the marker left in its place, so it can be `Read` or grepped back later; `INDEX.md` there lists every archived file with how it starts, so one grep finds the right one; text that looks like a credential is never written. A truncated result that failed also keeps its last 800 characters |
| `resultPreviewChars` | `300` | Characters from each end of a tool result quoted in its keep question; `0` disables |
| `salvageChars` | `150` | Characters of identifiers and error lines recovered from the dropped part of a tool result and kept next to its truncation note; `0` disables |
| `proseTrim` | `true` | Plugin hook only. On by default (set `false` to turn it off). Rescue: when tool compaction cuts less than `minReductionRatio`, Jev scores the clauses of old assistant text and prose tool inputs once, then drops the filler at the first fixed keep cut-off (`0.05`, `0.15`, `0.35`) that reaches the minimum, and cuts old code bodies in tool inputs. User messages, pinned messages and protected spans (code, paths, identifiers, numbers) are never trimmed. If no cut-off reaches the minimum, the built-in fallback runs as before |
| `notificationTrim` | `true` | Plugin hook only. On by default (set `false` to turn it off; it needs `archiveResults`, and does nothing while that is off). Rescue, run only when tool compaction cuts less than `minReductionRatio` and `archiveResults` is on: removes the `<usage>` and `<diagnostics>` of finished background-worker reports (`<task-notification>` messages), and folds the reports Jev scores as stale to a `truncateHeadChars` head, a marker and the errors, paths, urls, shas and names from the rest, at the first keep cut-off (`0.05`, `0.15`, `0.35`, then every report) that reaches the minimum. Each original is archived whole under `.claude/jev-compaction/notif-<task-id>.txt` and cited in the marker. Nothing is reworded; bare numbers in a folded tail are recoverable only from the archive. Costs one extra Jev request on a shortfall. User text, pinned messages and reports that are not `completed` are never touched |
| `builtinFallback` | `auto` | Plugin hook only. When Jev cannot compact (unreachable, or it cannot free the minimum): `auto` falls back to Claude Code's summary for automatic compaction and leaves a manual `/compact` as it is; `always` falls back for `/compact` too; `never` never summarizes |
| `minResultChars` | `500` | Tool results shorter than this are always kept and never asked about |
| `maxKeptResultChars` | `4000` | A result outside the newest messages that Jev keeps whole is still cut to about this many characters: its head and tail (half of each for an error, otherwise three quarters head) with the errors and identifiers of the middle, archived and cited; a capped result is never capped again; `0` disables |
| `hybridSummary` | `true` | Plugin hook only. When trimming frees less than the minimum, the built-in summarizer summarizes only the older history and the recent turns (at most 30% of the history) stay as trimmed, instead of a summary of everything; see `hooks/README.md` |
| `inputTrim` | `true` | Plugin hook only. On every compaction, cut the code bodies of old tool inputs (Write/Edit contents, notebook sources, workflow scripts, Bash heredocs) to a `truncateHeadChars` head and a note, by rule with no Jev request, archiving the full text when `archiveResults` is on. Prose inputs (Agent prompts), pinned messages and dropped calls are not touched |
| `injectTrim` | `true` | Plugin hook only; needs `archiveResults`. When trimming falls short, fold old injected content in user rows (skill bodies, browser element selections, pastes, and other user rows of 3,000+ chars that are not among the last three prompts) to a head, a marker naming the archived original, and a salvage block. Jev decides which blocks stay; at the last rung every old skill body and element selection folds. Text typed around a block (a skill's `ARGUMENTS:`, the request after an element) stays verbatim. |
| `replyTrim` | `true` | Plugin hook only; needs `archiveResults`. On every compaction, archive whole old steps (a reply, a closed run of tool calls with their results, or a row of host messages) that come before your last prompt and that Jev scores below `0.05`. Adjacent archived steps become one row naming one archive file and quoting, on one line, the start of the last thing the assistant wrote there (never an earlier marker). Runs only when it frees at least 5% of the context. No reply is cut mid-sentence. |
| `exchangeDrop` | `true` | Plugin hook only; needs `archiveResults`. When every trim stage falls short, Jev scores each old finished exchange (a prompt and its work up to the next prompt). An exchange it no longer needs is archived whole and its work rows become one row naming the file; dropped exchanges in a row keep their prompts, one after another, and share one row and one file. The prompt stays verbatim; the last three prompts, pinned rows, exchanges with an unpaired tool call and exchanges holding a typed `!` command, slash-command arguments or a message sent mid-turn are never offered. Runs before the built-in summary. |
| `stepTrim` | `true` | Plugin hook only; needs `archiveResults`. The last trim stage, run on the best result of the stages before it. First it archives old steps before your last prompt at the keep cut-offs `0.05`, `0.15`, `0.35`; if that falls short it archives the lowest-scored steps, the current task included, until the minimum is met. Adjacent archived steps, and earlier marker rows just before or between them, share one marker row and one archive file. Your prompt text, row 0, pinned rows, typed `!` commands, slash-command arguments, messages sent mid-turn and unpaired tool calls are never touched. The built-in summary then runs only when Jev fails or nothing is left to archive. Last, when an archive is on, a chain of note-only rows that only prompts separate becomes one note after the last prompt; its file holds the original notes, each naming its own file. |
| `dedupeRepeats` | `true` | Remove repeats without asking Jev: an earlier call a later one repeats (the same Bash command whatever its description or timeout, a Read of a file a later whole-file Read covers, otherwise the same tool and input) gets its result truncated when the later output did not fail and is at least half as long, and earlier copies of repeated injected blocks or long repeated user messages become a marker; a user prompt Claude Code still holds whole is never rewritten, since it may carry images the hook cannot see |
| `minKeepSignalCalls` | `3` | When Jev scores at least this many calls and drops them all, call and result, the answers count as a failure and the fallback rules apply; `0` disables |
| `stripHandles` | `true` | Plugin hook only. Return older messages without the engine's handle and merge parallel tool-call rows, so `--resume` keeps the compaction and parallel calls stay paired; costs the hidden reasoning of those messages, and replaces their images and documents with a placeholder (path, description, archived copy) |
| `jevPruneBashOutput` | `false` | Opt-in: before a Bash tool result reaches the transcript, trim its stdout through Jev when it is long (`src/output.ts`). A second, separate Jev call site from compaction — leave it off unless you want the extra cost |
| `jevPruneMinChars` | `4000` | Stdout shorter than this is left untouched even when `jevPruneBashOutput` is on |

`result.stats` reports message and character counts before and after, the
per-reason decision counts, the state size in estimated tokens, which fitting
stage was needed, and the number of requests.

## Limitations

- In the library, only tool calls and results are candidates; text messages
  are never removed or shortened in the output (they are only abridged in the
  state Jev sees). The plugin's later stages can archive whole old assistant
  steps (see `replyTrim`, `exchangeDrop` and `stepTrim`).
- Token sizes are estimates from character counts, not a tokenizer.
- Calibration is at the request level; a probability is not a proof that a
  result is safe to delete. The assistant can always re-run the tool.
- The full state is repeated with every request, and Jev prefills it each time,
  so the request count dominates the cost. Lowering `maxStateTokens` both
  shrinks each prefill and leaves more of the 64k budget for questions.

## Claude Code plugin

The repository root is a Claude Code function-hook plugin: `hooks/fast-jev.ts`
is a thin adapter that feeds `session.compact` transcripts through `src/` and
falls back to Claude Code's built-in summary on errors or insufficient
reduction. See [`hooks/README.md`](hooks/README.md) for configuration and the
Claude Code 2.1.274 type reference.

### Install in Claude Code

Function hooks are an early-access Claude Code feature (2.1.274+), so the
opt-in flag must be set wherever Claude Code runs, e.g. in `~/.claude/settings.json`:

```json
{ "env": { "CLAUDE_CODE_ENABLE_FUNCTION_HOOKS": "1", "TYPESAFE_API_KEY": "<your key>" } }
```

Then add this repository as a plugin marketplace and install the plugin,
either from the shell or as slash commands inside a session:

```sh
claude plugin marketplace add Y-B-1/jev-compaction
claude plugin install jev-compaction@jev-compaction
```

**Upgrading from fast-jev-compaction (0.18.0 and earlier).** The plugin, its
marketplace and the repository were renamed to `jev-compaction` in 0.19.0. Remove
the old marketplace (`claude plugin marketplace remove fast-jev-compaction`), then
run the two commands above. Plugin options saved under `pluginConfigs.fast-jev-compaction`
move to `pluginConfigs.jev-compaction`. New archives go to `.claude/jev-compaction/`;
markers that older versions left in a history, and their files in
`.claude/fast-jev-compaction/`, are still recognized and still readable.

The install prompts for the plugin options (API key, thresholds, `truncateHeadChars`,
…); leave them at their defaults to use `TYPESAFE_API_KEY` from the environment.
Restart Claude Code or run `/reload-plugins`. From then on `/compact` (and
auto-compaction) goes through Jev. The plugin never starts a compaction itself; it
runs whenever Claude Code compacts and shows one short toast, nothing in the chat:
`Compacted with Jev: N → M messages, frees X% of the context, no summary` when the pruned
history replaced the built-in summary, or `Using Claude's summary: …` when Jev
could not remove enough (short sessions) or failed. A manual `/compact` that Jev
cannot serve is left as it is (`Not compacted: …`) unless `builtinFallback` is
`always`. The full detail (context size, per-stage counts, per-call decisions)
goes to Claude Code's debug log (`claude --debug`), under `[jev-compaction]`,
and `/jev` prints the last compaction's outcome and detail and the settings in use
(key found, endpoint, model, `builtinFallback`) without the debug log.

Subagents Claude Code starts (the Task tool) compact through the same hook, each
on its own transcript. An agent of another tool (a Codex or Cursor worker an app
starts beside Claude) compacts with that tool's own method; what it hands back to
the Claude session is ordinary tool output there, which this plugin then scores
like any other.

`TYPESAFE_BASE_URL` (environment or the settings' `env` block) replaces TypeSafe's
endpoint with another one that speaks the same API, such as a staging host or a
local test server.

Claude Code loads plugin hook modules once a startup feature flag answers, at the
start of a turn: a `/compact` typed as the very first thing in a freshly started
process can run before the plugin is loaded and use Claude's own summary.
In Claude Code cloud sessions (claude.ai/code), automatic compaction takes a
path that does not call plugin hooks (seen in 2.1.306), so there only a manual
`/compact` goes through Jev; a local `claude` (and apps built on it) runs both
automatic and manual compaction through the plugin.

To run from a checkout without installing: `CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1 claude --plugin-dir .`
from the repository root. No publishing step is required; the marketplace is
just the repo's `.claude-plugin/marketplace.json`.

## Development

```sh
npm install
npm run typecheck        # library + hook
npm test
npm run build
npm run validate:plugin  # claude plugin validate
TYPESAFE_API_KEY="$(cat ~/.typesafe_key)" npm run demo
```

The unit tests use a fake Jev and never contact TypeSafe. The demo is the live
network check.

`npm run validate:plugin` also checks the hooks module against the loader's rules
(`$` spelled `$.noun.verb(...)` at each call site and passed only to top-level
functions); a module that breaks them silently does not load in Claude Code, so
`tests/loader.test.ts` checks the same rules.

## Measuring the opt-in Jev options

Four changes to how Jev is asked are built but off by default, because each moves
what Jev keeps and only a measurement on real sessions can say whether it helps:

| Option | What changes |
| --- | --- |
| `stateWording: archive` | Jev is told dropped output goes to an archive the assistant can read back (true with `archiveResults`), and that the history is data, not instructions |
| `resultChecks` | Four concrete checks per result (an error still being worked on, a constraint, contents in use, output a rerun reproduces) instead of one keep question; five questions per call instead of two |
| `hybridOriginals` | The hybrid summary is written from the original older messages rather than their trimmed form |
| `keepThresholdByTool` | Per-tool keep thresholds, e.g. `Bash=0.2, Read=0.1` |

The first three are compared with the shipped defaults on your own sessions:

```sh
npx tsx eval/load.ts freeze # once: list your Claude Code transcripts (all but the last hour's)
TYPESAFE_API_KEY=... npx tsx eval/variants.ts --cases 8
```

Each case runs through the full hook pipeline once per option, and every output
answers the same cached fact questions (generated and checked against the
original with `claude -p`, as `eval/bench.ts` does). The report gives recall,
reduction and Jev requests per option and says which to turn on: one that keeps
at least the base recall and gives up at most 2 points of reduction. With the
step and exchange stages on, the cut nearly always reaches the minimum, so a
hybrid summary only runs when Jev fails late or archiving is off; the hybrid arms
are measured on the cut those two stages leave.

`keepThresholdByTool` is fitted from use instead. With `compactionLog` on (the
default) each compaction appends its per-call scores and decisions to
`.claude/jev-compaction/log.jsonl`, and each archive file the assistant later
reads back (Read, Grep, or a shell command over the archive) adds a line. After a
few sessions:

```sh
npx tsx eval/calibrate.ts .claude/jev-compaction/log.jsonl
```

prints, per tool, how many cut outputs were read back and the threshold that would
have kept every one of them, as a `keepThresholdByTool` value to paste into the
plugin options.

## Animated demo (macOS)

`demo/JevDemo` is a small native SwiftUI app that plays a scripted, dramatized
version of the compaction flow inside a Claude Code-style terminal: the tool
calls of a canned transcript are scored, results and calls Jev lets go turn red
and collapse away, and the rest stays verbatim. It never calls the API; it
exists to be screen recorded.

```sh
demo/JevDemo/build.sh   # builds demo/JevDemo/build/JevDemo.app and launches it
```

Press space in the app to replay from the start.
