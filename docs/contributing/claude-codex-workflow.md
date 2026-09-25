# Claude–Codex Collaboration Workflow

> Applies when Claude Code (Opus 5) drives development in this repository with
> Codex reachable — either through the `codex` MCP server or the `codex` CLI on
> PATH. If neither is available, stop and report rather than silently taking
> substantial implementation on yourself.

## Role Split

| Agent | Model | Responsibility |
| ----- | ----- | -------------- |
| Claude | Opus 5 | Requirement analysis, planning, dispatch, small self-contained edits, final review |
| Codex | `gpt-5.6-sol` | Substantive code implementation |

Claude decomposes the request, dispatches implementation to Codex, and
aggregates, de-duplicates, and verifies what comes back.

Claude implements directly only when the change is small and self-contained: one
file, a contained blast radius, and a design that is already settled. Anything
spanning several files, anything whose approach is still open, and anything whose
failure would be expensive goes to Codex. When in doubt, dispatch. The Review Gate
below applies to Claude's own edits exactly as it does to Codex's — a
self-implemented change still needs its diff read back and its verification
commands run and reported.

## Invoking Codex

Codex is reachable two ways. They are not interchangeable; pick by how long the
task will run.

### Short turns (< ~20 min): the `codex` MCP tool

`mcp__codex__codex` / `codex-reply`. Use for quick reads, targeted questions,
follow-ups on an existing thread, and single-file changes.

- **It is synchronous and blocking.** Once called, the session can do nothing
  until it returns — no progress is observable, no other tool can run.
- **It dies at ~1800s of silence.** The call aborts, returns no `threadId`, and
  the work cannot be resumed. All of it is lost.

### Long tasks (cross-module review, large refactor): the `codex exec` CLI

Anything that could exceed ~20 minutes — `xhigh` reviews above all — goes
through the CLI, backgrounded. No timeout ceiling, and progress is observable.

```bash
codex exec -m gpt-5.6-sol \
  -c model_reasoning_effort="xhigh" \
  -s read-only \
  -C "$(git rev-parse --show-toplevel)" \
  -o <findings output file> \
  - < <prompt file>
```

- Launch it in the background, redirect to a log, then watch that log with a
  monitor that fires on **both** completion and crash. A monitor that greps only
  for the success marker stays silent through a crash, and silence then looks
  exactly like progress.
- Write the prompt to a file and pipe it in; do not inline it in the shell.
- **Never lower reasoning effort to fit a time budget.** If a task is genuinely
  too large, split it into scoped sub-tasks at full effort — do not downgrade the
  thinking.

### Settings common to both paths

- Every dispatch **must** pass an explicit working directory equal to the current
  Git root — `cwd` over MCP, `-C` on the CLI (`git rev-parse --show-toplevel`).
  Never rely on an inherited cwd.
- Always `model: gpt-5.6-sol` / `-m gpt-5.6-sol`.
- Readers and reviewers always run `read-only`. Only after the user approves
  implementation, switch to `workspace-write` with `approval-policy: never`.

### Reasoning effort — always explicit, because the default differs by path

Set the effort on every dispatch (`config: {"model_reasoning_effort": "..."}`
over MCP, `-c model_reasoning_effort="..."` on the CLI). The two paths do not
share a default:

- The **CLI** inherits `~/.codex/config.toml`.
- The **MCP** server is registered with its own `-c model_reasoning_effort=...`
  argument, which overrides `config.toml`.

These two are configured independently and have disagreed in practice, so an
unspecified effort is not the value you would predict from either one alone.

Use `medium` for mechanical refactors, contained single-file changes, and
documents whose facts and judgments are already settled. Use `xhigh` for complex
algorithm work, experiment design, RL objective derivation, hard cross-module
reasoning, and any refactor that must be a provable no-op for a running job.
Never raise the effort merely because several threads are running.

Design and implementation are separate dispatches: design or derivation at
`xhigh` yields a one-page plan; implementing that settled plan runs at `medium`.
Every implementation prompt restates the research code budget from the
user-global `~/.codex/AGENTS.md` ("Research code budget").

## Thread Continuity

Both paths carry a session identity; record it either way, because a follow-up
that starts a fresh thread throws away everything Codex already read.

- **MCP:** record the `threadId` returned by the first `codex` call. All
  follow-up turns for the same task use `codex-reply` with that `threadId`.
- **CLI:** the log's `session id:` line is the equivalent. Continue it with
  `codex exec resume <id>` (or `--last`).
- Never reuse a session across projects or across unrelated tasks. Start a new
  one instead.
- Parallel read-only tasks each get their own session. Follow-up questions
  within one investigation stay on that investigation's session.

## Parallel Threads

The parallelism policy is **not repeated here** — it lives in the user-global
working agreements (`~/.codex/AGENTS.md`, "Running agents in parallel") because
it applies to every project, not just this one. Read it before dispatching more
than one Codex thread.

Only the bindings specific to this repository are recorded here:

- Every parallel thread still passes the explicit Git-root working directory
  from "Invoking Codex" above (`cwd` over MCP, `-C` on the CLI). A thread that
  inherits one is a defect.
- Only one Codex thread may hold `workspace-write` at a time. All others in
  flight are `read-only`.
- Freeze the writer's diff before dispatching review threads, then run the
  Review Gate below against that frozen diff.
- **Do not write into a tree that a review thread is currently reading**, even
  from Claude's own edits. The reviewer's findings are anchored to `file:line`,
  and editing underneath it silently invalidates them. Use a separate clone or
  worktree for concurrent unrelated work.
- Backgrounded CLI dispatches do not block the session, so exploration and
  document work continue while a long review runs. Blocking MCP calls do block:
  with that path, finish short exploratory work *before* dispatching, or the
  work silently serializes.

## Review Gate

After Codex reports completion, Claude must:

1. Read `git diff` for every changed file.
2. Run the relevant verification commands and report their real output.
3. Confirm each stated requirement is actually satisfied.
4. Check size and ceremony against the research code budget. Over budget, or
   a new hash / manifest / ledger / identity check or fail-closed guard the
   prompt did not ask for, is a rejection: send it back for a smaller version.

When several reviewers report, Claude reconciles them into one de-duplicated
list. Agreement between agents is not evidence — two threads repeating a claim
does not make it truer than one. Verify it.

Report failures and skipped steps explicitly. Do not claim completion on
unverified work.

## Prohibited Without Explicit Approval

Neither agent may silently:

- `git commit` or `git push` of a change that is not part of an approved plan
- install or upgrade dependencies
- submit a scheduler job or launch a long-running GPU task the plan did not name
- start background workers or monitors the plan did not name

Approval attaches to the plan, not to each step: the commits, pushes and
script installs that implement an approved plan proceed without a separate
ask. Ask for a step the plan did not cover, or for an irreversible action.

One carve-out: a backgrounded `codex exec` dispatch and the single monitor that
watches it are part of an already-approved dispatch, not separate background
work, and do not need their own approval. Everything else backgrounded still does.

Last updated: 09/02/2026
