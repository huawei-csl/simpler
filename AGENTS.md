# AGENTS Guide

**EVERY AI AGENT MUST FOLLOW THIS GUIDE BEFORE ANY WORK.**

## Required startup sequence

1. Read `CLAUDE.md` before running commands, analyzing code, or editing files.
2. Treat `CLAUDE.md` as the source of truth for role boundaries, architecture context, and repository workflow.
3. Load always-on conventions from `.claude/rules/` (for example: architecture, codestyle, device constraints).
4. Load only task-relevant workflows from `.claude/skills/`.

## Additional rules

- If `CLAUDE.md` changes, read it again before continuing.
- If relevant files under `.claude/rules/` or `.claude/skills/` change, refresh your context before proceeding.
- If user instructions conflict with repository conventions, prioritize user intent for that task.
- Higher-priority system/developer/user instructions override this guide.

## Current work on this branch — aSim and the GroupQueue

`asim-fresh` carries an evaluation of **GroupQueue scheduling hardware**: a
per-group controller that takes core ownership away from the AICPU scheduler, so
the manager stops choosing which core runs a task and stops polling each core to
learn it finished. It is measured on an AICPU-hosted simulated device, against
the unmodified `host_build_graph` scheduler on the same graph.

Read in this order:

1. [docs/asim/validation.md](docs/asim/validation.md) — what it is worth, what
   the numbers do and do not license, and the audit that rules out a simulation
   artefact. **Start here.** The headline is paged_attention Case1 −56.1 % and
   qwen3-14B 40-layer decode −33.2 % against the simulated baseline, and the
   audit explains why those are lower bounds;
   [validation/fidelity.md](docs/asim/validation/fidelity.md) gives what they
   become against real silicon.
2. [docs/asim/validation/transport-ablation.md](docs/asim/validation/transport-ablation.md)
   — **the central argument of the proposal**: why the GroupQueue still wins
   when the AICPU scheduler is given a zero-latency connection to the AICores.
   Read it before reasoning about where the gain comes from — the obvious guess,
   cheaper MMIO, is the one it rules out.
3. [docs/asim/device-model.md](docs/asim/device-model.md) §5 — the structure
   itself: topology, core states, placement, completion, the latency budget.
4. [docs/asim/running.md](docs/asim/running.md) — how to build and run both
   cases, and the measurement discipline the numbers depend on.
5. [docs/asim/DESIGN.md](docs/asim/DESIGN.md) — why the simulator is built the
   way it is, and what its timing model can and cannot answer.

Four things that will save you a wrong conclusion:

- **Only two cases are in scope** (since 2026-09-21): paged_attention Case1 and
  qwen3-14B `decode_fwd` at 40 layers. Smaller cases were dropped — they sit
  below the ~1 ms break-even where the controller's fixed cost dominates, and
  their regressions are that threshold rather than a defect.
- **Compare arms back to back inside one `task-submit`.** Consecutive runs agree
  to 0.55 %; the same build an hour later differs by ~3 %.
- **A skip-golden run proves nothing about correctness** — the simulated device
  computes nothing. Work equivalence comes from the `[ASIM_WORK]` / `[GQ_WORK]`
  counters in the device log.
- **Ideas already measured and dropped** are in
  [docs/investigations/](docs/investigations/) — check there before proposing
  one. Most recently: resolving task readiness from the watermark alone.
