# Alternating half-core polling/dispatch in the AICPU scheduler loop

**Date**: 2026-07-23
**Verdict**: measured — mixed, no clear signal either direction; reverted

## Question

`resolve_and_dispatch`'s outer `while (true)` loop (one instance per AICPU
scheduler thread) does two core-touching passes every iteration: Phase 1
(`check_running_cores_for_completion`) polls every running core's COND
register for FIN/ACK, and Phase 4 (`dispatch_ready_tasks`) scans every
idle/pending core of this thread for dispatch opportunities. Each pass
touches the thread's full core set (`CoreTracker::core_num()` cores) every
single iteration.

Proposal: split each thread's core-bit range in half and alternate which
half Phase 1 and Phase 4 touch, one half per outer-loop iteration. A pass
that only has to look at half the cores does less MMIO/scan work per
iteration, so the loop cycles faster; a core in the "cold" half waits at
most one extra iteration to be looked at again. Whether the net effect on
reaction latency is positive or negative isn't obvious a priori — cheaper,
more frequent passes vs. half the cores going temporarily unwatched — hence
measuring it directly rather than reasoning it out. This is a different
mechanism from
[the LDR/compute-overlap investigation](2026-07-aicpu-ldr-compute-overlap.md)
(instruction-level pipelining within one poll of one core); this changes how
much of the core set one poll covers.

## What was tried

Implementation (see the diff on `overlap-COND-LDR` for the 4 touched files):

- `CoreTracker::half_mask(bool first_half)` (`scheduler_types.h`) — returns a
  `BitStates` mask over `[0, core_num())`, split at the midpoint.
- `check_running_cores_for_completion` (`scheduler_completion.cpp`) — takes a
  new `bool poll_first_half` param, ANDs `get_all_running_cores()` with
  `tracker.half_mask(poll_first_half)` before the poll loop.
- `dispatch_shape` / `dispatch_ready_tasks` (`scheduler_dispatch.cpp`) — same
  `poll_first_half` param threaded through, ANDed into the `cores` BitStates
  used for both the AIC/AIV `get_dispatchable_cores` and the MIX
  `get_cluster_offset_states` paths (both the initial fetch and the
  mid-`dispatch_shape`-loop refetch). `early_dispatch_shape` (Phase 4b, the
  gated pre-staging source) was deliberately left untouched — it's a
  different mechanism with its own doorbell/rendezvous protocol that a
  reduced core view could interact with in ways not exercised here.
- `resolve_and_dispatch` — a single `bool poll_first_half` local, flipped
  once at the top of every loop iteration (before Phase 1), so Phase 1 and
  Phase 4 of the same pass always agree on which half is live.

Correctness: `pytest tests/st/a2a3/tensormap_and_ringbuffer --platform
a2a3sim --device 0-3` — 40 passed both before and after (identical 2
pre-existing `need 2 devices` environment errors on both runs, unrelated to
this change).

Benchmark: `tools/benchmark_rounds.sh -d 3 -n 100 -r
tensormap_and_ringbuffer` (default 8 example/case combinations), run
unlocked on device 3 (`task-submit` not on `PATH` in this environment;
`npu-smi` confirmed no other process on device 3 before each run) — once
against HEAD (baseline, half-split stashed out via `git stash push --
<the 4 files>`), once against the working tree (current, half-split
applied), same device, back to back. Compared with the new
[`compare_benchmark_runs`](../../simpler_setup/tools/README.md#compare_benchmark_runs)
tool: 5%-trimmed Welch's t-test + Cohen's d per example × metric.

## Result

`Effective` (the headline device-domain metric) per example:

| Example | Δ Effective | p | Cohen's d | Verdict |
| --- | ---: | ---: | ---: | --- |
| alternating_matmul_add (Case1) | +0.38% | 0.0061 | -0.41 | REGRESSION |
| benchmark_bgemm (Case0) | -0.78% | 0.0000 | +0.84 | IMPROVEMENT |
| paged_attention_unroll (Case1) | -0.66% | 0.0000 | +1.46 | IMPROVEMENT |
| paged_attention_unroll (Case2) | +0.95% | 0.0000 | -0.92 | REGRESSION |
| paged_attention_unroll_manual_scope (Case1) | +0.22% | 0.0000 | -0.64 | REGRESSION |
| paged_attention_unroll_manual_scope (Case2) | -1.24% | 0.0000 | +1.27 | IMPROVEMENT |
| batch_paged_attention (Case1) | -0.29% | 0.0000 | +0.75 | IMPROVEMENT |
| qwen3_14b_decode (StressBatch16Seq3500) | -0.03% | 0.5445 | +0.09 | noise |

Overall: 4 improved, 3 regressed, 1 within noise (of 8, on `Effective`).
`Orch`/`Sched` track the same per-example sign as `Effective` in every case.
`Host` swings much larger (up to +32%/-30%) but is dominated by Python
dispatch-loop overhead on this shared box and is not a reliable read (see
the `benchmark` skill's Step 6 note on `Host` noise) — full per-metric table
in the raw tool output, not reproduced here.

No example shows a large effect in either direction (all `|Δ Effective| <
1.3%`, `|d| < 1.5`), and there's no obvious pattern by core count, shape
(AIC/AIV/MIX), or task size separating the improved set from the regressed
set at a glance.

## Why not (now)

The two hypothesized effects (cheaper-but-more-frequent passes vs. cores
temporarily unwatched) appear to roughly cancel, with which one wins varying
by example rather than by any characteristic identified so far. That's a
"no signal" result, not a "confirmed win" or "confirmed loss" — same
category as the LDR/compute-overlap follow-ons, where a real but small
effect at this call frequency doesn't clear the noise floor across a mixed
example suite. Root-causing the per-example split (why `benchmark_bgemm`
improves but `paged_attention_unroll_manual_scope Case1` regresses) was not
attempted — the change stops here pending a decision on whether that's
worth pursuing.

## When to reconsider

- If a specific workload's profile shows Phase 1 or Phase 4's per-iteration
  scan cost as a measurable fraction of scheduler time at a *larger* core
  count than this box's test devices carry (more clusters per thread means
  a bigger fixed-vs-half saving).
- If the per-example split can be explained (e.g. correlates with core
  count, block count, or MIX vs AIC/AIV shape) — that would turn this from
  "mixed" into "conditionally worth it," worth a targeted re-measurement.

### Closing: reverted

No net win, so it didn't ship. `scheduler_types.h`, `scheduler_completion.cpp`,
`scheduler_dispatch.cpp`, and `scheduler_context.h` were restored to their
pre-experiment state (`git restore`), rebuilt, and re-verified: `pytest
tests/st/a2a3/tensormap_and_ringbuffer --platform a2a3sim --device 0-3` — 40
passed, same 2 pre-existing environment-only `need 2 devices` errors as every
other run in this investigation, confirming the revert changed nothing else.

## References

- Working-tree diff (reverted): `scheduler_types.h`, `scheduler_completion.cpp`,
  `scheduler_dispatch.cpp`, `scheduler_context.h` on `overlap-COND-LDR`.
- [`compare_benchmark_runs`](../../simpler_setup/tools/README.md#compare_benchmark_runs)
  — the statistical comparison tool used above.
- Sibling investigation:
  [Can independent AICPU compute overlap a pending COND LDR?](2026-07-aicpu-ldr-compute-overlap.md)
