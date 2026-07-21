# Can independent AICPU compute overlap a pending COND LDR?

**Date**: 2026-07-21
**Verdict**: confirmed with caveats — real but modest overlap exists and a
pipelined ordering captures more of it than naive ordering, but only once
per-core "local work" exceeds roughly the LDR's own cost. Not worth
restructuring `check_running_cores_for_completion` today, since its current
per-core work (a handful of comparisons, a few ns) is already fully hidden
under the LDR floor either way.

## Question

[`docs/hardware/mmio-performance.md`](../hardware/mmio-performance.md) states
that a single AICPU thread cannot have two `Device-nGnRE` LDRs in flight at
once ("no outstanding LDR pipelining"), measured by rotating a 10000-LDR loop
across N cores with no change in per-LDR cost. That measurement used only
trivial loop bookkeeping between LDRs, so it only proves LDR-vs-LDR
serialization — it says nothing about whether *unrelated, non-Device*
compute (e.g. a scheduler's per-core slot-transition judgment) can execute
concurrently with a single outstanding COND LDR. If it can, restructuring a
polling loop as `fetch(i); fetch(i+1); process(i); fetch(i+2); process(i+1); …`
(process the *previous* core's data instead of the one just fetched) could
hide some or all of that compute behind the next fetch's bus round trip,
since `process(i)` has no data dependency on `fetch(i+1)`.

## What was tried

Added **Phase 15** to
[`tools/cann-examples/aicpu-mmio-probes`](../../tools/cann-examples/aicpu-mmio-probes/)
(`device/probes.cpp::RunPhase15Overlap`, result fields in
`shared/probes_types.h`, pretty-printer in `host/launch.cpp`). AICPU-only,
same tool used for the original Phase 4 / Phase 12 measurements.

For each compute-block size `iters` in `{0,4,8,16,32,48,64,128,256}`
(a sequential 64-bit LCG-style multiply chain — each step depends on the
previous, so the compiler cannot fold or hoist it, and no memory is
touched, so it can't be confused with a second Device access), measured
three 10000-iteration loops against COND on core 0:

- **calib** — `DummyCompute` alone, no LDR (isolates compute cost).
- **naive** — `v = *cond0; acc = DummyCompute(acc ^ v, iters)` each
  iteration — `process` has a *true* data dependency on the LDR just issued.
- **pipelined** — `v_cur = *cond0; acc = DummyCompute(acc ^ v_prev, iters);
  v_prev = v_cur` — `process` depends only on the *prior* iteration's LDR,
  never the one issued that same iteration.

Ran on device 0 (idle at the time; devices 3-7 were at 100% AICore
utilization from other users' jobs — `task-submit` was not on `PATH` for
this invocation, so the run was unlocked; picked an idle device and kept it
short to minimize collision risk). Repeated 3x for repeatability.

## Result

Per-iteration ns (ticks × 20ns / 10000, `ASCEND_HOME_PATH`'s 50 MHz sys
counter), one representative run:

| iters | calib ns | naive ns | pipeline ns | naive − pipeline |
| ----- | -------- | -------- | ----------- | ----------------- |
| 0     | 1        | 97       | 98          | −1                 |
| 16    | 32       | 97       | 97          | 0                  |
| 32    | 63       | 131      | 117         | 14                 |
| 48    | 95       | 164      | 134         | 30                 |
| 64    | 127      | 195      | 159         | 36                 |
| 128   | 256      | 323      | 287         | 36                 |
| 256   | 512      | 587      | 549         | 38                 |

The `naive − pipeline` column was reproducible across all 3 runs to within
~2ns at every point, despite the absolute baseline shifting between runs
(~97ns in run 3 vs ~128ns in runs 1-2 — some system-level noise, e.g. DVFS
or cache state, on this shared box; the *relative* effect was stable
regardless).

Two things fall out of this:

1. **Both orderings already show partial overlap.** Neither `naive` nor
   `pipeline` tracks the naive "no overlap" prediction (`compute_ns +
   ~97ns`) — at iters=256 that predicts ~609ns, but naive measures 587ns and
   pipeline 549ns. So the mmio-performance.md claim that a Device LDR fully
   blocks the core is **too strong**: some independent ALU work does execute
   concurrently with an outstanding COND LDR even without deliberately
   reordering the code, likely because consecutive LDRs are independent of
   any single iteration's compute and the core has at least some lookahead
   past a pending Device access.
2. **Pipelining captures more of it, but only once compute is non-trivial.**
   For iters ≤ 16 (compute ≤ ~32ns) there is no measurable difference — both
   orderings are already fully hidden under the ~97-99ns LDR floor. The gap
   opens at iters=32 (~63ns compute) and plateaus at a consistent ~36-38ns/
   iteration saved once compute ≥ ~127ns (iters≥64). At iters=48
   (~95ns compute — matching the "~100ns local work" case this was asked
   about), pipelining saves ~30ns/iteration, about 15-18% of the total.

## Why not (now)

`check_running_cores_for_completion`'s actual per-core work (reading
`early_dispatch_state`, extracting task id/state, deciding the slot
transition) is a handful of branches and comparisons — order of a few ns,
deep in the "iters ≤ 16" regime where this measurement shows zero benefit.
Restructuring the poll loop into a pipelined fetch/process shape today would
add real complexity (an extra "previous core" carry-over, more awkward
handling of the gated-core skip and multi-core cohort logic) for a saving
this measurement puts at effectively 0ns.

## When to reconsider

If `check_running_cores_for_completion`'s per-core judgment work grows to
resemble the ≥~50-100ns regime (e.g. folding in more decision logic from
elsewhere in the main scheduler loop, as proposed alongside this
investigation), pipelining is worth prototyping directly in the scheduler:
expect on the order of a 15-35 ns/core saving depending on how much compute
lands in that step, which at 24 cores would be roughly 0.4-0.8 µs off a full
polling round — worth it only if that round shows up materially in a
profile.

## References

- Tool: [`tools/cann-examples/aicpu-mmio-probes`](../../tools/cann-examples/aicpu-mmio-probes/)
  Phase 15 (`device/probes.cpp`, `shared/probes_types.h`, `host/launch.cpp`).
- [`docs/hardware/mmio-performance.md`](../hardware/mmio-performance.md) —
  the Phase 12 LDR-serialization measurement this refines (LDR-vs-LDR
  remains fully serial; this entry is about LDR-vs-unrelated-ALU-work,
  which is a different question the original phases didn't test).
- Scheduler code discussed: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_completion.cpp::check_running_cores_for_completion`.
