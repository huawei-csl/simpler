# Where qwen3's wall clock actually goes (and two scheduler theories it kills)

**Date**: 2026-09-08
**Verdict**: characterization. On qwen3 the AICPU scheduler threads are ~100%
saturated while the AICore lanes are **idle more than they are busy**, and most
of that idle is graph shape, not scheduling. Two plausible scheduler
explanations were falsified by direct intervention. Measured on a2a3 silicon,
`scan_and_claim`, `SAC_THREADS=4`, TraCR traces.

## The numbers

Core-lane idle (FIN → next start, measured off the `RESET` markers, not from
span lengths):

| | total idle | mean gap |
| --- | --- | --- |
| all gaps | 1,859.8 ms | 106.9 µs |
| after a **>200 µs** task | 1,364.7 ms (73%) | 479 µs |
| after a short task | 495.1 ms | 34 µs |

Against ~1,251 ms of core-busy. Scheduler-lane time is ~180 ms across 4 threads
over a 45 ms wall — saturated.

## Most of that idle is structural

The long tasks are 40 **MIX** tasks (`kernel_ids=[11,12,12]`, `block_num=24`),
one per layer, filling all 24 clusters × 3 cores for ~282 µs. So ~11.3 ms of
each round's 22.6 ms wall is a perfectly packed full chip with nothing to gain.
Their consumer chain then collapses: one `bn=24` cube task plus a dummy that
unlocks 26 single-core cube tasks — so 48 AIV cores have **nothing to run**. No
scheduler can fill them; that needs wider or fused stages in the graph.

The addressable budget is therefore the ~4.6 ms sac loses to hbg (45.2 vs
40.6 ms traced; 27.67 vs 30.32 core-equivalents busy), which lives in the stage
boundaries — not the 1,365 ms of idle.

## Theory 1, falsified: cube pipelining

The evidence looked strong — pipelined-start share ranked tmr 93.4% / hbg 74.5%
/ sac 48.4%, matching the wall-clock ranking exactly, on identical task sets
(15,960 cube + 3,248 vector starts). It is **correlation, not causation**.
Immediate dispatch raised sac to **71.3%** — hbg's level — with wall clock
**unchanged**; and it won 13.5% on PA Case2 where pipelining is ~0.1%
throughout. Do not rebuild this argument.

## Theory 2, falsified: dispatch latency at the boundaries

Speculative gated staging removes almost all AICPU work from the producer's FIN
and cost **+4.8%** —
[2026-09 speculative gated staging](2026-09-sac-speculative-gated-staging.md).
The boundary is not latency-bound.

## What immediate dispatch actually is

Placing resolved consumers onto free cores from inside the completion sweep is a
**scan-avoidance** mechanism, not a latency one. On PA Case2 `Scanning` fell
from 31.4 ms (63.5% of scheduler-lane time) to 0.86 ms (2.0%) and core occupancy
rose 21.54 → 32.06 core-equivalents, for −13.5% wall. On qwen3 `Scanning` was
already 4.6%, so there was nothing to reclaim and the result was a null. The
payoff scales with **wavefront width**, because width is what makes the window
scan expensive (PA 64/256 wide; qwen3 median 12).

## Measurement traps

- TraCR "core busy" **inflates when dispatch gets faster**: a lane's span runs
  to the next marker, so closing idle gaps lengthens spans (Case2 cube 135 →
  157 ms for an identical 16,384 tasks). It is not an occupancy measure — use
  the `RESET`-anchored gap walk.
- Marker *shares* are not comparable across trace vintages (a `Retiring` that
  predates the `Dispatching` marker folds dispatch into retirement). Compare
  wall clock and core-equivalents.
- Round 1 runs ~13% slow; benchmark with `--rounds 4` and drop `inv=1`, or a
  ~1% effect is unreadable.
- The `SIMPLER_SCHED_PROFILING=1` env var is **ignored** (`runtime_builder.py`
  reads only `SIMPLER_ENABLE_PTO_URMA_WORKSPACE`), and
  `build_runtimes --profiling-sched 1` refuses onboard platforms. The working
  route is a temporary edit of `src/common/task_interface/profiling_config.h`.
