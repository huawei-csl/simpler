# aSim validation — method, cases, fidelity, and measured deltas

How M0 is judged, which workloads can judge it, where the model stands, and
what the GroupQueue is worth against it. Landing page: [DESIGN.md](DESIGN.md).

## The gate is the scheduling window, not `device_wall`

On `host_build_graph` the scheduling window was most of `device_wall` (~120 µs
of a ~175 µs steady wall), so `device_wall` was a fair proxy. **On
`scan_and_claim` it is not.** Measured on paged_attention Case1, the scheduling
window is 19.7 ms of a 21.3 ms `device_wall` — but on bgemm it is 90 µs of an
886 µs wall, i.e. **10 %**. The rest is per-run host and AICPU fixed cost that
aSim reproduces by executing the *same code*, so matching it demonstrates
little and missing it blames the device model for something else.

The window is read from SAC's own per-thread `sched_cost=` device-log line
(`scheduler_cold_path.cpp`), taken as the **max over threads per round** and
then the **median over steady-state rounds**. It requires `SIMPLER_DFX=1` and
an open log level (`--log-level INFO`; the device level is inherited from the
host logger).

## Procedure

Build and run commands for both in-scope cases, and the discipline the numbers
depend on, are in [running.md](running.md).

```bash
# calibrate (one cold round; swimlane is gated on --rounds <= 1)
pytest <case> --platform a2a3 --skip-golden --rounds 1 --enable-chip-swimlane 3
python docs/asim/analyzers/asim_gen_calib.py outputs/<case>/chip_swimlane_records.json - out.calib

# compare (steady state, 4 threads)
SAC_THREADS=4 pytest <case> --platform a2a3     --skip-golden --rounds 14 --log-level INFO
SAC_THREADS=4 SIMPLER_ASIM_CALIB=out.calib \
              pytest <case> --platform a2a3asim --skip-golden --rounds 14 --log-level INFO
```

Round 1 is cold — on bgemm it is 1.42 ms against an 886 µs steady state — so
**never read a single-round number**. `--skip-golden` is mandatory, not a
convenience: aSim executes no kernel and writes no output tensor, so a golden
check can never pass (see [calibration.md](calibration.md)).

## Case selection

Two representative cases. Both have to be large enough that the scheduling
window dominates, and between them they have to span homogeneous and
heterogeneous work.

| case | tasks | kernels | character |
| ---- | ----- | ------- | --------- |
| `paged_attention` Case1 | 65 536 | 4 | homogeneous, AICPU-bound, cores 86 % idle |
| `qwen3_14b_decode` | 19 179 | 36 | heterogeneous; SPMD `block_num` 5–50, whole-device sync-start cohorts |

**Cases that cannot measure anything, and must not be quoted:** `benchmark_bgemm`
(128 tasks) and `paged_attention` small1 (**4 tasks**) have windows of 40–100 µs,
entirely inside run-to-run spread. Their apparent errors (+8.7 %, −18.8 %) are
noise with no information content. `paged_attention` Case2 is a smaller Case1
and adds no coverage.

## Fidelity against silicon

Both baselines' error against the hardware they model, the three-arm method, and
the residual blind spots are in **[validation/fidelity.md](validation/fidelity.md)**.
The short version, for reading the deltas below:

| case | M0 error vs silicon | M2 vs M0 | M2 vs silicon |
| ---- | ------------------- | -------- | ------------- |
| paged_attention Case1 | +5.0 % (M0 slow) | -55.1 % | **-52.9 %** |
| qwen3-14B decode, 40L | -4.1 % (M0 fast) | -28.1 % | **-31.1 %** |

**The error changes sign between the cases, so it does not cancel in a delta.**
Every GroupQueue figure in this document is quoted against M0; the last column is
what it becomes against a real device, and the correction goes one way on
paged_attention and the other on qwen.

An older gate reading ±0.5 % is still cited in places. It was measured with
`scan_and_claim` as M0, not the `host_build_graph` M0 these campaigns use, and
does not transfer -- see the subdoc.

## What the GroupQueue is worth (2026-09-21)

Two cases carry the campaign. Both arms of each pair run back to back in one
submission on one held card, so the delta is free of the ~3 % session drift.
paged_attention's figure is the pooled median of **eight interleaved reps**: its
M0 arm holds to 0.7 %, but its M2 arm spans 13.7 %, so a single pair cannot place
it closer than a few points. Every one of those eight reps fell between -58.1 %
and -52.0 %.

| case | M0 | M2 | vs M0 |
| ---- | -- | -- | ----- |
| paged_attention Case1, 65,792 tasks, grouped 32/4 | 22.285 ms | 9.776 ms | **-56.1 %** |
| qwen3-14b `decode_fwd`, 40 layers (pypto-lib) | 28.423 ms | 18.989 ms | **-33.2 %** |

**These are gains against M0, and M0's own error against silicon changes sign
between the two cases** -- see [validation/fidelity.md](validation/fidelity.md).
Corrected to hardware, paged_attention's gain shrinks (to -52.9 %) and qwen's
grows (to -31.1 %), so neither figure here transfers to a real device unchanged
and they do not move together.

**The gain is not the cost of transport.** Give M0 a zero-latency connection to
the AICores -- free send, free status read, instant completion notice -- and it
still loses to an unmodified M2 by 2.0x on paged_attention and 1.4x on qwen; free
transport closes 27 % and 2 % of the gap. The advantage is that readiness,
completion and placement leave the AICPU's critical path. The ablation and the
mechanism are in [validation/transport-ablation.md](validation/transport-ablation.md).

The in-repo expanded 40-layer qwen, a separate orchestration of the same model,
independently measures -32.6 %, so the qwen figure reproduces across two
codegen paths.

Sweeping `--fwd-layers` on the qwen case gives a flat per-layer cost in each arm
-- M0 ~710 us, M2 ~475 us -- so the ratio is a property of the runtime rather
than of the graph, and it asymptotes at 0.668 once there is enough work to reach
it.

> **The qwen arm is not established as ungrouped, so -33 % must not be read as
> the bare throughput gain.** An earlier version of this section split the two
> results that way, attributing PA's further -22 points to declared grouping.
> The pypto codegen patch was live for these runs and the generated qwen
> orchestration carries **6 `rt_group_begin()` calls**, so the graph reached the
> runtime with declarations on it. Whether the runtime resolved them or dropped
> them -- it drops a declaration that holds no internal edge, and earlier qwen
> declarations measured `deps_named=0`, so dropping is likely -- is unverified,
> because the `[GQ_GROUP]` counter that would settle it is not readable on the
> pypto path (see [patches/README.md](patches/README.md)). Until it is read,
> -33.2 % is "qwen-40L with declarations emitted, resolution status unknown",
> and the grouping-versus-throughput split is an open question rather than a
> result. The in-repo expanded orchestration is a separate path whose
> declarations are controlled directly, and it is the cleaner place to settle
> the split.

### The two cases are opposite graph shapes

Both are recorded by dep-gen and profiled the same way: level = longest path in
edges, width = tasks at that level, "fillable" = a level wide enough to occupy
all 72 cores.

| property | paged_attention Case1 | qwen3-14B decode |
| -------- | --------------------- | ---------------- |
| tasks | 65,792 | 11,085 |
| components | **256, disjoint** | **1** |
| critical path | 33 levels | 195 levels (pypto path) |
| mean width | — | 22.7 |
| work at fillable levels | — | 15 % |
| maximum fan-in | **3** | **86** |

paged_attention is wide and shallow, in 256 pieces that share no edge. qwen is a
single long thin chain. Nearly every difference between the two — which one
rewards grouping, which one triggers the steal path, which one can be expressed
with four dependency comparators — follows from that one contrast, so it is worth
checking a new case's shape before predicting how it will behave.

**Width does not predict the sign of the M0-vs-M2 delta**, which was tested and
falsified: the 16-layer qwen graph is 195 levels deep and 22.8 wide with only 15 %
of its work at fillable levels, and it still wins 32.7 %. What width predicts is
*which mechanism* pays.

One caveat on the recorded graphs. The in-repo expanded qwen orchestration emits
40 layers in a loop, yet its recorded longest path is 32 — and forty chained
layers cannot have a path shorter than 40. So dep-gen under-records some
cross-layer edges, and a width read off `deps.json` is an upper bound. It does
not affect any timing here: both arms execute the runtime's real dependencies,
not the recording.

### The break-even is about 1 ms of device work

M2 carries a fixed setup cost, so below roughly 1 ms it loses. Measured on one
case and one code path, varying only the work:

| qwen `--fwd-layers` | M0 | M2 vs M0 |
| ------------------- | -- | -------- |
| 1 | 841 us | +4.3 % |
| 2 | 1.55 ms | -13.4 % |
| 4 | 2.97 ms | -23.3 % |
| 16 | 11.4 ms | -32.7 % |
| 40 | 28.4 ms | -33.2 % |

Everything the campaign measured falls on this curve, including the two cases
dropped from scope on 2026-09-21 for being too small to matter: deepseek v4
`expert_routed` (615 us, +5.0 %) and the bare qwen invocation (824 us, +7.5 %).
Their regressions are the threshold, not a defect peculiar to them.

Two earlier readings of the same data were wrong and are recorded here so they
are not re-derived. **Graph size does not predict the sign** -- qwen at
`--fwd-layers 2` is ~1,100 tasks and wins, while the 4,470-task bare invocation
loses. **Width does not predict it either** -- the 16-layer graph is 195 levels
deep and 22.8 wide, with only 15 % of its work at levels able to fill 72 cores,
and it wins -32.7 % all the same. Width still describes *how* the two in-scope
cases differ (PA's 256 disjoint groups against qwen's long thin chain), which is
why PA is the one that rewards grouping; it just does not set the sign.

## Is an M0-vs-M2 delta an artefact? (audit, 2026-09-17)

The M2 wins are large enough that the burden is on the simulator to show it is
not handing them out. Everything below was checked with the two arms running the
same graph back to back on one locked card.

**Symmetric, and therefore not a source of advantage:**

| property | M0 (`asim`) | M2 (`asimgq`) |
| -------- | ----------- | ------------- |
| compute issued/round, PA | 91,128 us | 91,051 us (-0.08 %) |
| compute issued/round, qwen | 746,555 us | 746,519 us (-0.005 %) |
| tasks completed | 65,792 / 11,087 | 65,792 / 11,087 |
| cores | 72, from `get_worker_count()` | 72, same call |
| compute draw | table + bounded Irwin-Hall | same code, same calib |
| slots per core | 2 (`active` + `pushed`) | 2 (`outstanding 0..2`) |
| submit -> work starts | 1008 ns | 1038 ns |

M2's dispatch is the longer of the two: it pays the die crossing to reach its
controller *and* the controller-to-core link, where M0 crosses once.

**The one large asymmetry runs against M2.** M0's modelled status read (195 ns)
exceeds what the simulator costs to execute one, so it never overshoots and its
window is model-faithful: overrun is 0-4 us per thread per round. M2's modelled
poll (5-10 ns) is *below* its own cost, so nearly every poll overshoots and the
excess lands in the measured window -- 15,211 us per thread per round on qwen,
about 65 % of it. The published deltas are therefore **lower bounds**.

**Cheapening the simulator does not fix it, because the poll count is
demand-driven.** Cutting the poll from 136 ns to 117 ns raised the count from
104k to 114k and left wall-per-poll at 210 -> 206 ns; only ~120 ns of that is the
simulator, the rest being the scheduler's own loop. Since `asimgq` spins to a
real-time deadline, a faithful window needs modelled latency >= the real
per-poll cost (~200 ns). At 10 ns it is 20x under, so the loop outruns the model
and the window measures the host. The fix is virtual time -- advancing a model
clock instead of spinning -- not a faster simulator.

**What this licenses.** Rankings and the direction of an M0-vs-M2 delta are
sound. M2's absolute `device_wall` is not: it is a floor set by simulator plus
scheduler cost, which is also why qwen barely moves between a 5 ns and a 10 ns
read model -- both sit under the floor. Only a 30 ns *per word* read (~300 ns on
a ten-deep scan) rises above it and moves the number.

**Measurement discipline this established.** Three consecutive runs of one
binary agree to 0.55 %; the same build an hour later differs ~3 %, with the card
held throughout. Compare arms back to back inside one submission. The
`[GQ_WORK]` counters themselves cost ~1.3 % of `device_wall`, so an instrumented
run is its own baseline.

## The M2 campaigns

What the GroupQueue is worth has been measured twice, against different baselines
and with different gates. They are kept apart because their figures do not compare:

| campaign | gate | M0 | headline |
| -------- | ---- | -- | -------- |
| [2026-09-04](validation/m2-scheduling-window.md) | scheduling window (`sched_cost=`) | `scan_and_claim` | qwen +4.6 %, paged_attention −11.8 % |
| [2026-09-14](validation/m2-group-queue.md) | `device_wall` | `host_build_graph` (`a2a3asim`) | paged_attention −55 %, qwen a loss at every group size |

The later one adds the task-grouping contract: the ready queue holds groups rather
than tasks, and a controller resolves a group's internal edges. It wins where
groups are independent of each other and loses where they are chained.
