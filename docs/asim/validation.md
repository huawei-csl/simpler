# aSim validation — method, cases, fidelity, and measured deltas

> **M2 figures dated before 2026-10-07 over-state the GroupQueue.** Until the
> simulated cores ran on a model clock ([DESIGN.md](DESIGN.md) §3), the window
> correction removed simulator work the cores had computed through. Current
> figures are in [What the GroupQueue is worth](#what-the-groupqueue-is-worth);
> earlier ones are kept below for their method and marked superseded.
>
> **The first model-clock figures, the same day, under-state it on qwen** (M2
> +13.0 %): their manager retired a finish only in prefix order and never put a
> second task behind a running core. Both are fixed; the figures below are on the
> fixed manager.

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
| paged_attention Case1 | +3.6 % (M0 slow) | -47.9 % | **-46.1 %** |
| qwen3-14B decode, 40L | -4.4 % (M0 fast) | +1.2 % | **-3.3 %** |

**The error changes sign between the cases, so it does not cancel in a delta.**
Every GroupQueue figure in this document is quoted against M0; the last column is
what it becomes against a real device, and the correction goes one way on
paged_attention and the other on qwen.

An older gate reading ±0.5 % is still cited in places. It was measured with
`scan_and_claim` as M0, not the `host_build_graph` M0 these campaigns use, and
does not transfer -- see the subdoc.

## What the GroupQueue is worth

Measured 2026-10-07 on the model clock and the fixed manager, all arms
interleaved inside one device lock -- 48 samples per arm on paged_attention, 33-36
on qwen. qwen is the in-repo expanded 40-layer orchestration.

| case | silicon | M0 | M2 | M2 vs M0 | M2 vs silicon |
| ---- | ------- | -- | -- | -------- | ------------- |
| paged_attention Case1, 65,792 tasks | 21.256 ms | 22.012 ms | 11.465 ms | **-47.9 %** | **-46.1 %** |
| qwen3-14B decode, 40 layers | 34.673 ms | 33.143 ms | 33.527 ms | **+1.2 %** | **-3.3 %** |

**paged_attention: the GroupQueue halves the time.** Its cores sit ~90 % idle
under M0, because the AICPU scheduler cannot feed them faster than ~1.1 us a task;
the GroupQueue takes readiness, completion and placement off that path. What it
leaves is balance across groups: its cube cores spend ~1.2 ms a round idle while
a peer queue holds cube work they cannot take. That was measured on 2026-10-07 by
a per-poll classification the simulator no longer carries, because its cost
landed in the overrun the model clock has to remove.

**qwen: parity.** qwen is bound by its cube cores' compute -- the busiest carries
30.8 ms of work and M0 finishes within ~4 % of that -- and it declares no groups,
so M2's manager does the same software job as M0's scheduler. The two arms'
scheduling windows agree to 0.4 %; the rest of M2's 1.2 % lies outside the
window, in setup and teardown.

What cube idle qwen has is mostly its 40 sync-start cohorts a round: 0.61 of the
0.78 ms each cube core stands idle in the window falls while a cohort waits for
the device to empty. Releasing cohorts from the queues instead of draining the
device from the AICPU ([device-model.md](device-model.md)) measured level with
the drain, -0.4 % and -0.1 % in two sessions: the wait is for every core to finish
what it was running, which no release scheme removes.

**The first model-clock measurement had M2 13 % slower on qwen**, from two defects
in the manager rather than the controller:

- **It retired a finish only once every earlier position had.** A task that
  finished ahead of the prefix kept its consumers waiting and its capacity token
  out until the oldest unfinished task completed -- the prefix stall
  [the readiness investigation](../investigations/2026-09-groupqueue-watermark-readiness.md)
  measured on the controller side. Worth ~4.5 points.
- **It never put a second task behind a running core.** The core tracker it shares
  with `host_build_graph` reopens a core's pending slot on the running task's ACK,
  which the GroupQueue manager never reads. M0 starts 94 % of its 665 cube tasks
  per core back to back; M2 started none, each waiting ~3.7 us for a refill.
  Worth ~6.5 points.

Cutting M2's end-of-run logging to M0's volume took about one point more, since
those lines are written inside `device_wall`. paged_attention did not move under
either fix: its grouped entries take no capacity token.

**The paged_attention gain is not the cost of transport.** Give M0 a zero-latency
connection to the AICores -- free send, free status read, instant completion
notice -- and it is still 1.64x the time of M2; free transport closes 30 % of the
gap. The ablation and the mechanism are in
[validation/transport-ablation.md](validation/transport-ablation.md).

Compute issued agrees between the arms to 0.06 % (paged_attention) and 0.02 %
(qwen), and every M2 round completes every task. One of the three qwen M0 runs
stalled in its twelfth round: a ready sync-start cohort the drain never
dispatched, every core idle, until the scheduler timeout. M0's code did not change,
so the defect is in the sync-start drain the two runtimes share; the run's eleven
completed rounds are kept, and dropping them moves M0's median by 2 us.

## A second qwen shape: the Graph orchestration

The in-repo qwen case ships two orchestrations of the same network. Every figure
above uses `decode_fwd_layers_expanded.cpp`, which submits all 11,085 tasks, so
the manager sees the whole graph. The shipped `decode_fwd_layers.cpp` instead
submits each of the 40 layers as **one Graph task the device Scheduler expands**,
which the manager never sees the inside of. Measured 2026-10-09, all three arms
interleaved in one device lock, 48 samples each:

| arm | `device_wall` | scheduling window | vs silicon |
| --- | ------------- | ----------------- | ---------- |
| silicon | 38.297 ms | 37.399 ms | — |
| M0 | 38.230 ms | 37.272 ms | **-0.2 %** |
| M2 | 38.136 ms | 36.959 ms | **-0.4 %** |

**This is the simulator's closest agreement with hardware on any case** -- both
baselines inside half a percent, where the expanded orchestration sits at ~4 %
([validation/fidelity.md](validation/fidelity.md)). The manager handles 47 tasks
a round instead of 11,085, so almost none of the window is the AICPU software
this model does not reproduce; what is left is compute, which it draws from the
same calibration. A case whose scheduling is nearly free is the one a device
model can match, and that is also why it cannot discriminate between designs:

**M2 is level with M0 here too, at -0.2 %.** Its cube cores are 87 % busy (32.1
of 37.0 ms a round) against 11.2 ms on the vector cores, so this shape is
cube-bound like the expanded one -- and the GroupQueue has even less to win,
because the dependency hops it removes happen inside the Scheduler's graph
expansion rather than on the manager's path.

The two orchestrations are **not the same workload**: the Graph form issues 1,308
ms of core time a round against the expanded form's 746 ms, and runs 38.3 ms
against 34.7 ms on silicon. Quote them separately. Compute issued agrees between
the arms to 0.01 %, and every round of both arms completed every task.

Two mechanical notes for reproducing it. The driver lives in the
`tensormap_and_ringbuffer` case but its copy of this source does not compile
against the `host_build_graph` orchestration API, so the `host_build_graph`
copy is the one to pass. And `deps.json` cannot name the func_id of a task the
device expands, so the calibration comes from the swimlane's own per-task kernel
ids -- `asim_gen_calib.py` already prefers them, and takes `-` in place of a
dep-gen round.

## Is an M0-vs-M2 delta an artefact?

The 2026-09-17 audit checked the two arms for asymmetries that could hand M2 its
wins: what it found symmetric, the one large asymmetry it measured, and the
measurement discipline it established are in
[validation/artefact-audit.md](validation/artefact-audit.md). The fix it called
for, virtual time, is the model clock that now runs ([DESIGN.md](DESIGN.md) §3).

## The M2 campaigns

What the GroupQueue is worth has been measured four times, against different
baselines and gates. They are kept apart because their figures do not compare:

| campaign | gate | M0 | headline |
| -------- | ---- | -- | -------- |
| [2026-09-04](validation/m2-scheduling-window.md) | scheduling window (`sched_cost=`), raw | `scan_and_claim` | qwen +4.6 %, paged_attention −11.8 % |
| [2026-09-14](validation/m2-group-queue.md) | `device_wall`, over-corrected | `host_build_graph` (`a2a3asim`) | paged_attention −55 %, qwen a loss at every group size |
| [2026-09-21](validation/m2-campaign-2026-09-21.md) | `device_wall`, over-corrected | `host_build_graph` (`a2a3asim`) | paged_attention −56.1 %, qwen −33.2 % |
| [2026-10-07](#what-the-groupqueue-is-worth) | `device_wall`, model clock, fixed manager | `host_build_graph` (`a2a3asim`) | paged_attention −47.9 %, qwen +1.2 % |

The raw window of the first campaign includes the simulator's own work, which
inflates M2; the second and third removed that work including the part the cores
computed through, which over-states M2. The fourth runs the cores on a model clock
so the removal is exact, on the manager fixed to retire on notice and pipeline a
second task per core, and is the one to quote.

From the second on, the campaigns carry the task-grouping contract: the ready queue
holds groups rather than tasks, and a controller resolves a group's internal edges.
It wins where groups are independent of each other and loses where they are
chained.
