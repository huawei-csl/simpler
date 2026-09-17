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

## Current fidelity (2026-09-04, 4 threads)

| case | real | aSim | error |
| ---- | ---- | ---- | ----- |
| paged_attention Case1 | 19 550 µs | 19 453 µs | **−0.50 %** |
| qwen3_14b_decode | 43 418 µs | 43 542 µs | **+0.29 %** |

Two properties make this a gate rather than a fit:

- **The signs are mixed.** Every earlier stage was uniformly negative, the
  signature of a missing term. Mixed signs at this magnitude is noise.
- **The errors are below the real measurement's own reproducibility.** Case1's
  real window read 19 772 µs in one campaign and 19 550 µs in the next — 1.1 %
  apart. There is nothing further to demonstrate without many more reps.

Progression, on Case1 at 4 threads: **−9.0 % → −3.7 % → −0.50 %**. The first
step was correcting the in-situ poll cost, the second was adding the AICore-side
setup that was missing from the model entirely — both in
[calibration.md](calibration.md).

## Residuals and known blind spots

1. **One fitted parameter remains** (`read = 195 ns`). It is *identifiable*
   rather than free — poll count differs 3.4× between the two cases, so a
   materially wrong value would give task-count-scaled errors of consistent
   sign, and instead both land within ±0.3 % with opposite signs. It was
   nonetheless fitted *before* `ack` was corrected, so it may have absorbed part
   of the setup term.
2. **No core-bound case is validated.** Both cases leave the cores ≥86 % idle,
   so the 2-deep pipeline's overlap assumption — that `receive_to_start` hides
   behind the previous task's compute — is never exercised. It would bite in a
   case where cores saturate.
3. **Compute-distribution shape is unmodelled and measured not to matter here.**
   1.89 % of real tasks exceed mean+3σ, which aSim's bounded Irwin-Hall draw
   cannot produce, with tails to 4.6× the mean. Quadrupling σ moved the window
   0.4 %, so with cores idle this cannot supply a meaningful error — but it is a
   latent term for a core-bound case.
4. **Bias stability across thread counts was the fitness test for A/B use.**
   Before the corrections the error ran −13.9 % at 1 thread to −9.0 % at 4;
   fitting `W(N) = S + P/N` localised it to the parallel term (serial −2.1 %,
   parallel −16.2 %), which is what identified the poll cost as the cause. Any
   future regression should be re-checked the same way, because a bias that
   varies with configuration does not cancel in an M0-vs-M2 delta.

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
