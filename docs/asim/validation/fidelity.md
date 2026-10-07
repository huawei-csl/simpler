# aSim fidelity — how close the simulated device is to the real one

> **Caveat (2026-10-07) — the M2 figures on this page overstate the GroupQueue.**
> Every M2 `device_wall` here has the simulator's entire self-overrun subtracted
> (`SIM_WINDOW_CORRECTION` in `src/common/platform/include/aicpu/device_phase_aicpu.h`,
> on since 2026-09-14), including the part the simulated cores were computing
> through. On qwen that puts M2 below the physical floor: M0's busiest cube core
> carries 30.8 ms of work in a 32.0 ms window, so no scheduler can gain more than a
> few percent there, and the qwen gains quoted here are an artefact. Raw scheduler
> windows, same session: paged_attention M0 19.11 -> M2 14.25 ms (-25 %); qwen
> M0 32.00 -> M2 36.83 ms (+15 %). paged_attention's true gain lies between the
> raw -25 % and the corrected figure until the correction is fixed. M0 figures,
> including M0's fidelity against silicon, are unaffected.
>
> On this page that affects the M2-vs-M0 and M2-vs-silicon columns; the M0 error
> against silicon is measured without any correction and stands.

Two baselines have been checked against silicon, and they are different numbers
measuring different things. Which one applies depends on which M0 a figure was
taken against. Method and run commands are in [running.md](../running.md); the
measured GroupQueue deltas are in [validation.md](../validation.md).

## The `scan_and_claim` baseline (2026-09-04, 4 threads)

This is the gate for the campaign of the same date, whose M0 was
`scan_and_claim`. **It does not cover the `host_build_graph` M0 that the
GroupQueue comparison actually runs against** -- that one is measured in the
next section, and it is a different number.

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
[calibration.md](../calibration.md).

## The `host_build_graph` baseline (2026-10-06, 4 threads)

The M0 that every GroupQueue figure is quoted against, measured directly against
the silicon it models. **Three arms**, interleaved inside one device lock so they
share a session and cancel its drift:

| arm | platform / runtime | what it is |
| --- | ------------------ | ---------- |
| base | `a2a3` / `host_build_graph` | real silicon |
| sim | `a2a3asim` / `host_build_graph` | M0, simulated device, no group controller |
| simgc | `a2a3asimgq` / `group_queue` | M2, simulated device with the GroupQueue |

paged_attention ran 4 reps of 14 rounds, qwen 3; the first two rounds of every
run are dropped, giving 48 and 36 samples per arm. qwen is the in-repo expanded
orchestration (11,087 tasks), which the simulated arms require -- the shipped one
submits each layer as a Graph that a GroupQueue controller has no Graph Execution
to expand. M2 includes `6244769b` (the controller offers both of a package's
vector cores).

| case | base (silicon) | sim (M0) | simgc (M2) | M0 error | M2 vs M0 | M2 vs silicon |
| ---- | -------------- | -------- | ---------- | -------- | -------- | ------------- |
| paged_attention Case1 | 21.173 ms | 22.282 ms | 9.430 ms | **+5.2 %** | -57.7 % | **-55.5 %** |
| qwen3-14B decode, 40L | 34.632 ms | 33.147 ms | 23.779 ms | **-4.3 %** | -28.3 % | **-31.3 %** |

**The M0 error replicates.** An earlier session, before `6244769b`, read +5.0 %
and -4.1 % -- within 0.2 points of these. paged_attention's M2-vs-M0 figure here
(-57.7 %) sits inside its arm's 14.9 % spread of the pooled -56.1 % in
[validation.md](../validation.md), so it is not a different result.

**The error changes sign between the cases, so it is not a bias that cancels.**
aSim's M0 runs 5.2 % slow on paged_attention and 4.3 % fast on qwen. Mixed signs are
the good case -- a uniform bias is the signature of a term the model is missing
-- but the magnitude is an order of magnitude above the ±0.5 % the
`scan_and_claim` gate above recorded, and because it differs per case it does
not cancel in an M0-vs-M2 delta.

**So the correction from "gain against M0" to "gain against hardware" runs in
opposite directions on the two cases.** M0 is slow on paged_attention, so it
flatters the GroupQueue there; M0 is fast on qwen, so it understates it. Both
corrected figures are in the table's last column, and they are the ones to quote
about hardware.

On paged_attention the error is **in the scheduling window, not in fixed cost**:

| | base | sim (M0) | error |
| - | ---- | -------- | ----- |
| `device_wall` | 21.173 ms | 22.282 ms | +5.2 % |
| scheduling window | 18.098 ms | 19.081 ms | +5.4 % |
| remainder | 3.075 ms | 3.201 ms | +4.1 % |

The remainder -- everything `device_wall` holds that the window does not -- is
0.13 ms apart in absolute terms, against 0.98 ms in the window, which is where it
should be, since aSim reproduces that part by
running the same code. So the discrepancy is the device model, not the harness
around it.

Three cautions:

- **The M2 arm's window is not comparable and is left out.** Its `sched_cost`
  read 14.97 ms against a 9.99 ms `device_wall` -- larger than the wall that
  contains it, so the two runtimes do not mean the same thing by that marker.
  Only `device_wall` is quoted for M2 until that is understood.
- **qwen's -28.3 % is softer than the -32.6 % recorded for this same expanded
  path** in [What the GroupQueue is worth](../validation.md#what-the-groupqueue-is-worth-2026-09-21).
  The gap is 4.3 points against 0.8 % run-to-run noise, so it is real. The
  bitmap look-ahead is independently recorded as costing qwen 3.2 %, which would
  put the expected figure near -29.8 % -- close, but this has not been isolated
  on this build, so treat it as the likely cause rather than the established one.
- **Both cases leave the cores far from saturated**, so neither exercises the
  pipeline-overlap assumption. A core-bound case could sit anywhere.

Work equivalence holds on both cases -- compute issued, M0 against M2:

| case | M0 | M2 | delta | completed |
| ---- | -- | -- | ----- | --------- |
| paged_attention Case1 | 91.132 ms | 91.072 ms | -0.065 % | 65,792 / 65,792 |
| qwen3-14B decode, 40L | 746.582 ms | 746.324 ms | -0.034 % | 11,087 / 11,087 |

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

