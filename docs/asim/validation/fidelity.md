# aSim fidelity — how close the simulated device is to the real one

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

## The `host_build_graph` baseline (2026-10-07, 4 threads)

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
to expand. Both simulated arms run on the per-thread model clock described in
[DESIGN.md](../DESIGN.md) §3, so their `device_wall` excludes the simulator's own
work exactly.

| case | base (silicon) | sim (M0) | simgc (M2) | M0 error | M2 vs M0 | M2 vs silicon |
| ---- | -------------- | -------- | ---------- | -------- | -------- | ------------- |
| paged_attention Case1 | 21.249 ms | 21.775 ms | 11.327 ms | **+2.5 %** | -48.0 % | **-46.7 %** |
| qwen3-14B decode, 40L | 34.551 ms | 33.116 ms | 37.424 ms | **-4.2 %** | +13.0 % | **+8.3 %** |

**The error changes sign between the cases, so it is not a bias that cancels.**
aSim's M0 runs 2.5 % slow on paged_attention and 4.2 % fast on qwen. Mixed signs
are the good case -- a uniform bias is the signature of a term the model is
missing -- but the magnitude is several times the ±0.5 % the `scan_and_claim`
gate above recorded, and because it differs per case it does not cancel in an
M0-vs-M2 delta. The correction from "against M0" to "against hardware" therefore
runs in opposite directions on the two cases; the last column is the one to quote
about hardware.

On paged_attention the error is **in the scheduling window, not in fixed cost**.
M0's window is on its model clock -- the raw window less the thread's ledger:

| case | | base | sim (M0) | error |
| ---- | - | ---- | -------- | ----- |
| paged_attention | `device_wall` | 21.249 ms | 21.775 ms | +2.5 % |
| | scheduling window | 18.176 ms | 18.575 ms | +2.2 % |
| | remainder | 3.073 ms | 3.200 ms | 0.13 ms apart |
| qwen | `device_wall` | 34.551 ms | 33.116 ms | -4.2 % |
| | scheduling window | 33.525 ms | 31.989 ms | -4.6 % |
| | remainder | 1.026 ms | 1.127 ms | 0.10 ms apart |

The remainder -- everything `device_wall` holds that the window does not -- agrees
to a tenth of a millisecond on both cases, which is where it should be, since aSim
reproduces that part by running the same code. The discrepancy is the device
model, not the harness around it.

**M0's push overrun was most of its paged_attention error.** Before the model
clock, the same comparison read +5.2 % and -4.3 %. The host_build_graph device
did not account the work its push does beyond the 5 ns it models -- drawing a
compute sample, updating the pipeline -- so that work sat inside M0's window. It
is 0.8 ms per thread per round on paged_attention, which pushes 16,448 tasks a
thread, and 0.3 ms on qwen, which pushes 2,772; removing it took paged_attention
from +5.2 % to +2.5 % and left qwen where it was.

**qwen is the core-bound case.** Its cube cores carry 99 % of its compute, and the
busiest of them is busy 30.8 ms of M0's 32.0 ms window; a critical-path trace of
M0 is 97 % compute. So M0 holding -4.2 % there is the validation of the
pipelined, saturated regime that paged_attention, with its cores ~90 % idle,
cannot give. It also bounds what any scheduler can gain on qwen: a few percent.

Work equivalence holds on both cases -- compute issued, M0 against M2:

| case | M0 | M2 | delta | completed |
| ---- | -- | -- | ----- | --------- |
| paged_attention Case1 | 91.130 ms | 91.091 ms | -0.043 % | 65,792 / 65,792 |
| qwen3-14B decode, 40L | 746.107 ms | 746.578 ms | +0.063 % | 11,087 / 11,087 |

## Residuals and known blind spots

1. **One fitted parameter remains** (`read = 195 ns`). It is *identifiable*
   rather than free — poll count differs 3.4× between the two cases, so a
   materially wrong value would give task-count-scaled errors of consistent
   sign, and instead both land within ±0.3 % with opposite signs. It was
   nonetheless fitted *before* `ack` was corrected, so it may have absorbed part
   of the setup term.
2. **Only one core-bound case is validated.** paged_attention leaves its cores
   ~90 % idle; qwen saturates its cube cores and holds -4.2 %, which exercises
   the 2-deep pipeline's overlap assumption -- that `receive_to_start` hides behind
   the previous task's compute. A vector-bound case has not been checked.
3. **Compute-distribution shape is unmodelled.** 1.89 % of real paged_attention
   tasks exceed mean+3σ, which aSim's bounded Irwin-Hall draw cannot produce,
   with tails to 4.6× the mean. Quadrupling σ moved paged_attention's window
   0.4 %, so with its cores idle this cannot supply a meaningful error. On a
   core-bound case such as qwen a missing tail lands on the critical path, and it
   is one candidate for qwen's -4.2 %.
4. **Bias stability across thread counts was the fitness test for A/B use.**
   Before the corrections the error ran −13.9 % at 1 thread to −9.0 % at 4;
   fitting `W(N) = S + P/N` localised it to the parallel term (serial −2.1 %,
   parallel −16.2 %), which is what identified the poll cost as the cause. Any
   future regression should be re-checked the same way, because a bias that
   varies with configuration does not cancel in an M0-vs-M2 delta.
5. **The model clock is per thread.** It is exact for everything a thread's own
   cores do; a thread delayed by another thread's simulator work, through state
   the scheduler threads share, is not corrected. The threads' ledgers diverge by
   at most 0.02 ms (M0) and 0.37 ms (M2) per round on these cases, which bounds
   that residue at about 1-2 % of the window.
