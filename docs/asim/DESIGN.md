# aSim — an AICPU-hosted simulated device for scheduling-overhead studies

**Status (2026-09-04): M0 credibility gate met.** The `a2a3asim` variant runs
the real `scan_and_claim` scheduler on the real AICPU against simulated cores,
and reproduces the measured scheduling window to **−0.50 %** (paged_attention
Case1) and **+0.29 %** (qwen3_14b_decode) — both inside the 1.1 % run-to-run
reproducibility of the real measurement itself. See
[validation.md](validation.md).

**Branch:** `asim-on-sac`, off `scan_and_claim`@`8a740df3`.

**The baseline is `scan_and_claim`, not `host_build_graph`.** The hardware
scheduler under study forks from SAC, so the M0 control must too; hbg is out of
scope entirely. Where this document or its children still say "HBG", read it as
history.

Living document — update on every design refinement and every measured result.

## Map

| Document | Contents |
| -------- | -------- |
| [device-model.md](device-model.md) | The seam and its wire contract, the per-core state machine, substrate and bring-up, `a2a3asim` build scaffolding, the GroupQueue |
| [calibration.md](calibration.md) | Every injected number, its provenance, and which are measured vs fitted |
| [validation.md](validation.md) | Method, case selection, the measured GroupQueue deltas, and which M2 campaign is which |
| [validation/fidelity.md](validation/fidelity.md) | How close the simulated device is to silicon, per baseline, and the residual blind spots |
| [validation/transport-ablation.md](validation/transport-ablation.md) | **Why the GroupQueue wins even against a zero-latency AICPU-to-AICore connection** -- the central argument, with the ablation behind it |
| [validation/m2-scheduling-window.md](validation/m2-scheduling-window.md) | M2 against `scan_and_claim`, gated on the scheduling window (2026-09-04) |
| [validation/m2-group-queue.md](validation/m2-group-queue.md) | M2 with the task-grouping contract, gated on `device_wall` (2026-09-14) |
| [log.md](log.md) | Dated design log |

## 1. Goal

Estimate, as faithfully as possible, the **positive** device-performance impact
of dedicated **hardware-based scheduling structures** — specifically their
effect on the scheduling overheads the AICPUs currently incur.

The vehicle is a **simulated device (“aSim”)** that runs on the AICPUs in place
of the real AICore fabric. aSim does not compute; when a task is dispatched to a
(virtual) core it returns a **sampled execution time** drawn from a distribution
learned by generous sampling on real silicon. With a faithful baseline in hand,
we then introduce **simulated hardware scheduling elements** and read off their
impact on predicted device time.

The device model (§3, §4a) follows the compute-core state machine described in
*Accelerate the Accelerator: Maximizing scheduling performance of Ascend NPUs*,
§2 — the authoritative description of the AICPU↔core register protocol, the
2-deep task pipeline, and the core state machine aSim must reflect.

## 2. Core principle — strict separation of concerns

- **The scheduler is untouched.** The a2a3 `scan_and_claim` scheduler must run
  as if driving real hardware — ideally copy-pasted verbatim. No simulation
  awareness leaks into scheduler code.
- **All simulation machinery lives on the aSim side**, including *all
  timekeeping*. The scheduler never reads a virtual clock; it only issues the
  same hardware ops it issues today.

## 3. Timing model — DECIDED: Mode A (real-time fidelity)

aSim behaves as exactly like a real device as possible: it takes the **same
time** a real device would. The scheduler runs for real on the AICPU (its own
compute is incurred naturally), and aSim reproduces the device’s contribution —
compute time and MMIO latencies — on the scheduler's own timeline. Makespan is
the scheduler’s `device_wall` marker.

**aSim's time is each scheduler thread's model clock: real time less that
thread's ledger** — every tick its model has spent beyond the latencies it
charged. The model runs on the same AICPU thread it serves, and its own work can
exceed the latency it models: a status read the GroupQueue charges at 10 ns does
a retirement's worth of bookkeeping. The ledger takes that excess out of the
modelled world, so the cores stand still while it accrues, and the phase recorder
removes the same excess from `device_wall`. The removal is exact because the
cores did not move: subtracting it from a timeline the cores kept running on
would delete time they spent computing, and on a busy-core case would report
less than the work they carried.

The clocks are per thread, which makes them exact for everything a thread's own
cores do. A thread can still be delayed by another thread's excess, through the
shared state the scheduler threads coordinate on; that residue is bounded by how
far the threads' ledgers diverge, which `[SIM_WINCORR]` and `[ASIM_WORK]` report
per thread. Measured on the two in-scope cases it is 0.2–0.4 ms per round, about
1–2 % of the window.

### What Mode A measures — and what it cannot

On `scan_and_claim` the overhead splits differently than it did on hbg, and the
difference matters for what Mode A can measure. Measured on paged_attention
Case1 (65 536 tasks, 4 threads):

- The scheduling window is **19.7 ms while the AIC cores are busy only 2.8 ms** —
  **86 % idle**. The window is AICPU-bound, at **~1.2 µs of scheduler work per
  task per thread**, and a free scheduler would finish in ~2.8 ms. That is **~7×
  headroom** for any structure that feeds the cores faster.
- Completion polling is **~0.95 polls per task**, worth **15 % of the window at
  4 threads and 23 % at 1**. Real, but no longer the majority as it was on hbg
  (~58 % there). SAC moved the cost into its scan rather than removing it.

- **In scope:** structures that cut the MMIO interaction (HW ready-queue,
  smarter doorbell, parallel multi-core poll), measured by changing the injected
  latency and reading the makespan delta.
- **Out of scope:** offloading the scheduler’s *own* CPU work. Mode A has no
  lever on it without editing the scheduler, which would break the
  “scheduler unchanged” principle — and on SAC that work is now the *majority*
  of the window, so this exclusion costs more than it did on hbg. A structure
  that removes scan work (as the GroupQueue design does, by pushing to cores
  itself) can only be evaluated by running its own scheduler, not by tuning a
  latency.

### Two implementation pillars (fidelity depends on both)

1. **Cores are concurrent state machines driven by model-clock deadlines, never
   inline-spun.** On a push, aSim admits the task into core C’s state machine
   ([device-model.md](device-model.md)) and schedules its **ACK** and **FIN** absolute deadlines on the
   thread's model clock, then returns immediately. It must **not** spin the
   compute duration inline — that would serialize all “compute” onto the
   scheduler thread and destroy the parallelism that sets makespan. A status read
   evaluates the machine against the model clock. N cores thus “run”
   concurrently, exactly like silicon; ACK/FIN-notice latency emerges from when
   the scheduler next polls C.
2. **Per-op MMIO latency is the inline part — a `cntvct` busy-spin, not a
   sleep.** Each `read_reg`/`write_reg` against aSim busy-spins the calibrated
   latency (a2a3 real poll ≈ 92 ns `nGnRE`; plus the doorbell-write cost). Spin,
   not sleep — codestyle rule 5 forbids sleeping on the dispatch path, and a
   spin gives ns precision. Work that does not fit inside the latency goes on the
   thread's ledger rather than into the modelled world. These injected latencies are simultaneously the
   fidelity mechanism **and** the experiment knob (zero the poll cost → measure
   the makespan drop from a HW structure that removes it).


## 4. Milestones

- **M0 — faithful baseline.** aSim reproduces the real scheduling window within
  the measurement's own run-to-run band, driven by the unmodified SAC scheduler.
  This is the credibility gate; nothing after it is trustworthy until it holds.
  **Met 2026-09-04** — see [validation.md](validation.md).
- **M1 — instrument the seam.** Sample submit/poll latencies; confirm the
  simulated interface reproduces the real interface's contribution. Largely
  subsumed by the M0 calibration work; the one open item is measuring the
  in-situ poll cost directly (see [calibration.md](calibration.md)).
- **M2 — inject simulated HW scheduling structures** and measure predicted
  device-time deltas against M0. The structure is the **GroupQueue** — a
  per-group ready-task ring fed by one AICPU, whose controller holds every core's
  state and pushes work to it, with completion reported as a watermark register
  rather than per-core polling. Its model is in
  [device-model.md §5](device-model.md#5-the-groupqueue-the-m2-structure); the
  measured deltas are in [validation.md](validation.md).

## 5. Open questions

1. **The one fitted parameter.** `read = 195 ns`, the in-situ COND poll cost,
   is 2.1× the microbenchmark value. Its poll *count* is measured; its per-poll
   cost is inferred. Instrumenting the real poll site under `SIMPLER_DFX` would
   convert it to a measurement. See [calibration.md](calibration.md).
2. **M2 cannot keep the scheduler unchanged.** The GroupQueue replaces
   `push(core, task)` / `read_status(core)` with `submit(task)` /
   `read_watermark()`, so M0 and M2 are different programs. The M0 tree stays
   runnable as the control rather than being migrated.
3. **Cross-shape fidelity is validated on two cases.** Both have large task
   counts; a *core-bound* case (cores near saturation) is untested, and it is
   the regime where the `receive_to_start` overlap assumption in the 2-deep
   pipeline would actually bite.
4. **Where cohort agreement lives is a decision, not a finding.** A
   `require_sync_start` cohort spans all four queues, so co-residency has to be
   agreed above any one of them. The alternatives were a device-level cohort
   barrier — new hardware that would remove the AICPU from it entirely, and the
   place the GroupQueue is most likely to show a benefit — or keeping the
   agreement in software. Software was chosen, which makes the M2 number on
   cohorts unchanged by construction. The barrier remains the interesting
   unmeasured option.
5. **aSim's own per-call cost is part of the M2 delta.** M0 and M2 are different
   programs, so simulator work that does not fit inside the latency it models
   lands in the result rather than cancelling. Three such leaks have been found
   and removed, the largest worth 58% of the window; the `[ASIM_POLL]` device-log
   line is the check. See [validation.md](validation.md).
6. Determinism: seed control for the samplers, so a prediction is reproducible
   while still spanning the sampled distribution across reps.
