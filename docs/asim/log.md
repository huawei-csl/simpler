# aSim design log


- 2026-09-01 — Doc created; branch `asim-device-emulator` off `main`@95fb4523.
  Design captured; §4 (two-clock) flagged as the decision that gates the rest.
- 2026-09-01 — §4 DECIDED: **Mode A** (real-time fidelity — aSim takes the same
  real wall-clock a real device would). Two pillars fixed: cores as concurrent
  `cntvct` deadlines (no inline compute-spin), per-op MMIO latency as inline
  `cntvct` busy-spin. Scope: dominant MMIO overhead in, scheduler-compute
  offload out. Validated context: upstream `main`@95fb4523 passes all 4 in-repo
  a2a3 HBG scene tests (baseline environment is healthy).
- 2026-09-01 — Device model pinned to the paper’s compute-core state machine
  (§2): added §4a (2-deep pipeline, ACK vs FIN, 4 steady states + 7 transitions
  + 2 forbidden), corrected pillar 1 (per-core state machine, not a single
  completion deadline), split sampling into push/status-read/ACK-delay latencies
  (paper ~64 ns push, ~77 ns read).
- 2026-09-01 — §3: recorded the exact `COND`/`DATA_MAIN_BASE` wire encoding
  (bit-31 state + 31-bit task_id; ACK=0/FIN=1; reserved IDs; idle/exit
  sentinels) from `platform_config.h`, so aSim emits bit-identical status
  words. Seam DECIDED (open Q1): **function calls** (`asim_push` mutates,
  `asim_read_status` advances-then-returns), replacing the passive MMIO
  load/store — the one sanctioned scheduler-side change. Flagged the
  `get_reg_ptr(...)` direct-poke sites (e.g. `scheduler.h:721`) as the concrete
  port task.
- 2026-09-01 — Mapped the platform backends (§4b): `read_reg`/`write_reg` are
  shared; the onboard↔sim split is in `reg_load_acquire`/`reg_store_release`;
  the sim backend runs the real executor on one host thread per core and models
  no timing. Substrate DECIDED (§4c): **onboard / real-AICPU** — a new variant,
  passive per-core state machines (no core threads), with the four bring-up
  deltas (no AICore launch, aSim-memory registers, seed COND=idle, active hot
  poll). Recorded the porting surface (§3).
- 2026-09-01 — **M0 machinery works.** `benchmark_bgemm` on `a2a3asim` (real
  AICPU, simulated cores, golden skipped) passes clean: 1000 tasks dispatched
  through the aSim state machine (`QPROBE rq[AIC] pushes=500`, `rq[AIV]
  pushes=500`), zero deinit timeouts, `device_wall = 1,065,640 ns` vs the real
  benchmark_bgemm `1,086,960 ns` (~2%, *uncalibrated* — compute is a 1000 ns/task
  placeholder; the closeness reflects this kernel being overhead-bound). Bring-up
  bugs fixed along the way: variant registration + the `endswith("sim")` traps
  across `simpler_setup`/`simpler`/`conftest`; `PTO_ISA_ROOT` for the asim
  variant; `set_platform_regs` uint64 cast; ccec/aarch64 toolchains for
  a2a3asim; `_platform_arch`/worker sim-branches; and the deinit exit handshake
  (aSim completes it immediately — no AICore writes `AICORE_EXITED_VALUE`).
  **Next: calibrate the compute-duration model against real per-kernel samples
  and A/B aSim-vs-real device_wall (≥5 reps) for a true M0 validation.**
- 2026-09-01 — **Generalization validated on a 2nd, 4-kernel test
  (paged_attention_unroll_manual_scope).** Its calib table has 4 func_ids with a
  24× compute spread (2.6µs QK-family … 63µs), where core-type/single-constant
  would be badly wrong. aSim reproduces it: sched makespan 1177µs vs real 1120µs
  (+5%). benchmark_bgemm (2 kernels) is +10% — both via their own
  `SIMPLER_ASIM_CALIB` tables. Found+fixed a real bug the multi-kernel test
  surfaced: the `asim_compute_ns_` array was mid-struct in Runtime, shifting
  later members' offsets; the JIT orchestrator (no `__ASIM_DEVICE__`) then saw a
  different layout and populating the array corrupted the shared Runtime
  (benchmark_bgemm tolerated it; paged_attention failed prepare with code 13).
  Fix: the array is placed LAST in Runtime so no existing offset moves.
- 2026-09-01 — **Generalized the compute model to per-func_id with a per-test
  calibration table.** compute is keyed by the kernel a task runs (func_id =
  `kernel_id[slot]`), not core-type. The table is per-test calibration data
  carried host→device: `Runtime.asim_compute_ns_[RUNTIME_MAX_FUNC_ID]` (gated
  `__ASIM_DEVICE__`), filled host-side in `bind_callable_to_runtime` from a
  `func_id ns` file named by **`SIMPLER_ASIM_CALIB`**; the scheduler passes
  func_id through the dispatch seam to `asim_push`, which looks it up (default
  fallback). aSim is HBG-only, so `build_runtimes` builds only host_build_graph
  for the asim variant (TMR's Runtime lacks the field). `asim_gen_calib.py`
  produces the table from a level-3 swimlane + deps. Validated on
  benchmark_bgemm: calib-driven makespan ~566µs matches the hardcoded result.
  **Calibration pipeline for any test:** (1) real run at chip-swimlane level 3
  + a dep-gen run; (2) `asim_gen_calib.py swimlane deps out.calib`; (3) run
  `a2a3asim` with `SIMPLER_ASIM_CALIB=out.calib`.
- 2026-09-01 — **Calibration + makespan decomposition (benchmark_bgemm).**
  Sampled real per-task compute on a2a3 via chip-swimlane level 3 (per-task
  lifecycle + scheduler phases). benchmark_bgemm = 2 kernels: AIC (500 tasks,
  compute mean 21.0µs, occupancy 21.7µs, CV 13%) + AIV (500 tasks, 10.5µs /
  11.3µs, CV 19%); func_id degenerates to core-type here (1 AIC + 1 AIV kernel).
  Real per-task lifecycle (AIC): submit 8.5µs (mostly pipeline-queue, not MMIO),
  recv→start 0.7µs, compute 21.0µs, completion-notice 6.0µs. Critical-path
  (busiest AIC core): span 476µs = 443µs compute (93%) + 33.6µs gaps (7%),
  1866ns/task exposed. Scheduler thread (513µs): dispatch 54% + complete 40% +
  ~6% idle — scheduler ~94% busy while cores 93% busy. HW-headroom on this
  compute-bound kernel ≈ 70µs (13%); task-dense kernels will show more.
  aSim keyed compute per core-type (21658/11313 ns, calibrated) reproduces it:
  scheduler wall 507µs vs 513µs (−1%), complete 40%=40%, per-task aicpu-seen
  26.1µs vs 26.6µs (−2%); one gap — dispatch phase 45% vs 54% (aSim's
  idle-core scan is cheaper; next calibration target). Analyzers:
  `asim_calib*.py`, `asim_decomp.py`, `asim_parallel.py`.
- 2026-09-01 — Wrote `asim_core.{h,cpp}` (the §4a state machine; syntax-clean).
  Mapped onboard bring-up (§4d): the readiness gate is the GM `aicore_done`
  flag, spun on with no timeout, so aSim must publish `aicore_done`/
  `physical_core_id`/`core_type` per worker + seed COND idle; reg bases become
  aSim-owned backing. Recorded the variant scaffolding (§4e) incl. the
  `"a2a3asim".endswith("sim")` name trap. Seam selector: `__ASIM_DEVICE__`
  variant macro (approved).
- 2026-09-02 — **Inter-core latency matrix measured (§5a).** All four
  primitive exchange costs the timing model needs, on a2a3, pipelining disabled
  (barrier-forced single-op latency), many samples with stddev. AICPU↔AICPU
  same-pkg 64 ns (all 4 AICPU cores are one NUMA package — cross-pkg does not
  arise here); AICPU→AICore COND read 92 ns, DMB write 5 ns posted / 310 ns
  completed / 633 ns core-observed; AIV↔AIV GM ~245–320 ns; AICPU↔AIV GM 527 ns
  (AICPU `dc civac` dominates). Harnesses: statistical MMIO+ping-pong probe in
  `scheduler_cold_path.cpp` (peer-park + `pthread_attr` placement to avoid
  self-starvation), and TMR examples `aiv_pingpong` / `aicpu_aiv_pingpong`
  (GM `dcci`/`dsb` kernel side, `invalidate_range_impl`/`flush_range_impl`
  AICPU side). Confirms the seam's push cost is the 5 ns posted write, and the
  633 ns propagation is the core ACK-delay, not the push.
- 2026-09-02 — **AIV↔AIV placement made deterministic; corrects §5a.** Added a
  filler kernel + `mode` toggle to `aiv_pingpong` (one filler shifts the pair
  across a cluster boundary; odd count crosses) and a warmup exchange to strip
  the ~90k-cycle peer-dispatch startup skew. Result: latency is deterministic
  per physical-core-pair (±0.1 %) but governed by die position, not the package
  boundary — same-package pairs span 248→318 ns by region; the pure package
  effect is ~1–13 ns, dwarfed by a ~40–70 ns spatial gradient. The earlier
  "236 same / 260 cross package" was a distant-core (52) artifact of one
  region, now retracted. Bucket by the recorded `get_coreid()&0xFFF` pair;
  placement is emergent so the scheduler cannot pin a NoC path.
- 2026-09-02 — **Latency recalibration + multi-kernel phase-split validation (IN
  PROGRESS — resume point).** Recalibrated the injected MMIO latencies from the
  §5a a2a3 measurements: `aicpu_executor.cpp::asim_bringup` defaults now push=5 /
  read=92 / ack=628 (was 64/77/10), and both `asim_calib/*.calib` LAT lines are
  `5 92 628` (was `64 77 700`). **These edits are uncommitted.** The
  scheduler_cold_path/scheduler_context/aicpu_executor LATPROBE+PPPROBE probe was
  reverted (only the recal remains in aicpu_executor.cpp) so real-a2a3 runs are
  clean.

  Measured scheduler phase splits (chip-swimlane level 3, 4 threads), real vs
  aSim(5/92/628):

  | kernel | src | dispatch | complete | idle | sched_wall |
  | ------ | --- | -------- | -------- | ---- | ---------- |
  | benchmark_bgemm | real | 42.4% | 28.8% | 28.9% | 520 us |
  | benchmark_bgemm | aSim | 28.7% | 32.2% | 39.1% | 498 us |
  | paged_attention_unroll_manual_scope | real | 15.8% | 19.9% | 64.3% | 741 us |
  | paged_attention_unroll_manual_scope | aSim | 11.8% | 15.0% | 73.2% | 565 us |

  Findings: (1) the read=92/ack=628 recal **fixed the complete phase** (bgemm
  30~32% vs real 29%; old read=77/ack=700 over-shot at 40%). (2) The physical
  push=5 **under-models dispatch** (bgemm 29% vs real 42%); the old push=64
  happened to match (~45%) because it absorbed ~59 ns/task of un-modeled
  per-dispatch scheduler CPU work. So `push_ns` conflates the 5 ns doorbell with
  a per-dispatch overhead — the clean fix is a SEPARATE per-dispatch-overhead
  parameter, not inflating the physical push. (3) The earlier memory value
  "real 54/40/6" for bgemm was wrong; clean real is 42/29/29. (4) New gap:
  paged_attention aSim wall 565 vs real 741 us (-24%, vs bgemm -4%) — aSim
  under-predicts the makespan for the scheduler-idle kernel; unexplained (compute
  calib vs real serialization — TODO).

  NEXT: (a) get task counts (bgemm ~1000, paged unrolled) and test whether the
  missing dispatch overhead is a single constant ns/task across both kernels; if
  so add it as a calibration knob. (b) chase paged's -24% wall gap. (c) add the
  scheduler-bound extreme (deepseek_v4_flash_decode, 250 tasks) to anchor the
  high-overhead end. Data on disk: real records under `outputs/*Test*/` (also
  `tmp_real/<ex>/`), aSim under `tmp_asim/<ex>/`; phase split via
  `python tmp_phase_split.py <records.json> <label>`. Run scripts:
  `tmp/real_run.sh` (real a2a3), `tmp/asim_run.sh` (a2a3asim), `tmp/suite_*.sh`;
  all via `tmp/qsub.sh <script> <args>`. Both platforms rebuilt clean this
  session.
- 2026-09-02 — **Per-task compute variance implemented; per-case calibration
  established (§5b).** Two defects found by calibrating across four cases
  (benchmark_bgemm Case0 + paged_attention Case1/2/3), spanning compute-bound to
  scheduler-idle. (1) **Calibration must be per (case, func_id)**: the same
  kernel at different parameters has a different compute time — across the three
  paged cases one func_id runs 61.6 / 99.7 / 120.6 us — and the single
  per-example calib was applying Case1's compute to Case3, which alone caused
  that case's -24% wall error. (2) **The mean-only compute model was the root of
  the phase-count divergence**: identical durations finish cores in lockstep, so
  aSim entered 2-3x fewer dispatch/complete phases than real while the per-phase
  cost already matched. Implemented Gaussian sampling (calib `func_id mean_ns
  sigma_ns`, `Runtime::asim_compute_sigma_ns_`, per-core xorshift64 +
  integer Irwin-Hall in `sample_compute_ticks`), and switched the generator to a
  robust `1.4826*MAD` sigma because the raw stdev is inflated by a
  right-skewed interference tail. Result: max wall error 23.8% -> 10.5%,
  paged Case1 within +0.4% and 13% on phase counts, dispatch gap spread
  +306..+632 -> -33..+304. Residuals in §5b. Note the earlier recalibration
  finding still stands and is now better explained: `push_ns` was never
  under-modeling a constant — the workload-dependence came from the missing
  variance, not from a missing per-dispatch constant.
- 2026-09-02 — **Calibration suite completed (5 cases incl. qwen).** Added
  qwen3_14b_decode, which forced the deps.json assumption to be retired: a
  graph-execution case expands its DAG on device, so its 19208 executed tasks
  have no deps node and only 4 of 45 deps nodes carry a kernel at all. func_id
  is now recorded at completion into the spare pad of
  `ChipSwimlaneAicpuTaskRecord` (32B wire size unchanged) and emitted as the
  parallel key `aicpu_task_func_ids`; wired at all four completion sites
  (a2a3/a5 x hbg/tmr — TMR reaches the descriptor as `slot_state.task->`).
  qwen then calibrates to 37 kernels (e.g. func_id 11, n=2880, 267 us, CV 3.6%)
  and predicts its 38.8 ms wall to **-4.0%**. Suite: qwen -4.0%, bgemm -1.6%,
  paged Case1 +0.2%, Case2 +4.8%, Case3 +11.6%. Also widened the qwen driver's
  `-p` choices to accept the aSim variant of whatever its case declares (the
  scene-test `_ASIM_BASE_PLATFORM` mapping already allowed the case itself).
  Remaining residuals unchanged in kind (§5b): complete phase over-modeled
  everywhere, bgemm/paged-Case3 over-fragment. Real-side run-to-run variance is
  2-8%, which bounds how much of the residual is even meaningful.

- 2026-09-03/04 — **Baseline pivoted to `scan_and_claim`; hbg dropped.** The
  hardware scheduler under study forks from SAC, so the M0 control must too.
  aSim ported onto `scan_and_claim`@`8a740df3` as branch `asim-on-sac`
  (`3d0359dd`). The seam is identical in shape — same five sites across the same
  five files — so the device model ported verbatim; what changed was one
  runtime's worth of hooks and the `a2a3asim` build wiring. Two traps worth
  keeping: a patch can apply cleanly across a 74-commit base gap and still fail
  to compile (context matched, identifiers had been renamed), and
  `"a2a3asim".endswith("sim")` is `True`, which would have routed aSim down the
  host-sim path in three `worker.py` sites.
- 2026-09-04 — **The gate moved from `device_wall` to the scheduling window.**
  On SAC, bgemm's window is 10 % of its wall; the rest is per-run fixed cost
  aSim reproduces by running the same code, so `device_wall` neither
  demonstrates nor refutes anything. Also established that bgemm (128 tasks) and
  paged small1 (4 tasks) cannot measure a window at all. Case set reduced to
  paged_attention Case1 and qwen3_14b_decode.
- 2026-09-04 — **M0 met: −0.50 % (Case1), +0.29 % (qwen).** Two corrections got
  there, and the order matters because the first was diagnosed wrongly at first.
  (1) The in-situ COND poll cost is ~2.1× the microbenchmark value; an initial
  sweep at 4 threads read as null because a +40 ns change moves the window 3.5 %
  against a 3.1 % spread — an Amdahl fit over a `SAC_THREADS` 1–4 sweep
  localised the deficit to the parallel term and pointed at the sensitive
  regime. (2) `receive_to_start` — the AICore's dcci+ack before the kernel
  starts, 491–1510 ns/task — was excluded from compute *and* absent from `ack`,
  so it was missing from the model entirely (`2c7710c8`). Residual signs are now
  mixed and both cases sit under the real measurement's own 1.1 %
  reproducibility.
- 2026-09-04 — **Structure of the problem on SAC, measured.** The scheduling
  window is AICPU-bound: on Case1 the AIC cores are busy 2.8 ms of a 19.7 ms
  window (**86 % idle**) at ~1.2 µs of scheduler work per task per thread, so a
  free scheduler would finish in ~2.8 ms — **~7× headroom**. An earlier reading
  that SAC's counter-based resolution had *shrunk* the headroom was wrong: SAC
  moved the AICPU cost from polling into its scan rather than removing it.
- 2026-09-04 — Doc split: this file plus `device-model.md`, `calibration.md`,
  `validation.md` under a landing `DESIGN.md`, which had reached 781 lines.
