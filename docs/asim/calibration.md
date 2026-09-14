# aSim calibration — what is measured, and the parameter budget

Every number aSim injects, where it comes from, and which of them are
measured versus fitted. Landing page: [DESIGN.md](DESIGN.md).

## Parameter budget

Every number aSim injects, and where it comes from. **Exactly one is fitted.**

| parameter | value | provenance |
| --------- | ----- | ---------- |
| `push` | 5 ns | measured — the posted doorbell write (§5a) |
| `read` | 195 ns | **fitted** — the in-situ COND poll cost, 2.1x the 92 ns microbenchmark |
| `ack` | 628 ns + per-case setup | measured — two sequential costs, see below |
| `notice` | per-case p1 of (finish − end) | measured, 40–220 ns |
| compute mean, σ | per (case, func_id) | measured from that case's AICore records |

**`ack` spans the whole push→kernel-start path and is the sum of two
independently measured, sequential costs.** The AICPU→core propagation is the
633 ns latch less the 5 ns posted write; the AICore's own dcci+ack before the
kernel begins is carried per task in the record as `receive_to_start_cycles`
(measured 491 ns on Case1, 581 on qwen, up to 1510 on tiny cases — it scales
with how much the kernel invalidates). Compute is `end − start` and deliberately
excludes that setup, so an `ack` holding only the propagation term **dropped the
setup from the model entirely**; that omission was the larger half of the M0 gap.
`asim_gen_calib.py` now emits `ack = 628 + measured setup` per case, so it costs
no tunable parameter.

**Why `read` is fitted but identifiable.** Its poll *count* is measured — the
scheduling window is linear in `read_ns` with a slope implying 0.95 polls per
task — but the per-poll cost is inferred from the residual. It is constrained
rather than free: poll count differs 3.4x between the two validation cases, so a
materially wrong value would produce task-count-scaled errors of consistent sign.
Both instead land within ±0.3 % with opposite signs. Instrumenting the real poll
site under `SIMPLER_DFX` would convert it from inference to measurement, and is
the one open calibration item.

**Sweep in the sensitive regime.** `read_ns` was first swept at 4 threads and
read as *null*, which was wrong: at 4 threads a +40 ns change moves the window
3.5 % against a 3.1 % run-to-run spread — the experiment was underpowered. At 1
thread the parallel term is 83 % of the window and the response is unmistakable.
Pick the operating point with leverage before concluding a parameter is inert.

**aSim computes nothing, so golden is not a gate.** `asim_bringup` allocates the
register table as AICPU-local memory and `kernel.cpp` skips `set_platform_regs`
under `__ASIM_DEVICE__`, so no AICore is ever addressed and no output tensor is
written. Always run `--skip-golden`; the size of a golden diff says nothing
about the simulator.

## 5. What we sample from real silicon

- **Per-kernel compute duration** — Gaussian (mean, σ) per task/kernel class,
  from generous repeated sampling. This is `active.fin_at − active.ack_at`.
  Implemented: the calib carries `func_id mean_ns sigma_ns` and aSim draws a
  per-task duration (§5b). **Calibration is per (case, func_id), not per
  func_id** — the same kernel at different parameters (matrix shape) has a
  different compute time; across the three paged_attention cases one func_id
  ranges 61.6 -> 120.6 us. A calib generated from one case and applied to
  another mis-predicts its makespan by that ratio.
- **Push latency** (ns) — inline cost of the push MMIO write (paper ~64 ns).
- **Status-read latency** (ns) — inline cost of one status-register read
  (paper ~77 ns; our a2a3 ~92 ns `nGnRE` COND read).
- **ACK delay** (ns) — push→ack on an *idle* core (`t_ack`): how fast an idle
  core latches a freshly pushed task. Distinct from status-read latency and from
  compute; needs its own sampling. (On a busy core, ACK is not sampled — it is
  pinned to the active task’s `fin_at` by the auto-ack rule — see
  [device-model.md](device-model.md).)

Distributions, not point estimates: run-to-run variance is real and the whole
premise is faithfulness (cf. the ~3.3% device_wall noise floor already
characterized for hbg; on `scan_and_claim` the real scheduling window
reproduces to ~1.1 % across campaigns).

## 5a. Measured inter-core latency matrix (a2a3 silicon)

The primitive costs the timing model injects. All measured on a2a3 with
**pipelining disabled** (a `dsb`/barrier after each op forces the true
single-op latency, not posted-write throughput) and over many samples so the
noise (stddev) is explicit. Latencies are **half round-trip** (single-hop)
unless noted — a ping-pong measures `elapsed / (2 · iters)`; the 50 MHz sys
counter gives 20 ns/tick.

| path | mechanism | half round-trip | noise |
| ---- | --------- | --------------- | ----- |
| AICPU ↔ AICPU, same package | on-die cache-line ping-pong | **64 ns** | sd ≈ 0 (64–66) |
| AICPU ↔ AICPU, cross package | — | **n/a** — all 4 AICPU cores are one package on this box | — |
| AICPU → AICore poll (COND read) | `nGnRE` MMIO LDR (serialized) | **92 ns** | sd ~1 |
| AICPU → AICore submit, posted | MMIO store throughput | **5 ns** | ≈ 0 |
| AICPU → AICore submit, true completion | MMIO store, barrier-forced | **310 ns** | sd ≈ 0 |
| AICPU → AICore dispatch → core-observed | store + NoC + core poll cycle | **633 ns** | ±236 (CV 37%) |
| AIV ↔ AIV through GM (die-position dependent) | GM + `dcci`/`dsb(DDR)` | **~245–320 ns** | per-pair sd <0.2%; band = die position |
| AICPU ↔ AIV through GM | GM + `dc civac`/`dc cvac` | **527 ns** | sd ~0.8 ns |

Readings that shape the model:

- **On-die AICPU↔AICPU (64 ns) is the fast path**; every GM-mediated exchange
  is 4–8× slower. This is why moving scheduler coordination onto GM (rather
  than keeping it in AICPU-local structures) has a real cost.
- **AIV↔AIV is a die-position spatial gradient, not a same/cross-package
  binary.** The latency is deterministic per physical-core-pair (±0.1 % within
  a run) but is governed by *where* on the die the two cores sit: same-package
  (adjacent-AIV) pairs alone span 248 ns (cores 54↔55) to 318 ns (cores 38↔39)
  by region. Controlling for position, the pure package-boundary effect is
  small and inconsistent — ~1 ns (52–57 region) to ~13 ns (43–48 region) —
  swamped by the ~40–70 ns die-position gradient. Both AIV cores reach each
  other only through shared GM/L2, so co-locating them in one package barely
  helps; physical distance across the NoC dominates. (An earlier one-region
  sample read as a clean "236 same / 260 cross package" split; that 24 ns was a
  distant-core artifact, not a package effect.) Corollary for the seam: a
  GM-resident completion flag does not get cheaper by co-locating cores in a
  package — only by shortening the NoC path, which the scheduler does not
  control (placement is emergent).
- **AICPU↔AIV through GM (527 ns) is the most expensive exchange**, dominated
  by the AICPU's `dc civac` line-invalidate + `dsb sy` + `isb` per iteration
  (heavier than the AICore-side `dcci`). Relevant if a HW-scheduling proposal
  routes AICPU↔core signalling through GM instead of MMIO.
- The MMIO submit has **three distinct numbers** by question asked: 5 ns
  (posted, what the scheduler pays), 310 ns (globally completed), 633 ns
  (observed by the target core). The seam's `asim_push` cost should model the
  5 ns the scheduler actually pays; the 633 ns propagation belongs to the
  core's ACK-delay (`t_ack`), not the push.

**Harness (reproducible).** AICPU↔AICPU + the MMIO probe are a statistical
probe in `scheduler_cold_path.cpp::post_handshake_init` (gated `#ifndef
__ASIM_DEVICE__`, leader-only; peers `usleep` while `latprobe_active_` so the
ping-pong cores are uncontended; threads are placed with
`pthread_attr_setaffinity_np` so role-1 is never starved onto role-0's core).
The GM ping-pongs are TMR examples `examples/a2a3/tensormap_and_ringbuffer/
aiv_pingpong` (two AIV tasks, shared `turn` as input + separate results →
concurrent placement) and `aicpu_aiv_pingpong` (orchestration plays role 0 via
`aicpu_cache_maintenance::invalidate_range_impl`/`flush_range_impl` on the
device GM address; one AIV kernel is role 1). Kernel-side GM coherency uses the
`grid_intrinsic.hpp` pattern: `dcci(ptr, cache_line_t::SINGLE_CACHE_LINE)`
before each read, and pre-dcci → store → post-dcci → `dsb(DSB_DDR)` on write.
`aiv_pingpong` records `get_coreid() & 0x0FFF` per task and buckets latency by
the actual pair (placement is emergent, so trust the recorded ids, not the
intended relationship); a `mode` toggle inserts one filler task to shift the
pair across a cluster boundary, and the kernel does one warmup exchange before
timing to strip the peer-dispatch startup skew. Run each with `--skip-golden`;
results land in `[LATPROBE]`/`[PPPROBE]`/`[PPGM]`/`[PPAICPU]` device-log lines.

## 5b. Compute-duration model — per-task variance

aSim draws each task's compute duration from `N(mean, sigma)` per
`(case, func_id)` rather than returning the mean. `set_compute_ns_table` takes a
sigma array alongside the means; `sample_compute_ticks` draws with a per-core
xorshift64 and an integer Irwin-Hall deviate (three uniforms over
`[-2^20, 2^20)` sum to stddev exactly `2^20`, so scaling by `sigma >> 20` costs
no float and no sqrt on the push path). Calib format is
`func_id mean_ns sigma_ns`; a missing/zero sigma means deterministic.

**Why sigma is load-bearing, not a refinement.** With one duration per func_id
the simulated cores finish in **lockstep batches**, so one completion phase
harvests many tasks and the scheduler enters far fewer dispatch/complete
episodes than real. The per-phase *cost* was already right (~1.0-1.3 us in both);
the entire discrepancy was **phase count**. Measured on paged_attention Case1,
aSim entered 141 dispatch / 629 complete phases against real 420 / 1220; adding
sigma moved it to 470 / 1390 (real 420 / 1220).

**Use a robust (MAD-based) sigma, not the raw stdev.** The observed per-task
duration is right-skewed — its tail is scheduling interference, not compute
variance — and feeding the whole spread in over-fragments the simulated
completions. `asim_gen_calib.py` emits `1.4826 * MAD`. The correction is
selective, which is the evidence it is the right one: strongly skewed
distributions shrink 2-5x (Case2 `func_id=0`, mean 49.7 us / median 41.5 us:
17.4 -> 3.3 us) while a symmetric one is untouched (benchmark_bgemm, 2.68 ->
2.82 us).

**func_id is recorded, not resolved from deps.json.** The collector originally
documented "func_id is resolved post-process from deps.json". That is
structurally impossible for a **graph-execution** case: its tasks are expanded
on device from a cached DAG, so `deps.json` holds only the pre-expansion nodes
(qwen: 45 nodes, 41 of them kernel-less GRAPH containers) while the run executes
19208 tasks whose tokens carry no kernel identity. `ChipSwimlaneAicpuTaskRecord`
therefore carries an `int32_t func_id` (it fits the struct's existing pad, so the
32B wire size is unchanged), set at completion from the first non-negative
`kernel_id` — the same convention the calib keys on. The host emits it as
`aicpu_task_func_ids`, a key **parallel to** `aicpu_tasks` rather than a fifth
column, because existing readers unpack that row as exactly four fields.
`asim_gen_calib.py` prefers it and falls back to the deps join when absent, so
calibration no longer needs a dep-gen round at all.

**Fidelity across the calibration suite** (real vs aSim, per case, 4 threads):
**Fidelity.** Current numbers, case selection and residuals live in
[validation.md](validation.md); they are kept there so this document stays about
provenance rather than results.

**Historical note.** An earlier fidelity table in this section reported five
`host_build_graph` cases with wall errors of −3.7 % to +12.5 %. That baseline is
retired: aSim now measures the scheduling window on `scan_and_claim`, and
`device_wall` was shown to be the wrong gate (see [validation.md](validation.md)).

## 5c. GroupQueue latency model (M2)

The GroupQueue does not exist in silicon, so none of these can be calibrated
against it. Each is anchored to a measured a2a3 figure from 5a and the anchor is
named, so a reader can see which are transplanted measurements and which are
design assumptions. Agreed 2026-09-04.

| path | value | anchor | who pays |
| ---- | ----- | ------ | -------- |
| AICPU → GQ, issue | **5 ns** | posted MMIO store | the AICPU, per submit |
| AICPU → GQ, visibility | **310 ns** | barrier-forced MMIO store | nobody — pipelined, the AICPU issues on |
| core → GQ, FIN report | **150 ns** | the `notice` floor of 5b | nobody — device-internal |
| GQ → AICPU control register | **310 ns** | barrier-forced MMIO store | nobody — the GQ writes it |
| AICPU polls that register | **64 ns** | on-die AICPU↔AICPU | **the AICPU** — the only occupancy term |
| GQ → core, push | **5 / 60 ns** | posted store; per-entry trip for a mix group | an idle core waiting for work |

Two properties of this table matter more than the numbers.

**Only one line occupies the AICPU.** Submit and watermark-write are *visibility*
delays: they change when a task can be granted, or when the manager can learn a
task finished, but they do not hold the AICPU. Modelling them as occupancy would
invent cost that is not there — the same distinction 5a already draws between the
5 ns posted write and the 310 ns completed one.

**The saving is a collapse in poll count, not a cheaper poll.** Today the manager
polls roughly once per task, because it has to ask each core in turn — 62 000
polls for 65 536 tasks. A watermark is read once per scheduler loop and can
retire many tasks at once, so the count drops to the loop count. Estimating the
saving by making each poll cheaper while holding the count fixed understates it
by roughly an order of magnitude.

**The controller↔core link is a design parameter, not a measurement.** A
dedicated on-die control link between adjacent blocks would very likely beat the
figures above; they are anchored to the fastest things actually measured on this
die, chosen because being pessimistic biases against the structure under
evaluation. They belong in the swept set.

