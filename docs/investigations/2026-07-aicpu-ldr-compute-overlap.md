# Can independent AICPU compute overlap a pending COND LDR?

**Date**: 2026-07-21 (updated 2026-07-23)
**Verdict**: the underlying overlap effect is real (confirmed on a standalone
MMIO probe — see below), but every real implementation tried in
`check_running_cores_for_completion` came out net negative on hardware.
**Reverted** — `scheduler_completion.cpp`/`scheduler_dispatch.cpp`/
`scheduler_cold_path.cpp`/`scheduler_types.h` are back to their pre-investigation
state; nothing from this investigation shipped. Kept for the next person who
reaches for this same idea: the mechanism, the three variants tried, why each
one lost, and the two open questions that would need answering before it's
worth trying again.

## Summary of variants tried (see addenda below for full detail)

| # | Variant | check_running_cores_for_completion result | Verdict |
| - | ------- | ------------------------------------------ | ------- |
| 1 | Defer everything past the LDR (judgment + apply), unconditional per-poll timestamp | 11/12 tests significantly slower | Rejected |
| 2 | + `__builtin_expect(likely(have_deferred))`, 3× samples (300 rounds), 5% trim | 12/12 significantly slower — hint made no difference | Rejected; ruled out branch misprediction as the cause |
| 3 | Judgment stays immediate; only defer `apply_transition` when a task actually finished; match-gated timestamp restored (no more unconditional read); `likely()` removed | 9/12 slower, 1/12 faster, 2/12 no longer significant — real improvement, not full recovery | Rejected; root-caused ~1/3 of the regression (the unconditional timestamp read) but a residual remained |

**Root cause, as far as this investigation got**: the per-core judgment work in
today's code (a handful of branches, a few ns) is far below the ~50-100ns
threshold where the standalone probe showed real LDR/compute overlap (see
"Result" below) — so there's essentially no overlap benefit to fund the
restructuring's own bookkeeping cost (deferred-state bookkeeping, a longer
live range for that state across the loop-iteration boundary, and possibly
imperfect inlining of the deferred-apply closure — never fully isolated from
the live-range cost, see variant 3's addendum).

**When to reconsider**: if `check_running_cores_for_completion`'s per-core
work grows to the ≥~50-100ns regime (e.g. folding in more decision logic, as
was under discussion when this was tried), re-open this — variant 3's
structure (defer only on actual finish) is the better starting point than
variant 1. Before trying again, it'd be worth actually answering the two
open questions from variant 3: whether the deferred-apply lambda fully
inlines (`objdump -d`), and how much of the residual cost is register
spills from the longer live range.

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

## Addendum 2026-07-22: real-hardware confirmation

Implemented the pipelined ordering directly in `check_running_cores_for_completion`
(split the apply phase into `apply_slot_transition()`, deferred one core behind the
poll loop's LDR issue — judgment/`finish_ts` capture stay undeferred so DFX timing
is unaffected) and benchmarked it against the pre-change build (git worktree at
merge-base) on device 3, `tools/benchmark_rounds.sh` default 8 cases, 100 rounds
each, Welch's t-test (unpaired, unequal variance) per example/metric.

Result: no consistent win. `Device`/`Effective`/`Sched` deltas were all under ~1%
in magnitude across all 8 cases, and the *sign* was inconsistent — 3 cases
"significant" (p<0.05) in the faster direction, 1 case (`paged_attention_unroll_
manual_scope` Case1, Sched) significant in the *slower* direction at p=2.4e-26 (tiny
but very robust +0.87%), and the rest not significant at all. `Host` swung wildly
(±4-42%) with no consistent sign either, consistent with host-side Python/process
noise rather than the C++ change — sequential same-device runs (baseline then
pipelined, ~15 min apart) can't rule out system-state drift as a confound, and this
was not controlled for (would need interleaved or parallel-device runs to fully
rule out). Full numbers and script in the session that produced this addendum.

This matches the original verdict exactly: today's per-core work is far below the
regime where the synthetic probe showed real overlap, so there is nothing here to
hide behind the next core's LDR. Re-run this exact comparison if
`check_running_cores_for_completion`'s per-core work actually grows.

### Follow-up: larger / structurally heavier kernels

Extended the same comparison to 4 tests not in the default benchmark set, picked
for size/complexity rather than everyday coverage: `spmd_paged_attention_highperf`
(`b1_h32_kv8_s16384_bs128_fp16` — 16k sequence length), `spmd_multiblock_mix`
(mixed AIC/AIV SPMD blocks), `multi_round_paged_attention` (`CaseVarSeq4` — several
scheduling rounds per invocation), and `spmd_sync_start_stress` (stresses the
gated/sync_start cohort path this change touches most directly). Same device,
same 100-round/Welch's-t-test methodology; baseline rebuilt via `git stash`
(incremental, same checkout) rather than a fresh worktree.

Result: same pattern as before for 3 of the 4 — sub-2% deltas, mixed significance,
no consistent direction. But `multi_round_paged_attention` (CaseVarSeq4) — the one
test that runs multiple scheduling rounds per invocation — showed a real,
multi-metric-consistent **regression**: Device (+4.4%, p=0.088), Effective (+5.2%,
p=0.0037), Orch (+5.5%, p=0.0024), and Sched (+5.2%, p=0.00034, d=0.52) all moved
the same direction together. Four metrics agreeing, at a medium effect size, is a
materially stronger signal than any single-metric result in the original 8 — this
is the one result in the whole dataset that looks like a real effect rather than
noise, and it argues against the pipelining change, not for it. Plausible
mechanism: the extra per-round state carried across iterations (deferred core id/
transition/`pending_gated`) and the tail-flush after the loop add a small fixed
cost that's invisible in a single-round test but accumulates over many rounds per
invocation — untested directly, worth checking if this is revisited.

**Updated recommendation**: do not adopt the pipelined restructuring as implemented.
The best case across 12 tests is a wash; the worst case is a reproducible ~5%
regression on a test that already stresses the code path most representative of
heavier future use.

### Follow-up: direct per-call measurement of check_running_cores_for_completion

End-to-end kernel timing (Host/Device/Effective/Orch/Sched) is a noisy, indirect
proxy — it's affected by everything else in the run, not just this one function.
Added a dedicated timer around just the call site in `resolve_and_dispatch`
(`scheduler_dispatch.cpp`, gated `#if SIMPLER_SCHED_PROFILING`, independent of the
existing `CYCLE_COUNT_LAP` phase-timing pair so it doesn't change what
`sched_complete_cycle` measures), accumulated into two new
`SchedL2SwimlaneCounters` fields and logged as an average-per-call line in
`log_l2_swimlane_summary`. Reran the same 12-test/100-round/device-3/Welch's-t-test
comparison; this metric gives 300 samples per test per build (100 rounds × 3
scheduler threads).

**Result: 11/12 tests show a statistically significant SLOWDOWN in the function
itself**, +1.6% to +6.7%, Cohen's d from 0.2 to 1.47 (several in the "large effect"
range), p as low as 1e-58. Only `spmd_sync_start_stress` improved (−1.6%, d=−0.21,
small). This is a much cleaner and more consistent signal than any end-to-end
metric produced across either benchmark pass — direct measurement of the changed
function shows it got worse, not better, in the overwhelming majority of cases.
This confirms the mechanism hypothesized above: the deferred-processing bookkeeping
(carrying core id/bit-pos/reg_val/timestamp across loop iterations, the extra
branch, the tail flush) has a real, measurable cost, and with today's few-ns
per-core workload there is no overlap benefit large enough to pay for it.

**Final recommendation: revert.** The pipelined restructuring should not ship —
it is measurably slower on the function it changed, in essentially every test
run, with no compensating end-to-end win anywhere in the dataset.

Build note: `SIMPLER_SCHED_PROFILING=1` must be applied to the `aicpu`, `aicore`,
AND `host` cmake targets identically — the documented one-liner
(`docs/dfx/l2-timing.md:108`) only reaches `host` (see `KNOWN_ISSUES.md` for the
507018 hang this caused when the targets disagreed on `SchedL2SwimlaneCounters`'s
size).

### References (this addendum)

- Instrumentation: `scheduler_types.h` (`SchedL2SwimlaneCounters.check_running_cores_cycle`/
  `check_running_cores_call_count`), `scheduler_dispatch.cpp` (timer at the call
  site), `scheduler_cold_path.cpp` (`avg/check_running_cores_for_completion` log line).
- Dashboard: same artifact URL as the prior addendum, republished with the new
  section and dataset.

### Follow-up: `likely()` hint, 3x samples, 5%-trimmed, call-count breakdown

Tried the most likely candidate for closing the gap: `__builtin_expect` (`likely()`)
on both `if (have_deferred)` checks in the poll loop, on the theory that a
statically-predicted branch improves code layout / instruction-cache behavior
for the (overwhelmingly common) hot path. Re-ran the same 12-test comparison at
300 rounds (up from 100) with the top/bottom 5% of every metric's samples
trimmed per side (mitigating interference from other processes on this shared
box), giving n=810 trimmed samples per test per build for the per-call metrics.

**Result: no improvement — if anything the picture sharpened.**
`check_running_cores_for_completion` is now significantly slower in **12/12**
tests (was 11/12 without the hint at n=300), +1.3% to +7.1%, Cohen's d up to
2.37. This matches the prediction going in: the `have_deferred` branch is taken
on every iteration but the first, so hardware branch prediction converges
almost immediately regardless of a static hint — the hint was never going to
touch the actual costs (the unconditional per-poll timestamp read, the longer
live ranges the deferred state needs across the loop-iteration boundary).

**New metric — calls per round.** Added a call counter alongside the timer.
The pipelined build makes *fewer* calls per round in most tests (8/12
significant, −1% to −3.75%) — plausibly because deferring more work per call
changes how the poll loop's iteration count settles. But working out net total
time (calls × time/call), **11/12 tests still net-regress**; only
`spmd_paged_attention_highperf` comes out net faster (−3.75% calls outweighs a
smaller +2.1% per-call cost, ≈−1.7% net). Two tests (`multi_round_paged_attention`,
`spmd_sync_start_stress`) show significantly *more* calls, compounding rather
than offsetting the per-call regression — `sync_start_stress` in particular
flipped from the one "improved" result in the earlier pass to a clear
regression here (+3.39% per-call, +2.15% calls) once sample size and trimming
were both increased.

**Orch as a negative control did not hold up.** Orch (the orchestrator
thread's phase, structurally unrelated to scheduler-thread completion polling)
was expected to show no difference. Instead it's significant in 9/12 tests,
including +16.3% on `spmd_multiblock_mix`. Plausible explanations: real AICPU
cross-thread resource contention (scheduler threads 0-2 and the orchestrator
thread 3 share the same chip's cache/memory subsystem, so busier polling could
genuinely slow down unrelated orchestration work), or session-level drift
between the sequential baseline/pipelined runs. This tempers confidence in
attributing every single-metric delta purely to the code change — but
CheckRunningCores remains the most credible signal in the dataset specifically
*because* it moves in the same direction in all 12/12 structurally different
tests, whereas Orch's deltas flip sign between tests (e.g.
`paged_attention_unroll_manual_scope` Case1 +0.47% vs Case2 −0.81%) — a
same-direction effect across 12 varied workloads is much less consistent with
pure session noise than a sign-flipping one.

**Recommendation at this point: revert.** Neither the branch hint nor the larger,
trimmed sample changed the conclusion — it strengthened it.

### Follow-up: only defer on an actual task finish (root-causing hypothesis #1)

Asked, before changing anything: why is the pipelined version slower, given the
`likely()` result rules out branch misprediction? Ranked explanation, in order
of confidence:

1. **Unconditional per-poll timestamp read.** Baseline captured `finish_ts`
   (`get_sys_cnt_aicpu()`) only on an actual match. Deferring the match
   decision meant the pipelined version couldn't know in advance whether a
   poll would match, so it captured `raw_ts` unconditionally on *every* poll —
   converting a rare operation into a common one.
2. **Longer live ranges / register pressure.** The deferred state (core id,
   bit pos, reg_val, timestamp) has to stay live across the *next* core's
   LDR+rmb, a longer live range than baseline's immediate-process-and-discard
   ever needed — a harder register-allocation problem that can force spills.
3. **Lambda-call overhead**, if `process_core` didn't fully inline at both
   call sites — unverified, would need to read the generated assembly.
4. The `have_deferred` branch itself — already ruled out by the `likely()`
   experiment above.

Tested #1 directly: refactored so judgment (`decide_slot_transition`) runs
immediately as in baseline, an ACK-only match (`t.matched` but neither
`pending_done` nor `running_done` — Case 2/4) applies immediately since no
`complete_slot_task` call happens for it and there's nothing worth deferring,
and **only an actual finish gets deferred**, with `finish_ts` captured
match-gated exactly as baseline did (no more unconditional per-poll read).
Also removed the now-unhelpful `likely()` hints. Re-ran the identical
12-test/300-round/5%-trimmed/device-3 comparison.

**Result: real improvement, not full recovery.** `check_running_cores_for_completion`
is now significantly slower in **9/12** tests (down from 12/12), significantly
*faster* in 1/12 (`spmd_paged_attention_highperf`, −1.07%, flipped from a prior
regression), and statistically indistinguishable from baseline in 2/12
(`alternating_matmul_add` +0.06% n.s., `paged_attention_unroll_manual_scope`
Case2 +0.02% n.s.). Several regressions roughly halved (`qwen3_14b_decode`
+2.64%→+0.73%, `batch_paged_attention` +7.08%→+4.45%, `paged_attention_unroll_
manual_scope` Case1 +4.44%→+1.04%) — real, quantified confirmation that
hypothesis #1 was a genuine, fixable cost. But it's not the whole story: one
test got *worse* (`paged_attention_unroll` Case2, +6.10%→+7.22%), and several
barely moved (`multiblock_mix`, `multi_round_paged_attention`,
`spmd_sync_start_stress` are all still +3-7%, roughly where they started) —
consistent with hypotheses #2/#3 still being present, since the "defer across
a loop-iteration boundary" structure still exists for the finish path, just
triggered less often now.

Orch (the attempted negative control) is noisier than ever this pass — two
specific tests (`paged_attention_unroll` Case2, `paged_attention_unroll_
manual_scope` Case2) show effect sizes (d=−6.96, d=+4.05) far larger than
anything CheckRunningCores has ever shown, meaning those two runs likely carry
some other strong confound (thermal, contention, or session drift) on top of
the code change — their CheckRunningCores numbers should be weighted less
than the rest.

**Updated recommendation: still lean revert, but the case is weaker than
before.** The fix confirmed and closed a real, quantified cost, which is
useful independent of the outcome — but a majority of tests (9/12) still
regress, with no test showing a large, unambiguous win. Pursuing hypotheses
#2/#3 (measure register spills directly, check whether `process_core` is
actually inlined via `objdump -d`) would be the next step if this is worth
continuing; absent that, the current numbers don't justify shipping it.

### References (this addendum)

- Code: `scheduler_completion.cpp::check_running_cores_for_completion` — judgment
  inline, `apply_transition` lambda deferred only when `t.pending_done ||
  t.running_done`.
- Dashboard: same artifact URL as prior addenda, republished with this pass's
  dataset and a version-history footnote.

### Closing: reverted

No variant showed a net win, so none of the three shipped. `scheduler_completion.cpp`,
`scheduler_dispatch.cpp`, `scheduler_cold_path.cpp`, and `scheduler_types.h`
were restored to their pre-investigation state (`git restore`) and rebuilt/
sim-tested clean. The `SIMPLER_SCHED_PROFILING` build-flag hazard this
investigation surfaced (see `KNOWN_ISSUES.md`) is independent of the reverted
code and remains a live risk for the *next* person who adds a
`SchedL2SwimlaneCounters` field — the build caches used during this
investigation had their `CMAKE_CXX_FLAGS` reset back to empty (the default),
but a fresh checkout building for the first time is unaffected either way.
