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

## M2 — what the GroupQueue is worth

Measured the same way as M0: the scheduling window, 4 threads, 14 rounds,
median over steady rounds, both arms on the same calibration table and in the
same session. M0 is the `a2a3asim` control (`scan_and_claim` + simulated cores);
M2 is `a2a3asimgq` (`group_queue` + the GroupQueue device).

| case | M0 | M2 raw | M2 overrun | M2 corrected | delta |
| ---- | -- | ------ | ---------- | ------------ | ----- |
| `qwen3_14b_decode` | 43 753 µs | 47 048 µs | 1 276 µs | 45 772 µs | **+4.6 %** (raw +7.5 %) |
| paged_attention Case1 | 20 861 µs | 20 627 µs | 2 224 µs | 18 403 µs | **−11.8 %** (raw −1.1 %) |

All four arms measured back to back in one session, nothing else on the device.
Do not compare either case against a figure from another campaign: M0's own Case1
reading moved 19 432 → 20 861 µs between them.

**The correction, and why only one arm gets it.** `M2 overrun` is the mean per
manager per round of the simulator work that did not fit inside the latency it was
modelling, which is the only part that inflates a window — work that fits is
hidden by the spin to the deadline. Subtracting it is first-order: the overrun is
interleaved, and delaying a manager can push work past a core's idle moment, so
the corrected figure is a **lower bound on the structure's benefit**, not an exact
value. M0 is not corrected because it does not overrun: measured at 0–14 µs
(0.07 %) on Case1, and its read advances one core in 15–27 ns against a 92–195 ns
budget on both cases, so it has no room to.

Case1 inverts under the correction, and that is the honest reading of it: raw it
looks like a wash, but 2 224 µs of its 20 627 µs M2 arm is simulator cost the
hardware would not pay.

qwen was +11.1 % until each ring was bounded to what its own cores can run. See
*Intake is a commitment horizon* and *The arms are not taxed alike* below.

**Run-to-run spread, on repeats of the same binary: qwen 0.4 %, Case1 3.0 %**
(M0's Case1 spread is 0.8 %). Nothing below those thresholds is a result, which is
why qwen carries the headline. Six back-to-back Case1 pairs read +1.7 %, +5.3 %,
+1.9 %, +0.7 %, −0.5 %, −2.3 % — the trend tracks the simulator-cost reductions
below rather than any scheduling change, since the scheduling behaviour was
identical across all six.

The one unambiguous secondary effect: **completion reads collapse.** A manager
polls its queue ~850 times for ~16 000 dispatches — one read per ~19
retirements — where the per-core poller read once per core per pass.

### Three corrections to earlier numbers

Every one of these inflated the GroupQueue's apparent benefit, and all three were
defects in the M2 arm rather than properties of the structure. They are recorded
because the same shapes will recur.

1. **An idle core reported itself free "since forever."** A lazy advance clears a
   core's slot as of the poll, not as of the finish, so a task granted to it took
   its start from the entry's arrival and could begin before the core's previous
   task had ended. Worth **7.3 %** once a core reported its true free time.
2. **M2 skipped a scan pass M0 performs.** While MIX was refused, dispatch never
   asked for it, so M2 did not pay the MIX discovery scan that `scan_and_claim`
   pays on every loop. Worth **9.2 %** once MIX was modelled and the pass
   restored.
3. **The simulator's own cost was most of the window.** See below.

### aSim's own cost has to fit inside the latency it models

M0 and M2 are different programs, so simulator work that does not fit inside the
latency it models lands in the result rather than cancelling. This is the largest
hazard in the whole method, and it has bitten three times:

| leak | cost |
| ---- | ---- |
| a poll walked all 72 cores to find the 18 its queue serves | 4.1 % |
| arbitration walked all 24 packages to find its own 6 | ~10 % |
| every poll replayed the whole device to catch it up | ~58 % of the window |

The third was structural rather than a slip. Advancing a shared machine on every
access costs work proportional to the machine, not to what happened, so a poll
modelling 64 ns took 4.8 µs — 74× its own budget. Replacing it with a per-manager
event schedule (see
[device-model.md §5](device-model.md#5-the-groupqueue-the-m2-structure)) cut
total simulator work from 253 ms to 3.2 ms per manager, about **1.2 %** of the
window.

**Check this before quoting any M2 number.** The `[ASIM_POLL]` line in the device
log reports, per manager, the calls made and the mean and worst work inside them.
Mean work far above the modelled poll latency means the window is measuring the
simulator.

### qwen's residual is real, and not yet explained

The cost is not simulator overhead: it was checked first. Folding the cohort's
two queries into the status read the manager already makes — so assembly costs it
no access it was not already making — moved the number 0.4 % (53 070 → 52 871 µs)
and brought poll work to a mean of 100 ns against the 64 ns modelled, 0.3 % of the
window. The measurement is sound; the cost is the design's.

**Seven attempts have failed to move it**, which is worth recording because each
ruled out a plausible cause:

| attempt | qwen |
| ------- | ---- |
| baseline | 52 871 µs |
| a manager withholds ordinary work while a cohort assembles | 52 862 µs |
| ...and nothing is gated until the device can seat the cohort whole | 53 026 µs |
| cohort members stage behind running work instead of on idle packages | 53 060 µs |
| MIX grouping removed entirely | 52 833 µs |
| report latency taken to zero | no change |
| the controller absorbs the package frontend controllers | 48 758 µs (from 48 865) |

The staging attempt was the strongest hypothesis and the most wrong.
`scan_and_claim` never drains the device for a cohort — `count_global_available`
passes `include_pending`, and staging goes into the pending slots of busy cores —
so the GroupQueue's demand for idle packages looked like a pure artefact worth 40
device drains per round. Removing it changed nothing. The cost is not in how a
cohort is assembled, and not in the drain.

**Head-of-line was the leading candidate and is now excluded.** `mix_head` is 0 on
every manager, `idle_aic` is 0 at every stall, and every stalled poll is the *cube*
ring blocked with all six AIC cores full. Nothing is misplaced and nothing is held
up behind an unplaceable head.

**Only one change has ever moved this number: the look-ahead buffer** (−6.3 %),
which let a finished task retire before the tasks in front of it. That points at
the completion path rather than at placement or throughput, and the loop split
agrees:

| | M0 | M2 |
| --- | --- | --- |
| AICPU work (complete + dispatch) | 18.9 ms | 13.6 ms |
| idle | 24.5 ms | 34.3 ms |
| ready → dispatch | 62 µs | 22 µs |
| AIC occupancy | ~70 % | ~63 % |

M2 does a third less AICPU work and dispatches ready tasks three times faster, then
idles 9.8 ms longer while its cube cores go hungry. The binding path is upstream of
"ready" — how quickly a finished task reaches the manager — not how quickly the
manager places one.

The buffer's own depth is *not* the remaining bound: instrumented, it reaches 17 of
18 on qwen and 9 on Case1 and **never overflows**, so no finish ever falls back to
waiting for the watermark.

### The finish -> ready chain, split at the manager's door

A consumer cannot be dispatched until its last producer's completion has
travelled from the core that ran it to the manager, and the manager has resolved
the fanout that releases it. Both arms are instrumented identically, and the
accumulation counts **only retirements that released a consumer** — one that
releases nobody costs the makespan nothing. qwen, per manager per round:

| segment | M0 | M2 | delta |
| ------- | -- | -- | ----- |
| fin -> retire (n ~ 4 800) | 2.62 µs | 3.16 µs | **+0.54 µs** |
| retire -> ready (n ~ 425) | 187 ns | 186 ns | **0** |
| ...of which the fanout walk | 126 ns | 124 ns | 0 |
| ...and the work before it | 60 ns | 61 ns | 0 |
| fin -> ready (releasing edges) | 3.17 µs | 4.14 µs | +0.97 µs |

`resolve_fanouts` over *every* completing task, not just releasing ones, is 94 ns
against 100 ns — the same walk on the same graph, as it must be.

**The whole difference is the trip to the manager; the manager's own work is
identical.** 0.97 µs × ~425 edges is 0.41 ms per manager per round against a
5.1 ms gap, so the chain accounts for under 10 % of it even counting the ~110
releasing fanouts per manager that resolve outside any retire window. **Over 90 %
of the gap is not per-edge latency at all.**

The +0.54 µs on the trip is what the structure trades for: M2 reads one register
per scheduler loop and retires many positions from it, where M0 reads every core
in turn and catches each finish sooner. It is 7× the 460 ns of modelled hardware
propagation, so most of it is waiting for the next poll rather than the wire.

#### A retire window is required, and its absence is not a small error

`on_mixed_task_complete` has call sites that are not the continuation of a
retire: a completion deferred to the mailbox, and a graph node publishing its
outer ring task. Measuring `retire -> ready` against a timestamp held in a
per-thread variable therefore charges those fanouts **the gap since that manager's
last retirement**, which is not a quantity that means anything.

It is not a small error and it does not cancel between the arms. M2 idles 34 ms
of a 48 ms window where M0 idles 24 ms of 43, so the same flaw inflates M2 far
more. Uncorrected it read M0 2.08 µs against M2 3.87 µs and made the manager's own
fanout resolution look like two thirds of the chain delta; corrected, the two are
187 ns and 186 ns and the difference is zero. The tell was the spread: M2's four
managers reported 2258–5535 ns where M0's reported 1172–1641, which is too wide
to be work.

The fix is to open the window immediately before a completion, close it after,
and count what falls outside (`orphan_n`, ~110 per manager per round in both
arms) rather than charging it.

The three gates below are also why the residual imbalance has to be closed by
*placing* work evenly rather than refusing it — and the one design that does that,
letting any manager submit to any group, is blocked by `require_sync_start`: a
cohort needs 24 wholly-free packages device-wide and its staging assumes a manager
owns its own, which stops holding the moment any manager's work can occupy any
group's packages. qwen carries 40 cohorts a round, so that is fatal rather than
awkward.

Nor is the manager's reaction time the whole story, measured on its own as the
wait between a task becoming reportable and a poll making it legible:

| case | mean | worst | per manager |
| ---- | ---- | ----- | ----------- |
| qwen | 1.6 µs | 19 µs | 4 800 retirements |
| Case1 | 12 µs | 73 µs | 16 400 retirements |

**Case1 carries eight times the lag and is the neutral case.** So the lag is not
the mechanism on its own — it costs only where something is waiting on it, and
Case1's cores are 86 % idle. The chain measurement above separates the fraction of
it that lands on the dependency critical path, and that fraction is small.

### Intake is a commitment horizon, and bounding it is worth 4.6 %

Submitting a task to a GroupQueue binds it to that group's 18 cores; no other
group can ever run it. So whatever bounds a manager's intake also bounds how far
ahead it commits work, on information about which group will have a core free
that is that many tasks stale.

Sizing intake by the index window — 36, which is `scan_and_claim`'s in-flight
capacity, where every outstanding dispatch is *on a core* — lets whichever manager
drains fastest accumulate work its own group cannot start. It then sets the
makespan while the others idle. Bounding each ring to the cores it feeds (one
queued entry per core, so three tasks per slot: running, pipelined, queued) took
qwen from 48 737 µs to **46 471 µs**:

| | index window | ring bounded to cores |
| --- | --- | --- |
| window | 48 737 µs | **46 471 µs** |
| ring spread, mean / max | 6–9 / 24 | **2 / 6** |
| ring wait per task | 34–41 µs | **12 µs** |
| AIC busy, max over mean | +9.7 % | **+4.3 %** |

The direction is confirmed from the other side: **widening** the window to 72
spread per-manager task counts from 8 % to 47 % and cost 32 % (48 737 → 64 329 µs).

### Deferring to peers is a net loss, monotonically

Bounding intake leaves ~4.3 % imbalance, and three attempts to close it by having
a manager *withhold* work when a peer had a better place for it all made the
window worse — each one improving balance and losing more than it gained:

| | window | imbalance | pipeline wait | AIC occupancy |
| --- | --- | --- | --- | --- |
| ring bound only | **46 471 µs** | +4.35 % | **19.5 µs** | **66.2 %** |
| defer if a peer has a free core | 47 526 µs | +3.10 % | 11.9 µs | 64.7 % |
| defer if a peer holds less load | 47 953 µs | **+1.76 %** | 10.6 µs | 64.1 % |

**Any test that fires when a group's own cores are full fires almost always**,
because being full is what makes a manager the most loaded — so every such gate
blocks pipelining outright and the cores run dry. Peers pop from the same shared
ready queues, so withholding work helps only when work is scarce and idles this
group's cores when it is not.

The remaining imbalance therefore has to be fixed by *placing* work evenly rather
than by refusing it.

### The arms are not taxed alike, and only the excess matters

aSim's modelled latency is honest only while the simulator's own work fits inside
it: a call does its work, then spins to the deadline, so work that fits is hidden
and costs the measurement nothing. **Only the part that overruns the deadline
inflates a window**, which makes the useful figure the overrun, not the raw work
and not the work as a share of the window.

The two arms overrun very differently, because their device calls do different
amounts of work for the same modelled cost:

| | work per call | modelled | overruns |
| --- | --- | --- | --- |
| `scan_and_claim` | 15-27 ns | 92-195 ns read | never |
| `group_queue`, qwen | ~237 ns | ~64 + 64 per look-ahead word | ~70 % of polls, **2.7 % of the window** |
| `group_queue`, Case1 | ~1 600 ns | same | nearly every poll, **11 % of the window** |

`scan_and_claim` advances one core per read, so its work is O(1) and always fits.
A GroupQueue poll retires everything that has come due — ~0.3 positions per poll
on qwen but ~11 on Case1 — so its work scales with retirement density, and 89 % of
it is retirement at ~133 ns a position. **Real hardware does that work inside the
queue block, in parallel and off the AICPU.** aSim does it serially on the thread
being measured, so the M2 arm is charged for it and the M0 arm is not.

Four reductions to aSim's bookkeeping — none touching the modelled device — took
per-poll work from ~2 000 ns to ~1 600 ns and moved Case1 from +1.7 % to
−0.5 %..−2.3 %. What remains is structural: closing it means moving retirement off
the measured thread, not optimising it further.

**Read `[ASIM_OVERRUN]` before quoting any M2 number**, and prefer it to
`[ASIM_POLL]`: the latter reports total work, which over-states the problem where
the work fits and says nothing about where it does not.

### MIX is modelled; sync_start is what qwen needed

qwen's task 16 is *both* a MIX and a `require_sync_start` cohort; modelling MIX
alone got it past the first and straight into the second. Both are now modelled,
so **the M2 gate is two cases**, and they disagree: Case1 is homogeneous,
AICPU-bound and cannot resolve the effect against its own 2.6 % spread; qwen is
heterogeneous, carries 40 device-wide cohorts per round, and reads +11.8 %. Neither
is a core-bound case, which is still untested — though qwen's cube cores run ~63 %
busy, so it is the closer of the two.

### What the delta holds equal

In-flight capacity: the live index window is twice a manager's core count, which
is exactly what the 2-deep per-core pipeline allowed. A deeper queue is a
separate experiment.

## M2 — the ready group queue (2026-09-14 campaign)

A separate campaign from the one above, and not comparable to it: that one reads
the scheduling window against a `scan_and_claim` M0, this one reads `device_wall`
against an M0 of `host_build_graph` + simulated cores (`a2a3asim`). Both arms
here are `a2a3asimgq`; what changes between them is the grouping contract.

**The structure.** The shared ready queue holds *groups*, not tasks. A group's
dependencies are on sink tasks only, so a group no external edge reaches is ready
from the start; a scheduler thread takes one and feeds the whole of it to its
controller, internal edges included. paged_attention Case1 annotates as
`group = local_id / 257`: 65 792 tasks in 256 components of exactly 257,
contiguous in local id, with zero edges between them.

**Positions are taken and filled in the same step.** A group is fed in ascending
task order, and each entry takes its position immediately before being submitted.
Ascending order is what lets a controller hold an internal edge at all — a
consumer can only name producers that already hold positions — and it does not
depend on the order the per-shape rows drain. Taking and filling together is what
keeps the manager retiring: it harvests completions as a contiguous prefix, so a
position reserved now and submitted later halts that prefix at itself and nothing
behind it ever retires. An earlier build reserved a whole group up front and
measured `push=3840, watermark=0` — every thread wedged at the first hole.

**Hardware limits.** 32 task slots per controller, 4 dependency comparators per
slot. A slot is occupied from submit until the *manager learns* the task finished,
by watermark or ahead-notification — an entry the controller has finished but not
yet reported still holds its slot. A task with more unmet producers than there are
comparators cannot be expressed, so the manager holds it, and with it the rest of
its group, which is fed in order.

| config | PA `device_wall` | vs M0 |
| ------ | ---------------- | ----- |
| M0 (`a2a3asim`) | 21.907 ms | — |
| M2, grouping off | 18.200 ms | −16.9 % |
| M2 ready group queue, unlimited | 7.794 ms | −64.4 % |
| **M2 ready group queue, 32 slots / 4 deps** | **9.87 ms** | **−55 %** |

The 32/4 figure moved from 9.453 ms as the queue was made correct on a graph of
chained groups (see qwen below): removing a whole-graph pre-scan from seeding took
it to 8.575 ms, covering the position table properly put it back to 9.35 ms, and
handing groups out concurrently costs the rest.

**Same work, verified.** A skip-golden run proves only that nothing deadlocked,
so work equivalence is measured rather than assumed: compute issued by the
controller, dispatches charged, and tasks completed, all at `--rounds 1` so no
counter's reset semantics can be mistaken for a difference.

| | compute/round | dispatches | completed |
| - | ------------- | ---------- | --------- |
| M0 | 97 030.0 µs | 63 701 | 65 792 / 65 792 |
| M2 grouping off | 97 005.7 µs | 62 923 | 65 792 / 65 792 |
| M2 group queue, 32/4 | 96 982.1 µs | 63 898 | 65 792 / 65 792 |

Compute agrees to −0.06 %. The mechanism shows up in the balance rather than the
total: per-thread compute spread is 5.6 % ungrouped and 0.1 % grouped, which is
the same "fill the idle cores" effect the zero-latency balancer oracle found.

**The comparator count is the parameter that decides the design.**

| comparators | PA `device_wall` | stalls forced |
| ----------- | ---------------- | ------------- |
| 1 | 28.254 ms | 293 843 |
| 2 | 22.549 ms | 195 025 |
| 3 | 9.690 ms | 0 |
| 4 | 9.453 ms | 0 |

The cliff is between 2 and 3, not a gentle curve: at 1 comparator the group queue
is *slower than not grouping at all* (18.200 ms) and slower than M0, because a
task that cannot express its producers holds everything behind it in its group's
feed order, and the ungrouped path pays no such penalty — it can dispatch any
ready task from anywhere. PA needs 3; 4 buys margin. The slot count is the milder
knob: 32 versus unlimited costs 21 %.

### qwen decode — where the contract stops working

paged_attention is the shape the contract is built for. qwen decode is not, and
running it is what says which of the two the structure needs.

Its graph has to be expanded first: the shipped orchestration submits each of the
40 decoder layers as one Graph the device Scheduler expands, so dep-gen records 45
outer nodes rather than tasks. Expanded (`decode_fwd_layers_expanded.cpp`) it is
**11,085 tasks, 23,601 edges, all forward — and ONE weakly-connected component**,
against paged_attention's 256 disjoint ones. Its critical path is 33 levels, so
the parallelism is there; the grouping is what fails to use it.

| group size | qwen `device_wall` | internal edges |
| ---------- | ------------------ | -------------- |
| ungrouped | **31.93 ms** | — |
| 277 (one layer) | 91.5 ms | 84.4 % |
| 64 | 75.1 ms | 32.2 % |
| 32 | 62.7 ms | ~20 % |
| 16 | 51.9 ms | ~12 % |

**Grouping loses at every size, and gets better as it does less.** Performance
improves monotonically as internal-edge capture collapses from 84 % to 12 %, which
says none of the gain comes from the controller resolving edges — only from
shrinking the unit of serialisation. Extrapolated, the best group size is 1, which
is not grouping.

The mechanism is the inverse of the one that wins on paged_attention. Per-thread
compute spread, ungrouped against grouped:

| case | ungrouped | grouped | |
| ---- | --------- | ------- | - |
| paged_attention | 5.6 % | **0.1 %** | grouping balances |
| qwen decode (G=16) | 0.5 % | **35.2 %** | grouping unbalances |

A group belongs to one thread, and a thread's controller owns 18 of the 72 cores.
When groups are interchangeable that pins work harmlessly and buys locality; when
they are chained it pins the machine to a quarter of itself. Compute issued is
identical either way (+0.03 %, 11,087 of 11,087 tasks completing), so this is
scheduling, not work.

**The discriminator is whether groups are independent of each other** — not group
size, not edge density, not how many comparators the controller has. The
comparator limit never fires on paged_attention and fires in the hundreds on qwen,
but even a controller with no limits would not fix a graph whose groups cannot run
at the same time.

### What this result does not cover

- **Nothing is numerically verified.** Every figure here is `--skip-golden`.
- **PA is the friendly case**: zero cross-group edges, so the controller resolves
  *every* edge. A group any external edge reaches still arrives task by task —
  the sink-completion counter that would make it general is not built.
- **MIX** tasks are fed as separate cube/vector entries rather than one 3-part
  package entry, which changes their placement.
- The bookkeeping core id handed to `complete_slot_task` is the thread's first
  core, not the controller's actual placement. Inert here (aSim writes no deferred
  slabs) but wrong for swimlane attribution; `queue_entry_core()` exists to fix it.
- **qwen does not run this yet** — see `KNOWN_ISSUES.md`.
