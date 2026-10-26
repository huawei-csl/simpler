# What a qwen round spends outside the dispatch window, and what a GroupQueue can take back

## Question

On qwen the scheduling window is closed: with every design constraint removed —
unlimited comparators, free rings, free entry, no residency gate, no look-ahead
limit, a perfect list placement — the
[oracle](../asim/analyzers/asim_local_oracle.py) reads 7.068 ms against M0's
measured 7.040 ms window, and the cube cores are busy 6.50 ms of that 7.04 ms
(92 %). Every arm executes identical core work (66.21 / 66.22 / 66.25 ms summed).
So a GroupQueue cannot beat M0 by scheduling better, and the only place a win can
come from is the part of the round that is **not** the dispatch window.

That part was unattributed: `chip.run` subdivided only into `bind` /
`runner_run` / `device_wall` / `validate`, because the AICPU device-phase markers
were stamped by `tensormap_and_ringbuffer` and by neither host-orchestrated
runtime. A two-point fit against task count put it at roughly 0.81 ms fixed +
34 ns a task for M0 and 1.02 ms + 40 ns for the partitions, which says how big it
is and nothing about what it is.

## The markers

`host_build_graph` and `group_queue` now stamp the same `AicpuPhase` buffer the
device-orchestrated runtime does. Three phases were appended for the shape a
host-orchestrated runtime has — `Classify` (every thread seeds its slice of the
ready set over the attached graph), `GroupSeed` (group_queue only: seed the ready
group queue) and `Retire` (give the cores back) — and the phases a device
orchestrator owns (`SoLoad`, `ConfigValidate`, `OrchWindow`) stay unset. See
[device-phases.md](../dfx/device-phases.md).

One trap is worth recording. A span's `dur` has the simulated device's own
overrun removed; its `ts` is a raw sys-counter offset. On an arm whose model
overruns they are on different clocks, so **any ts-derived interval is wrong** by
the overrun accrued before it — on the partition arm the `retire` span reads a
start of 14.3 ms inside an 8.6 ms round. Read durations only, and take what the
phases do not cover as `graph_build` minus its children.
[analyzers/asim_phases.py](../asim/analyzers/asim_phases.py) does both; the
recipe is in [running.md](../asim/running.md#splitting-device_wall-into-phases).

## Where the round goes

qwen3-14B decode, 8 layers, 2,238 tasks. Medians over 18 rounds an arm (three
paired reps, arms interleaved in one lock, first two rounds of each run dropped);
per-rep totals spread ±25 µs. `other` is the residual of `graph_build` after its
child phases — the per-thread `device_log` writes between them and the last
thread's finalize, less whatever the children's cross-thread union spans
double-count where they overlap. It is a remainder, not a measurement of any one
thing, and the arms' values are not comparable with each other.

| µs a round | M0 | GroupQueue, no partitions | GroupQueue partitions |
| ---------- | -- | ------------------------- | --------------------- |
| preamble (executor init: 72-core handshake) | 221.7 | 339.6 | 314.9 |
| arena_wire | 0.4 | 0.4 | 0.4 |
| sm_reset | 18.5 | 18.5 | 21.6 |
| group_seed | — | 121.9 | 137.5 |
| classify | 65.4 | 117.0 | 28.7 |
| **dispatch (sched)** | **7183.4** | **7353.8** | **7690.2** |
| retire | 222.0 | 224.9 | 310.1 |
| post_orch (teardown) | 38.0 | 40.0 | 39.0 |
| other (device_log + finalize) | 179.1 | 144.7 | 62.7 |
| **total** | **7934.3** | **8326.9** | **8637.8** |
| (independent `device_wall`) | 7947.9 | 8357.2 | 8662.9 |

The totals reproduce the arms' previously measured device walls (7.927 / 8.329 /
8.601 ms), so the markers cost nothing and the attribution is of the same round
the A/B measured.

**M0 spends 765 µs — 9.6 % of its round — outside the dispatch window.** Its own
model barely overruns, so for this arm alone the spans' `ts` is usable, and it
splits that into **304 µs before the window opens and 459 µs after it closes**
(304 + 7183 + 459 = 7947, its measured `device_wall`). The earlier guess that
this is dominated by the per-round core handshake is wrong: the handshake is
222 µs of the 765, and the tail is the larger half.

## What the partitions actually pay

| component | Δ vs M0 | what it is |
| --------- | ------- | ---------- |
| preamble | +93 | **not hardware** — see below |
| group_seed | +138 | seeding the ready group queue, a whole-graph walk |
| sm_reset | +3 | |
| classify | **−37** | the plan pre-resolves the ready set, so there is less to derive |
| dispatch | +507 | the window loss, i.e. the four-named-producer rule |
| retire | +88 | |
| post_orch | +1 | |
| other | −116 | the `graph_build` residual; see the note above |
| **total** | **+703** | |

The preamble delta is **the simulator, not the design**. `AicpuExecutor::init()`
is byte-identical between the two runtimes apart from the `asim_bringup()` call,
and `handshake_partition` / `pre_handshake_init` / `post_handshake_init` are
identical too, so all of it is inside the simulated-device bring-up —
`asimgq::init()` value-initialises eight `SimQueue`s, a few hundred KB of AICPU
memset a round, with no hardware counterpart. Logged in `KNOWN_ISSUES.md`.

So the GroupQueue's own excess outside the window is **group_seed (138 µs) plus
retire (88 µs), against a classify that is 37 µs cheaper** — about 190 µs net, a
quarter of the 703 µs gap. The rest is the dispatch window, which the oracle has
already priced.

## What this says about beating M0

Removing a cost both arms pay is a losing move here: hoist the handshake out of
the round and M0 drops 222 µs too, the absolute gap is unchanged, and the
relative loss gets worse on a smaller base. **A win requires overlapping the
prologue with compute, which a GroupQueue can do and a whole-graph fanin walk
cannot.**

The un-partitioned arm looked like the cheaper target: its prologue was 598 µs of
an 8327 µs round against only 170 µs of window loss, so hiding the prologue
behind dispatch read as a 2.6 % win. **Built and measured, it is not** — see
below. Only one of the prologue's four parts can hide behind dispatch; the others
are prerequisites for queueing a task at all, and they had to be removed instead.

## Built: the prologue off the critical path, un-partitioned arm

Two changes, both in the group_queue runtime, measured paired against M0 in one
lock (3 reps × 6 steady rounds each, medians of `device_wall`):

| arm | vs M0 | per rep |
| --- | ----- | ------- |
| as built | +5.01 % | +5.37, +5.09, +4.57 |
| + classify seeded from the dispatch loop, undeclared-group walk skipped | +4.07 % | +3.87, +3.86, +4.48 |
| + group-table clear skipped | **+3.04 %** | +3.11, +2.97, +3.05 |

`completed_tasks=2238 total=2238` on every round of every rep, and the
partitioned arm on the same build completes too (8419 µs against its 8663, the
same prologue saving).

That completion count is the whole correctness gate available here, and it is
worth being explicit about why. `group_queue` builds only for `a2a3asimgq`,
whose AICore is a model that consumes a calibrated duration and writes nothing,
so a numeric golden cannot pass on any arm of this campaign — run with the
golden on, the **unmodified M0 arm** fails it at `max_diff=4.94`. Every A/B
above therefore ran `skip_golden=True`, and what the arms are checked against is
that each one completes its whole graph, issues the same compute, and reaches
the same task count. A numeric check of the classify change needs a runtime that
computes, which `group_queue` does not yet have.

**What the two changes are.**

*The classify runs beside the cores, not ahead of them.* `classify_partition()`
became `classify_begin()` (take this thread's slice) plus `classify_chunk()`
(seed 32 of it), called from the dispatch loop, and the barrier that held every
thread until all slices were walked is gone. The barrier's stated job — no thread
dispatches against a half-seeded graph — turns out to be already covered:
`register_wake()` is a CAS loop that, finding a producer's list drained,
re-classifies the consumer against every fanin and either routes it or re-targets
the next unmet producer. A producer completing before its consumer registers is
the ordinary run-time case, not a boot-only hazard. What the loop does need is a
guard against concluding the run is over while a slice is unseeded, which a
`classify_pending_` count supplies.

*Two whole-table walks a run that declares no group cannot use.* The host already
counts the groups it kept, so it now ships that count in the task header;
`index_declaration()` returns on a zero instead of reading all 2,238 descriptors
to discover there is no group, and `reset_group_owners()` clears its 88 KB of
owner / opened / external / delivered tables only when a run actually wrote them.

The second of those cost a day and is worth recording, because the trap is
general. Skipping a clear is only sound when "not dirty" really means "already
in the cleared state", and `g_group_owner`'s cleared state is `NO_OWNER`, which
is `-1` — not the zero its static storage starts at. With the flag starting
false, the first run of a process skipped the one reset that had been doing the
initialising, every group read as owned by thread 0, `try_claim_group`'s
compare-and-exchange against `NO_OWNER` failed for all 256 of them, and
paged_attention stalled with 16,384 tasks ready, no group ever opened and
`deps_named=0`. qwen could not see it and neither could deepseek_v4: both
declare no group, so the whole path is dead for them, and the counters say so
(`[GQ_GROUP] groups=0 declared`). The flag starts **true** — the tables do not
begin cleared, and the first reset is what establishes the invariant the skip
then relies on.

The qwen figures below were taken before that fix, and they stand: the path it
repairs is dead on a graph that declares no group. A rep re-run on the fixed
build reproduces every phase the change touches — prologue 34.0 µs against
34.7, classify 0.4 against 0.5, group_seed 14.4 against 14.0, dispatch 7,382
against 7,384. Its `device_wall` ratio reads +3.35 % rather than +3.04 %, all
of the difference in `retire` and `preamble` on a single rep, which is the
noise the last section warns about rather than a cost of the fix.

**What it bought, by phase** (µs a round, medians over 18 rounds):

| phase | M0 | as built | overlapped |
| ----- | -- | -------- | ---------- |
| preamble | 211.9 | 285.4 | 336.0 |
| arena_wire + sm_reset | 18.4 | 19.9 | 20.1 |
| group_seed | — | 101.8 | **14.0** |
| classify | 64.7 | 114.7 | **0.5** |
| **prologue (inside graph_build)** | **83.2** | **236.3** | **34.7** |
| dispatch | 7182.2 | 7355.0 | 7383.9 |
| retire | 195.9 | 274.9 | 218.7 |
| `device_wall` | 7930.6 | 8334.3 | **8171.8** |

The overlap itself works as designed: 115 µs of classify left the prologue and
dispatch grew 29 µs, so **three quarters of it hid behind the cores** — which is
what a round at 92 % cube occupancy should do with AICPU work. And the prologue
is now 34.7 µs against M0's 83.2: **cheaper than the baseline's.**

## Why that is still not a win

With the prologue below M0's, the arm is +241 µs. What is left is not prologue:

| component | Δ vs M0 |
| --------- | ------- |
| dispatch window | +202 |
| preamble | +124 |
| retire | +23 |
| prologue | **−49** |
| other (`graph_build` residual) | −80 |

The +202 µs is the GroupQueue transport inside the window, and the unconstrained
oracle has already said that a free GroupQueue ties M0 there rather than beating
it. The +124 µs preamble is the simulator: `AicpuExecutor::init()` is
byte-identical between the runtimes apart from `asim_bringup()`, and
`asimgq::init()` zeroes eight `SimQueue`s a round. Discounting it leaves roughly
+1.5 %.

So the correction to the sizing above is this: **the prologue was not 598 µs of
overlappable work.** It was 115 µs of overlappable walk, ~190 µs of removable
per-round table work, a ~120 µs simulator artefact, and a ~210 µs core handshake
that M0 pays too. Overlapping and removing all of the first two gets the arm from
+5.0 % to +3.0 %, and the remainder is a transport cost the window cannot give
back.

## Still on the table

- **The 72-core handshake, ~210 µs in both arms.** It is the largest single item
  left outside either arm's window and it is per-round, though the cores it arms
  do not change between rounds. Hoisting it to once per bind helps M0 by the same
  amount, so it closes no gap — it just makes both rounds ~2.6 % shorter.
- **`asimgq::init()`'s per-round queue zeroing**, logged in `KNOWN_ISSUES.md`:
  ~120 µs off every group_queue measurement, with no hardware counterpart.
- `preamble` is the noisiest phase by far — 243 to 344 µs across reps on a
  byte-identical binary. Any claim about it needs more than three reps.
- **The +202 µs of in-window transport is a lower bound, and it is unprofiled.**
  Core work is identical (GQ `aic` 39,017 + `aiv` 27,246 = 66,263 µs against
  M0's 66,104, 0.24 % apart), so the whole delta is cube idle: the hottest cube
  core is busy 6,503 µs either way, which leaves it idle 679 µs inside M0's
  7,182 µs window and 881 µs inside the GroupQueue's 7,384. The bound is a lower
  one because the overrun correction is asymmetric — GQ's `poll_overrun_us` is
  2,411 µs a thread a round against M0's 4, so the phase subtracts ~1.9 ms on
  one side and nothing on the other. **Which edges carry that idle is not
  measured**; the counters cannot say, and it needs the chip swimlane
  (`SWIM=3` or `4` on the run scripts) replayed against a dep-gen round.
- **Whether the modelled latencies sum the same.** Both models read the same
  four calibrated latencies and share `ARRIVAL_NS` / `CORE_POLL_NS`, and GQ's
  modelled polling is the *cheaper* of the two (471 µs a thread a round against
  M0's 1,773, because it polls 2.4× as often at 10 ns against 92). Its genuine
  additions are `report` at 150 ns per completion a manager must see — ~146 µs
  at 971 positions a manager a round — and `ring` at 5 ns; `peer_link` is off on
  the plain arm, which declares neither silos nor partitions. Setting
  `set_queue_latencies_ns(0, 0)` and `SIM_GQ_RING_CYCLE_NS = 0` would price how
  much of the 202 µs those two hops are, in one build and one paired A/B.

## When to reconsider

- If the `device_log` writes between phases are turned off, every arm's round
  shrinks by 60-180 µs and the relative gaps widen slightly. The absolute prize
  for overlapping the prologue does not move.
- `group_seed` and `classify` walk the graph, so they scale with task count;
  `preamble`, `arena_wire`, `sm_reset` and `post_orch` do not, and `retire`
  scales with cores. At 40 layers the walks are ~5× these figures and the fixed
  part is unchanged, so the prologue is a larger absolute target and a smaller
  share of the round.

## References

- [2026-10-gq-partitions-qwen-why.md](2026-10-gq-partitions-qwen-why.md) — why
  the window loss is the four-named-producer rule, and why the plan is not where
  to fix it
- [docs/dfx/device-phases.md](../dfx/device-phases.md) — the phase set and how
  the host reduces it
- [docs/asim/validation/partitions.md](../asim/validation/partitions.md) — the
  three cases' measurements
- [docs/asim/running.md](../asim/running.md#splitting-device_wall-into-phases) —
  how to reproduce the phase table
