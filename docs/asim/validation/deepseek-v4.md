# deepseek_v4_flash_decode: no submission shape lets the GroupQueue win

Two dies, 43 layers, 368 kernels. A round submits ~16,000 tasks per rank, of
which 3,700-5,200 reach a core as 29,000-32,000 block dispatches. It is the one
case where M2 loses to M0, and this page records that loss, every submission-side
adaptation tried against it, and why none of them turns it.

## As shipped: the Graph form

Three arms interleaved in one lock, two independent sessions:

| session | silicon | M0 | M2 | M2 vs M0 | window |
| ------- | ------- | -- | -- | -------- | ------ |
| 2026-10-09, n=12 | 36.187 ms | 36.638 ms | 38.384 ms | **+4.8 %** | +3.4 % |
| 2026-10-12, n=18 | 35.740 ms | 36.710 ms | 37.911 ms | **+3.3 %** | +2.7 % |

Compute issued agrees between the arms to 0.5 %, every round completes every
task, and every duration comes from the calibration. M0 reads +1.2 % and +2.7 %
against silicon.

**The case is critical-path bound.** A round carries ~227 ms of cube and ~122 ms
of vector core time in the model -- cube cores ~26 % busy, vector cores ~7 %
(silicon's rank 0: 43 % and 5 %) -- so the cores are far from saturated, and the
wall is a chain: 43 layers at ~842 us each. paged_attention's -48 % came from
feeding idle cores faster where independent work was abundant; here the work
that sets the wall is serial.

**And the mechanism that would shorten a chain never engages: `deps_named=0`.**
The controller resolves no dependency, so the GroupQueue adds its indirection to
every hop without taking the AICPU out of any -- about 0.7 us across the ~1,670
serial hops of a round's critical path, which is the measured delta. M2 also polls 5.3x as
often as M0 (28.8k against 5.5k per thread per round) and carries a 6.1 ms
ledger against 0.32 ms.

**Grouping is structurally unreachable here, not merely undeclared.**
`OrchestratorState::begin_group` / `end_group` back-fill a group over a range of
**host-side task ids** (`task_allocator.active_count()`). This case's
`host_build_graph` orchestration is the Graph form -- one GRAPH task per layer,
expanded by the device Scheduler -- so its in-graph tasks are created on device
and never pass through that range. The same holds for qwen's Graph orchestration
above. **No graph-execution case can declare a group today**, which is why the
GroupQueue's central mechanism is dormant on both.

## The expanded form, and an early-dispatch artefact

`kernels/orchestration/decode_fwd_expanded.cpp` in the `host_build_graph`
example calls each Graph body where the Graph was submitted -- the path
`rt_submit_graph` takes itself for non-cacheable boundary args -- so every task
passes through the host and grouping becomes declarable. It is the same program:
it runs to completion on silicon on both ranks with the real fixture uploaded (the
case has no golden), and the same kernels reach cores the same number of times as
in the Graph form (5,175 tasks on rank 0 in both).

Arms interleaved per job, two reps of eight rounds, first two dropped (n=12):

| arm | early dispatch on (as shipped) | early dispatch off |
| --- | ------------------------------ | ------------------ |
| silicon | 32.19 ms | 32.86 ms |
| M0 | 26.27 ms | 33.36 ms (+1.5 % vs silicon) |
| M2 | 24.09 ms | 34.51 ms |
| **M2 vs M0** | **-8.3 %** (window -10.9 %) | **+3.5 %** (window +1.9 %) |

**The -8.3 % is an artefact of the simulator, not a GroupQueue win.** Both arms
start an early-dispatched consumer when it is published rather than when its
doorbell rings, so a pre-staged task can compute before its producers finish:
M0 in `asim_push`, M2 because `try_early_dispatch` submits the pre-stage to the
controller ungated. Early dispatch is worth ~2 % on silicon here, and 21 % in M0
and 30 % in M2. Turning it off in the orchestration -- every
`set_allow_early_resolve(true)`, 148 sites, made `false` -- brings M0 to +1.5 % of
silicon and M2 back to +3.5 %, the Graph form's verdict. Graph-body tasks never
carry the early-resolve bit, which is why the Graph form tracked silicon and the
expanded form, with early dispatch on, read M0 20 % fast.

Two further checks rule out the alternatives. Transport is not it: zeroing M0's
send latency is worth -3.1 %, and zeroing every modelled latency -2.9 %, against
M2's -8.3 %. And the calibration only changes the size: the v3 table's means are
tail-inflated (a few slow instances in one cold round), and a table of the
expanded run's per-kernel medians gives M2 -4.2 % with early dispatch on, the same
artefact.

## Why no group helps

**Declaring any group partitions the whole run.** It makes `gq_group::active()`
true, and each Scheduler thread then pops only its own span of the ready queues
(`partition_ready_queues_by_owner`, `ready_queues_for`). A task outside every
group is routed to the thread that released it -- `claim_group` returns the
caller for an ungrouped task -- and a wide task's unclaimed blocks are re-pushed
to that same span, so neither ever reaches another thread's cores. Only a group
moves work between threads, by being claimed.

On this case that pins the critical chain, and every wide task on it, to one
thread's 18 cores. **One 3-task group resolving 4 edges -- a probe that changes
nothing else -- takes M2 from 24.1 ms to 54.0 ms, +105 % against M0** (n=12,
early dispatch on, same lock as the table above).

The silicon critical path says why no grouping can earn that back. On the
expanded form's rank 0 it is 1,670 tasks: 18.7 ms of compute kernels, 3.5 ms of
comm kernels, and **8.9 ms of producer-end to consumer-start gaps** (median
3.7 us) -- the most grouping could ever recover. 330 of those tasks are 16-64
blocks wide, and confined to one queue they run in up to six times the waves:
~17 ms by wave count alone, +29.9 ms measured (24.1 -> 54.0 ms). The cube work
alone does not fit: a round's ~227 ms of it is a quarter of all 24 cube cores'
time, so one queue's six would have to be busy for the whole round.

So the only grouping that could pay here keeps wide tasks off a single queue:
split each into per-queue pieces declared in separate groups, so the four threads
claim them in parallel, and capture only the tile-local stage transitions inside
a group. The kernels take their tile from the launch's block index, which has no
offset, so each split is per-kernel view surgery. It is not attempted, and most
of this network's stage transitions are all-to-all, which no such split keeps
inside a group.

## Caveats

**Early dispatch must be off for any M0-vs-M2 delta on this case.** Both
simulators start a pre-staged consumer at publish, not at its doorbell, and the
two arms profit unequally; with early dispatch on, the delta measures that
defect -- `asim_push` in M0, `submit_to_queue` handing the pre-stage over ungated
in M2. Until it is fixed, the early dispatch silicon uses here cannot be modelled
at all.

**Two caveats on the absolute numbers, neither of which explains a delta**
(both arms share them). The per-rank asymmetry silicon shows -- 32,273 against
28,871 block dispatches a round, 5,175 against 3,717 tasks -- is lost, because
the 26 predicated MoE tile tasks key on `recv_count_out`, which `dispatch_meta` writes on device and the model
never computes; both simulated ranks then do identical work. And a comm kernel's
cross-die wait is baked into its calibrated duration, so it is charged but does
not respond to rank drift -- which is exactly what a scheduler change produces.
Until both are addressed this case is a **scheduler benchmark, not a fidelity
reference**.

## Reproducing it

Method and the discipline the numbers depend on are in
[../running.md](../running.md); the simulated baseline's own error against
silicon is in [fidelity.md](fidelity.md). Four things are specific to this case:

- **Two dies.** `task-submit --device auto --device-num 2`, and the run passes
  both of `$TASK_PHYS_DEVICE` to the driver.
- **The driver needs one line.** Its `_RUNTIME_ENV` table has entries for
  `tensormap_and_ringbuffer` and `host_build_graph` but not `group_queue`, which
  is `host_build_graph`-derived and takes the same (empty) ring sizing. Without
  it the run raises `KeyError: 'group_queue'`.
- **The expanded form.** Pass
  `examples/a2a3/host_build_graph/deepseek_v4_flash_decode/kernels/orchestration/decode_fwd_expanded.cpp`
  as the driver's `orchestration_source`. Early dispatch off is that file with
  `set_allow_early_resolve(true)` replaced by `(false)`; the probe group is
  `rt_group_begin()` before its 6th kernel submission and `rt_group_end()` before
  its 8th -- a run with no internal edge is dropped by the orchestrator, so a
  one-task probe registers nothing.
- **One calibration covers both ranks.** `deps.json` cannot name the func_id of a
  task the device expands, so it comes from the swimlane's recorded per-task
  kernel ids (`asim_gen_calib.py` prefers them; pass `-` for the dep-gen round).
  The two ranks' records are merged into one table, because the per-rank split is
  which kernels fired rather than how long they take.
