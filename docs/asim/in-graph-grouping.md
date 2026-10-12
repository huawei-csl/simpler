# In-graph grouping: letting a Definition declare a group

A *group* is a set of tasks whose internal edges the GroupQueue controller
resolves on its own cores: the manager submits a member naming its in-group
producers, and the controller holds the entry until those positions retire. The
contract is in
[`task_grouping.h`](../../src/a2a3/runtime/group_queue/runtime/scheduler/task_grouping.h).
It is the mechanism the whole design rests on -- on paged_attention, where 75 % of
tasks are released by the controller rather than the manager, it is worth -48 %.

**It cannot be reached from a Graph.** `OrchestratorState::begin_group` /
`end_group` back-fill a group over a range of **host-side task ids**
(`task_allocator.active_count()`). A Graph-expanded layer's tasks are created on
device by the Scheduler, so they never pass through that range, and
`deps_named` is 0 on every graph-execution case we run -- deepseek_v4
([validation/deepseek-v4.md](validation/deepseek-v4.md), where M2 loses 3.3-4.8 %)
and qwen's Graph orchestration
([validation.md](validation.md#a-second-qwen-shape-the-graph-orchestration), where
it is level).

A Definition is already the shape a group wants: bounded, internally connected,
recorded once and replayed. This note is what it would take to say so.

## The five pieces

| # | Piece | Where |
| - | ----- | ----- |
| 1 | **Declare during recording.** While a Definition records, `begin_group` / `end_group` must back-fill over the recording's task range rather than the global allocator's. One group id per recorded task, defaulting to "its own group". | `RecordedInGraphTask` + `begin_group`/`end_group`, `host/orchestrator.cpp` |
| 2 | **Carry it in the Definition.** `InGraphTaskDefinition` has a spare `reserved` byte -- enough for 255 groups in a body of ~80 tasks. `graph_fill_definition` copies it beside `ed_flags`, which is decided the same way: once per recorded shape, because a body's structure is fixed by its Definition. | `graph_execution.h`, `graph_fill_definition` |
| 3 | **Per-execution positions.** `task_position()` keys a flat `g_task_position[kPositionSlots]` by global local id, so in-graph ids from two executions collide. `GraphExecution` already carries per-execution arrays in its storage tail (`task_states`, the argument pools); a `task_positions` array indexed by in-graph local id belongs with them. | `graph_execution.h`, `GraphExecutionStorageLayout` |
| 4 | **An in-graph `group_deps`.** The existing one walks the global fanin and tests `tasks.is_completed(prod)`. The in-graph variant walks `execution.fanin_offsets` / `fanin_indices` and tests `execution.task_states`, then maps each unmet same-group producer through (3). The comparator cap stays `SIM_HELD_MAX_DEPS` = 4. | `SchedulerState::group_deps`, `scheduler.h` |
| 5 | **Group ownership.** A group is resolved by one controller, so one manager must own its members. `claim_group` keys on a group id today; an in-graph claim has to key on **(execution, group)**, because two run slots can hold executions of the same Definition at once. | `task_grouping.h`, `scheduler.h` |

Pieces 1 to 4 have landed. Piece 5 is the one that turns them on, and it has a
prerequisite the others did not.

## What piece 5 has to decide first

A grouped member is routed to its group owner's queue, `ready_queues_gq[owner]`,
and those per-thread spans exist only after `partition_ready_queues_by_owner()`
has cut them -- which returns early unless `gq_group::active()`, i.e. unless the
**global** declaration scan found a group. A body's declaration never reaches
that scan, so in-graph grouping cannot route anywhere until one of these is done:

- **Let a Definition's declaration make grouping active**, and partition on that
  too. The claim then lives in a per-execution owner array, which is the right
  home anyway: a group id indexes its own Definition, and `g_group_first` /
  `g_group_extent` describe contiguous runs of **global** ids that an in-graph
  task does not have. This is the shape to build.
- **Map each in-graph group into the global group id space**, so the existing
  route, claim and partition all work untouched. Cheaper to write, but it gives a
  per-execution thing a global name and has to invent extents for a run of ids
  that does not exist.

Either way piece 5 changes how a grouped run is scheduled, where pieces 1 to 4
are inert, so it wants a session of its own.

## The constraint that decides the grain

**A group is confined to one queue**, which serves 18 of the 72 cores -- and
declaring any group confines everything else too. Grouping makes every Scheduler
thread pop only its own span of the ready queues, and routes a task outside every
group to the thread that released it, so only a claimed group ever moves work
between threads.

On deepseek_v4 that is not harmless. Measured on the expanded orchestration, one
3-task group costs **+105 %** against M0: the critical chain and its 16-64-block
tasks end up on one thread's 18 cores, and a round's ~227 ms of cube work is a
quarter of all 24 cube cores' time
([validation/deepseek-v4.md](validation/deepseek-v4.md#why-no-group-helps)).
In-graph grouping would turn on the same partition, so on this case it cannot pay
unless the Definition also splits its wide tasks into per-queue pieces.

So the first grain to try is the **MoE expert tile** -- six tasks, all consuming
one `dispatch_gather` output -- only on a case whose wide tasks already fit one
queue. A layer-sized group runs the whole layer on a quarter of the device.

## What would say it worked

`deps_named` > 0 is the precondition, not the result: it only says the contract
was expressible. The result is M2's window against M0's.

The ceiling on deepseek_v4 is the critical path's producer-to-consumer gaps,
8.9 ms of a ~31 ms silicon round, and the partition already costs more than that.
