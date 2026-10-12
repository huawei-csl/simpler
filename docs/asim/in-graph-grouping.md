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

## The constraint that decides the grain

**A group is confined to one queue**, which serves 18 of the 72 cores. Declaring a
whole layer as one group would therefore run that layer on a quarter of the
device. On deepseek_v4 that is probably harmless -- its cube cores are 3.3 % busy
and the wall is a dependency chain, so there is no throughput to lose -- but it is
a real risk on any case that is not.

So the first grain to try is the **MoE expert tile**: six tasks, all consuming one
`dispatch_gather` output, which is how the case's README already describes them.
That is close to paged_attention's pattern, and it lets several tiles run on
different queues concurrently. A layer-sized group is the experiment to run
second, against the tile-sized one, not the one to build first.

## What would say it worked

`deps_named` > 0 is the precondition, not the result: it only says the contract
was expressible. The result is M2's window against M0's on deepseek_v4, which is
currently +2.7 % to +3.4 %. Grouping removes the AICPU from a hop, so the figure
to watch is the per-hop cost across the ~1,300 serial hops a round -- about 0.8 us
each today.

Two existing facts bound the expectation. Grouping is worth -48 % where it takes
75 % of hops off the manager's path and the cores are starved. Here the cores are
idle for a different reason -- the chain -- so the ceiling is the hop cost itself,
not the idle time.
