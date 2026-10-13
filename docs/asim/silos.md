# Silos: groups that see each other's completions

A *task* here is one function execution on one core -- one GroupQueue entry and one
position. A *supertask* is everything one submission launches: the runtime's task
with `block_num` blocks, each of up to three subtasks.

## The design

The GroupQueue's controllers stay what M2 made them: each is a silo that places work
on its own cores only, numbers it in its own manager's index space, and reports
completions to that manager. One thing is added: a controller sees the other
controllers' completions.

- **Forwarding.** A controller forwards every completion it records to the other
  three, which keep a copy: the same watermark and look-ahead window each keeps for
  itself. The link is `SIM_GQ_PEER_LINK_NS`, 310 ns, anchored where the
  AICPU-to-queue store is.
- **Range comparators.** A held task names each producer as `(queue, first position,
  count)`, in that queue's own numbering, and is placed once every position in the
  range has finished -- in its own queue, or in its copy of the other's. A
  producer's tasks sit on consecutive positions, so one range names all of them,
  whichever silo ran it.
- **The manager feeds in task order.** Each silo's manager takes every stamped
  supertask in task order: its block range of one wide enough to spread, all of one
  stamped to it. It names the pieces of every producer still running -- which it can
  do only once some silo has fed them -- and translates them from the shared
  task-to-position table the managers already keep.

Nothing is shared between AICPUs, no task finishes in two queues, and the limits are
M2's: 32 positions per controller, 4 comparators per task.

**The graph declares it.** `rt_declare_silos()` at the start of an orchestration
makes the host stamp every task it submits: one of at least
`TASK_SILO_SPREAD_MIN_BLOCKS` (4) blocks is spread, one block range per silo; a
narrower one follows its latest stamped producer, or takes the next silo in turn. A
task with a dispatch predicate or a sync-start cohort keeps the ordinary path: a
predicate is read when its task is ready, and a cohort needs every core at once. A
runtime without silos ignores the stamps, so the same graph runs on every arm.

## How the simulator models it

- **Publication.** A controller publishes each position's legible instant on the real
  clock as soon as it places the task, since its end is known then; a peer converts
  it to its own clock and adds the link. A controller that learns a peer's
  completion only after its own replay has passed it places the release where the
  replay stands and counts it late.
- **One replay.** Releases timed on a peer's completion are replayed together with
  the controller's own retirements, earliest first, so a release lands between the
  retirements around it as the hardware would see it.
- **Dependency sets.** Every entry of a piece waits on the same producers, so they
  share one record; in hardware each slot carries its own comparators.
- **Reads that report nothing are skipped.** A manager feeding silos reads its
  queue only when the controller has something to report. Every event carries its
  own instant, so a skipped read changes nothing the manager learns.
- **An ordinary dispatch is held, not refused.** A silo's cores also run what no
  core tracker counts, so a ready task the manager dispatches the ordinary way can
  find its ring full; it waits in the hold array instead, bounded by the position
  window like everything else.

A run that feeds no silo publishes nothing and takes none of these paths.

## deepseek_v4: silos win, by less than the oracle

The expanded orchestration with `rt_declare_silos()` as its first statement, against
the same orchestration without it. Arms interleaved per job, two reps of eight
rounds, first two dropped (n=12):

| arm | device_wall | window | vs M0 |
| --- | ----------- | ------ | ----- |
| silicon | 31.72 ms | 29.70 ms | |
| M0 | 32.29 ms | 30.86 ms | |
| M2 | 33.59 ms | 31.56 ms | +4.0 % (window +2.3 %) |
| **M2, silos** | **31.63 ms** | **29.72 ms** | **-2.1 % (window -3.7 %)** |

Every arm completes every task, and the core time issued agrees to 0.4 % (348.7,
348.0 and 347.5 ms a round). M2's and the silos' overrun ledgers are of a size, 15.3
and 18.0 ms a round, so the simulator's own cost does not separate them.

**Silos turn M2's loss into a win, and the cross-silo path does what it should.**
Each silo resolves ~4,450 producer ranges a round from its peers' publications, and
one a round arrives after its controller's replay has passed it, by 0.1 us.

**The oracle predicted more.** At the simulator's own hops -- 1.02 us inside a silo,
1.33 us across one -- [`analyzers/asim_silo_oracle.py`](analyzers/asim_silo_oracle.py)
gives -8.8 % against its M0 with the same 32-position window and predicates left with
the manager, -10.7 % with predicates free, and -12.5 % for an ideal fabric at that hop.

## Where the gap goes: a stop holds the whole feed

A silo feeds in task order, so whatever stops its head stops every later task in it.
Per silo per round, the feed stopped 806 times for want of room in the window, 704
for a producer piece another silo had not fed yet, 114 for more producer pieces
running than the four comparators hold, and 43 for a predicated producer the manager
had not finished. 286 of its ~2,400 pieces were fed after every producer had already
retired, and 82 more arrived after their named producers had finished.

The comparators are the stop that costs most. "Running" is judged per producer task,
by the managers, and a spread producer is four pieces, so any consumer of two spread
supertasks still running stops its silo's feed until a manager sees one of them
finish. With eight comparators per task, a single session of six rounds:

| arm | device_wall | window | vs M0 |
| --- | ----------- | ------ | ----- |
| M2, silos, 8 comparators | 30.49 ms | 28.42 ms | -5.6 % (window -7.9 %) |

which is within a point of the oracle's -8.8 %. The comparator stops all but
vanish and the window becomes the stop that remains; the predicated stops vanish in
that run too.

The oracle misses this because it releases each piece on its own: a piece held to
four comparators waits for the manager, but nothing behind it in its silo does.
There the window, the comparators, the feed order and up to 4 us of retirement
latency each move the result by 0.1 % or less.

## Limits

- **Two hardware numbers are design parameters, not measurements**: the peer link
  (310 ns is the AICPU-to-queue store) and the comparator count.
- **A mix block's subtasks are separate entries**, placed by type, as on M2's other
  paths; the model does not seat them in one package.
- **Predicated tasks never join a silo.** A predicate is read when its task is ready,
  so the manager dispatches or retires each one, and a silo's feed waits for it.
- **The managers run in real time against a device that stands still while the
  simulator overruns**, about 18 ms a round here as in M2's 15 ms; the window
  removes it, but a manager's reaction is cheaper against the device than it would be.
- **qwen and paged_attention have not been run with silos.** Both arms are unchanged
  without them: paged_attention M2 -48.2 % against M0, qwen's expanded form +0.8 %.

## Reproducing it

The silo arm is the expanded orchestration with `rt_declare_silos()` as the entry's
first statement; everything else is as in [validation/deepseek-v4.md](validation/deepseek-v4.md#reproducing-it).
`[GQ_SILO thread=N]` in the device log carries the feed's stops and stall time, the
cross-silo releases and the lateness counts above.

```bash
python docs/asim/analyzers/asim_silo_oracle.py <deps.json> <chip_swimlane_records.json> <calib>
# H_IN=1.02 H_N=1.33 PRED=2.68 for the simulator's own hops; W, ORDER, C,
# RETIRE_US, ENTER_US, SPLIT_MIN and EDGE as the script's docstring describes
```
