# Partitions: the controllers run the graph a partition at a time

[silos.md](silos.md) feeds each silo one supertask at a time, in task order, so
whatever stops the head of a silo's feed stops everything behind it. Here the graph
is split into *partitions*, each partition into one *subpartition* per group, and
each controller runs its subpartition from resident entries: every dependency inside
the partition is resolved by the controllers between themselves, without the
managers.

The terms: a *task* is one function execution on one core, and a *supertask*
everything one submission launches. A *partition* is a subgraph; the partitions are
disjoint, cover the graph, and are numbered `0 .. P-1`. A *partition-local index*
numbers the tasks of one partition and is used only to track their completion. A
*group* is one AICPU, its controller and its cores -- what [silos.md](silos.md) calls
a silo.

## The design

- **Partitions are declared up front.** `rt_declare_partitions(target)` as an
  orchestration's first statement declares silos and asks for partitions of about
  `target` supertasks. Once the graph is built the host -- standing in for the
  compiler, which would do this ahead of time -- moves each boundary, within a
  quarter partition of where it would fall, to the point the fewest dependencies
  cross. Here a partition is one layer: 372 supertasks.
- **Each partition is numbered by dependency level.** Every task of a partition
  takes an index -- one per block and active subtask of its supertask, the blocks
  consecutive -- in the order of its supertask's dependency level, so a task's
  producers in the partition sit below it. A dependency-only task takes none, and
  levels pass through it. An index is 11 bits: a partition spans at most 2,048.
- **Subpartitions are fixed.** A supertask of 8 or more blocks is split across the
  four groups, each taking a quarter of its block range; a narrower one runs whole
  in the group of its latest producer, or the next group in turn. Group `g`'s share
  of a partition is its subpartition, and reaches its controller through its own
  manager.
- **What a task waits on.** The tasks of one supertask's piece share one record:
  up to four named producer supertasks -- each one index range, whichever groups
  run it -- and a threshold of their own. The four named are the producers that end
  last; the threshold is where the fifth-latest ends, and the partition's watermark
  must reach it. So a task with any number of producers needs nothing from its
  manager. A task also waits until the watermark is within the 512-index look-ahead
  of it, so its finish has a bit to land in.
- **The controllers share one watermark.** Each controller tracks the partition's
  watermark -- every index below it finished -- and a 512-bit look-ahead past it,
  from its own finishes and the others' broadcasts. It broadcasts its state, tagged
  with the partition, when it changes, at most once per 100 ns, and the others have
  it 300 ns later; a receiver merges by maximum and OR. A producer is done for a
  consumer once its whole index range is known finished.
- **A controller moves on by itself.** It reaches partition `k + 1` once its
  watermark covers `k` whole; a broadcast tagged with a later partition means every
  earlier one has finished. The next subpartition is resident by then and is gated
  until the switch: the manager appends it, in index order, as the store frees.
- **Predicates are answered ahead.** The manager evaluates each dispatch predicate
  once the task writing its operand has finished, and the controller holds the
  answer until the task is ready -- [below](#predicates-in-hardware).
- **What stays with the manager.** A sync-start cohort keeps the ordinary path, as
  with silos; deepseek_v4 has none. A supertask with a producer the controllers do
  not run is held back and fed once that producer has finished.

Against M2 this adds, per controller: a store of 512 entries (~20 bytes each), where
the largest subpartition here is 318 tasks; a threshold and a look-ahead check per
record beside M2's four comparators, which match index ranges; a watermark and
512-bit look-ahead per partition in flight, the current one and the next; the
broadcast of the partition, the watermark and the 512 bits, ~70 bytes; and two bits
per record for predicates, with a write port for the answers and a decision queue.

## Predicates in hardware

A dispatch predicate lets a task skip itself: deepseek_v4's routed-expert tasks run
only when `recv_count[e] > 0`, the rows dispatched to expert `e`. That is six
supertasks per expert, 192 a layer, and a task that skips still has to finish in
order, because its consumers wait on it. Three parties share the work.

- **The host names the writer.** At submit it records the last task to write the
  element a predicate reads (`TaskDescriptor::predicate_writer`, from the tensormap
  it already keeps). The predicate's value is final once that task has finished --
  here `dispatch_meta`, one per layer, which finishes before the expert tasks' input
  rows have arrived.
- **The manager answers when the writer finishes.** It posts the piece marked as
  predicated, waits for the writer's completion -- which it learns from the
  completions the managers retire anyway -- reads the element, and writes run or
  skip to its controller: one posted store naming the piece.
- **The controller holds the answer.** Two bits per record, *pending* and *skip*. The
  answer is one more condition: a predicated piece is ready once its producers,
  threshold and look-ahead allow and its answer has landed. A skip then finishes the
  piece's tasks at once, without a core, and the controller broadcasts them like any
  finish; a run dispatches them. A piece that is otherwise ready while its answer is
  still pending goes to the decision queue, which the manager polls and answers --
  the path for a predicate whose writer the host cannot name, in a manual scope. On
  deepseek_v4 it goes unused, and it could go altogether: a manager that waited for
  all of such a task's producers instead of its writer would never need asking.

A skip waits for the piece to be ready, not just for the answer: a skipped task
finishing before its own producers would let its consumers start before them, out
of order with any write after their reads. What the early answer saves is the round
trip through the manager once the piece is ready.

Two variants cost more hardware. A predicate table in the controller would take one
answer per distinct predicate -- 32 a layer here, one per expert -- instead of one
per piece, at the price of a table and a field per record. A controller that read
the element itself would need a device-memory read port, a comparator and ~18 bytes
of predicate per record, and would take the manager off the path; on deepseek_v4
the early answer already does.

## How the simulator models it

- **The store.** Each queue keeps its entries resident in records, a record's named
  producers keyed by supertask. A producer is done for a record once every queue
  that runs a piece of it has reported that piece, which a queue learns at the same
  instant as the indices the piece covers. The look-ahead holds a record by its last
  entry: its threshold rises to that entry's index less 512.
- **The exchange.** A queue posts each piece it finishes to every other queue, in the
  first slot of a 100 ns grid from the finish, on a single-producer ring stamped on
  the real clock. The receiver converts the stamp to its own clock, adds the 300 ns,
  and learns the piece's indices and its supertask together.
- **Releases are replayed.** A queue's model advances when its manager touches it.
  A release due inside the interval being replayed is timed at its own instant; one
  learned only after the replay has passed it is placed where the replay stands and
  counted late.
- **The manager.** It builds its share of the next partition 16 supertasks per pass
  of its dispatch loop while the current one runs, and posts the share a piece at a
  time as the store frees -- the next partition's once its controller has reached
  the one before. It answers a predicate as soon as it sees the writer completed:
  each answer is one store to its queue, landing as a submit does.

## deepseek_v4: the partition boundary is gone, and so are the predicate stops

**What the oracle expects.** [`analyzers/asim_silo_oracle.py`](analyzers/asim_silo_oracle.py)
at the simulator's hops, one layer a partition cut at narrow points, supertasks
split from 8 blocks, and the switch costing a cross-group hop (`LAUNCH_US=1.32`),
against 30.66 ms today. The predicate column is what the path through a predicated
task costs above the controllers' own hops:

| predicate path | makespan | vs today |
| -------------- | -------- | -------- |
| free: answered before the task is ready | 27.32 ms | -10.9 % |
| 1 us | 27.50 ms | -10.3 % |
| a manager hop, 2.68 us | 27.87 ms | -9.1 % |

Four named producers are enough: 176 of 8,829 pieces have more, and the makespan is
27.32 ms whether every producer is named or those pieces wait for a manager instead.
The oracle gives a skipped task no index, so it cannot see the look-ahead; the
simulator can.

**Measured.** The expanded orchestration with `rt_declare_partitions(372)` as its
first statement, interleaved with M0 per job, eight rounds a rep with the first two
dropped. Each row is against the M0 of its own session, and the window is the
scheduler's on each thread less that thread's own ledger, the latest of the four:

| arm | session | device_wall | window | vs M0 |
| --- | ------- | ----------- | ------ | ----- |
| M2 | A, n=12 | 33.53 ms | 31.73 ms | +3.7 % (window +2.5 %) |
| M2, silos | A, n=12 | 31.62 ms | 29.90 ms | -2.2 % (window -3.5 %) |
| M2, partitions, decision queue, 256-bit look-ahead | D, n=12 | 31.27 ms | 29.73 ms | -3.2 % (window -3.8 %) |
| **M2, partitions** | E, n=12 | **30.32 ms** | **28.82 ms** | **-6.2 % (window -6.8 %)** |

M0 reads 32.33, 32.30 and 32.32 ms in sessions A, D and E (windows 30.97, 30.91 and
30.93 ms). Every arm completes every task, the core time issued agrees to 0.2 %, and
the two reps of session E agree to 0.3 %.

**The predicates are off the critical path.** Every predicated piece -- 2,064 per
controller per round -- had its answer before it was ready, and none waited for the
answer to land: the decision queue went unused. With the decision queue alone
(session D), each of the 2,064 predicated pieces per controller per round
waited for its manager once it was ready, ~2.4 us on average, one at a time: chains
of up to five skipped supertasks paid a round trip per link.

**512 bits, because skipped tasks take indices.** A partition spans ~1,120 indices,
~450 of them tasks that skip; at 256 bits the look-ahead held a piece back 78 times
a round, at 512 never. A skipped task needs its index all the same: the controllers
count it finished like any other, so its consumers can wait for it in the
controller.

**What remains** against the oracle's -10.9 % is not attributed. Two counters
stand out, per controller and round: ~250 releases learned only after the queue's
replay had passed them, ~440 us of lateness in all -- a simulator artifact, under
[Limits](#limits) -- and 47 records that waited for the watermark to pass their
threshold after their named producers had finished, ~330 us in all. Every next
subpartition was resident before its controller switched to it.

An earlier build of partitions -- a software barrier at each boundary, positions
numbered per launch in task order, and predicated supertasks left with the managers
-- measured -5.8 % (window -7.5 %) in session C. Its window reads lower against a
slower `device_wall` because it spent 1.8 ms of each round outside the window, where
this build spends 1.5 ms and M0 1.4 ms.

## Limits

- **The partitioning is the host's**, as a stand-in for the compiler: a boundary
  moves within a quarter partition to the narrowest point, and nothing checks that
  the graph has one there.
- **The managers still evaluate every predicate.** That costs nothing here because
  the writer finishes early; a predicate whose writer finishes just before its task
  is ready still pays the manager's reaction.
- **Releases learned late.** A queue's model replays only when its manager touches
  it, so a peer's finish can reach it after the replay has passed the instant it
  would have released something; that release is placed late. Here that is ~250
  releases per controller per round, ~440 us of lateness in all. It counts against
  M2 only, so the gains above are lower bounds.
- **The exchange period and the store are design parameters**, not measurements.
- **The managers run in real time against a device that stands still while the
  simulator overruns**; `device_wall` removes it per thread, as on every arm.
- **qwen and paged_attention have not been run with partitions**, and neither has a
  sync-start cohort: qwen's 40 a round keep the ordinary path.

## Reproducing it

The partition arm is the expanded orchestration with `rt_declare_partitions(372)`
as the entry's first statement; everything else is as in
[validation/deepseek-v4.md](validation/deepseek-v4.md#reproducing-it).
`[GQ_PART thread=N]` in the device log carries, per manager and round, the pieces
posted, the predicates answered ahead and from the decision queue, the store's
peak, the exchanged finishes, the releases timed and late, and the waits on the
threshold, the look-ahead and late answers.

```bash
H_IN=1.02 H_N=1.32 PRED=0 C=4 SPLIT_MIN=8 PART=372 RESIDENT=1 CUT=narrow LAUNCH_US=1.32 \
  python docs/asim/analyzers/asim_silo_oracle.py <deps.json> <chip_swimlane_records.json> <calib>
```
