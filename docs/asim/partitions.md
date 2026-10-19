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
- **Subpartitions are fixed, but for conditional work.** A supertask of 8 or more
  blocks is split across the four groups, each taking a quarter of its block range;
  a narrower one runs whole in the group of its latest producer, or the next group in
  turn. Group `g`'s share of a partition is its subpartition, and reaches its
  controller through its own manager. A narrower predicated supertask is placed when
  it is decided: skipped, in the group the host gave it; to run, in whichever group's
  turn it is -- [below](#conditional-work-is-placed-when-decided).
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
- **Predicated pieces are posted decided.** Submitting to a controller is
  committal: every entry it takes executes. The manager holds a predicated piece back
  until the value its predicate reads is final, then posts it to run or as an empty
  piece -- [below](#predicates-in-hardware).
- **What stays with the manager.** A sync-start cohort keeps the ordinary path, as
  with silos; deepseek_v4 has none. A supertask with a producer the controllers do
  not run is held back and fed once that producer has finished. A later partition's
  pieces leave room in the store for every piece held back from an earlier one, which
  its controller needs before it can move on. Nothing is kept for conditional work
  still to come: the store's size leaves room for it, not a guarantee --
  [Limits](#limits).

Against M2 this adds, per controller: a store of 1,024 entries (~20 bytes each),
which deepseek_v4 fills to at most 512; a threshold and a look-ahead check per record
beside M2's four comparators, which match index ranges, and one bit per named
producer, *any queue*, for conditional work; a watermark and 512-bit look-ahead per
partition in flight, the current one and the next; the broadcast of the partition,
the watermark and the 512 bits, ~70 bytes; and one bit per record, *empty*, for a
predicated piece that skips. The ring the managers share is software, in the
AICPUs' memory.

## Predicates in hardware

A dispatch predicate lets a task skip itself: deepseek_v4's routed-expert tasks run
only when `recv_count[e] > 0`, the rows dispatched to expert `e`. That is six
supertasks per expert, 192 a layer, and a task that skips still has to finish in
order, because its consumers wait on it. Submitting to a controller is committal --
an entry it takes executes, and nothing is withdrawn or answered later -- so the
decision is made before the piece is submitted. Three parties share the work.

- **The host names the writer.** At submit it records the last task to write the
  element a predicate reads (`TaskDescriptor::predicate_writer`, from the tensormap
  it already keeps). The predicate's value is final once that task has finished --
  here `dispatch_meta`, one per layer, which finishes before the expert tasks' input
  rows have arrived.
- **The manager decides, then posts.** It holds a predicated piece back until the
  writer has finished -- which it learns from the completions the managers retire
  anyway -- reads the element, and posts the piece decided: to run, or as an empty
  piece. When the host cannot name the writer, in a manual scope, the manager waits
  for every producer of the task instead: that is when M0 reads the predicate. The
  piece's consumers are posted as usual and wait on its indices, which finish
  either way.
- **The controller finishes empty pieces.** One bit per record, *empty*. An empty
  piece waits like any other -- for its producers, threshold, look-ahead and
  partition -- and then finishes its tasks at once, without a core; the controller
  broadcasts them like any finish.

An empty piece waits for its conditions rather than finishing on arrival: a skipped
task finishing before its own producers would let its consumers start before them,
out of order with any write after their reads. Because a held piece reaches the
store after the pieces behind it, the manager keeps room for it: a later partition's
pieces leave free the entries an earlier partition's held pieces will need, or the
store could fill with pieces gated behind a switch that cannot come.

Two variants cost more hardware. Posting predicated pieces up front and answering
them later -- built and measured below -- needs two bits per record, *pending* and
*skip*, and a port for the answers. A controller that read the element itself would
need a device-memory read port, a comparator and ~18 bytes of predicate per record,
and would take the manager off the path; on deepseek_v4 the writer finishes early
enough that the manager is not on it either.

### Conditional work is placed when decided

A predicated supertask of fewer than 8 blocks -- each of deepseek_v4's expert
supertasks -- runs whole in one group. Placed by the host, it lands where its chain
falls, by the parity of the expert's number: with the even experts running, ~78 % of
a layer's running expert work sat in one group and the partitions measured +22.8 %
against M0 (n=6). Which tasks run is known only from the data, so the placement waits
for the decision:

- **Skipped, it stays.** The manager of the group the host gave it posts it there as
  an empty piece, as above.
- **To run, it goes on a ring every manager shares.** The k-th task put there is
  posted by manager `k mod 4`, whole, to its own controller, the tickets of a manager
  in the order put. A ticket for a partition beyond the one after its controller's
  waits, and so do that manager's tickets behind it, until the controller switches.
- **Its consumers name it in any queue.** One bit, *any queue*, on the named
  producer: the first queue to report the supertask covers its whole index range.

Nothing is kept in a store for such a task before it arrives. Keeping each group an
even share of an earlier partition's undecided conditional work, besides its own held
pieces, cost 1.8 pp with 512 entries (-5.2 % against M0 instead of -7.0 %); the store
was doubled instead, to 1,024 entries.

## How the simulator models it

- **The store.** Each queue keeps its entries resident in records, a record's named
  producers keyed by supertask. A producer is done for a record once every queue
  that runs a piece of it has reported that piece, which a queue learns at the same
  instant as the indices the piece covers -- or, named in any queue, once one queue
  has reported it. The look-ahead holds a record by its last entry: its threshold
  rises to that entry's index less 512.
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
  the one before. It holds a predicated piece back until it sees the writer
  completed, or every producer of the task when the host named no writer, and posts
  it then, to run or empty, on the path held pieces take. Conditional work that runs
  goes on a 16,384-slot multi-producer ring instead; each manager takes its own
  tickets in order and posts each task as its store frees.

## deepseek_v4: the partition boundary is gone, and so are the predicate stops

**What the oracle expects.** [`analyzers/asim_silo_oracle.py`](analyzers/asim_silo_oracle.py)
at the simulator's hops, one layer a partition cut at narrow points, supertasks
split from 8 blocks, and the switch costing a cross-group hop (`LAUNCH_US=1.32`),
against 30.66 ms today. The predicate column is what the path through a predicated
task costs above the controllers' own hops:

| predicate path | makespan | vs today |
| -------------- | -------- | -------- |
| free: decided before the task is ready | 27.32 ms | -10.9 % |
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
| M2, partitions, answered when asked, 256-bit look-ahead | D, n=12 | 31.27 ms | 29.73 ms | -3.2 % (window -3.8 %) |
| M2, partitions, answered ahead | E, n=12 | 30.32 ms | 28.82 ms | -6.2 % (window -6.8 %) |
| M2, partitions, answered ahead | F, n=12 | 30.38 ms | 28.83 ms | -6.0 % (window -6.7 %) |
| M2, partitions, posted decided | F, n=48 | 30.00 ms | 28.46 ms | -7.2 % (window -8.0 %) |
| M2, partitions, posted decided | K, n=12 | 30.02 ms | 28.47 ms | -7.1 % (window -8.0 %) |
| **M2, partitions, conditional work placed when decided** | K, n=12 | **30.13 ms** | **28.52 ms** | **-6.8 % (window -7.8 %)** |

M0 reads 32.33, 32.30, 32.32, 32.32 and 32.34 ms in sessions A, D, E, F and K
(windows 30.97, 30.91, 30.93, 30.93 and 30.93 ms). Every arm completes every task and
the core time issued agrees to 0.2 %. Session F pairs each rep of the decided design
with an M0 rep run beside it in the same job, the two in alternating order: over
eight pairs the speedup is -7.16 % (sd 0.19 pp, 95 % interval -7.31 to -7.00 %, every
pair between -7.45 and -6.87 %), and -8.00 % in the window (sd 0.13 pp). Session K
pairs two reps of each design that way: conditional work placed when decided
measures -6.81 % (sd 0.29 pp; window -7.83 %) and the host's placement -7.11 % (sd
0.11 pp), the first trailing in both pairs, by 0.2 and 0.4 pp -- a difference two
reps do not resolve. As the case runs no task goes through the ring, every predicate
being false. With 1,024 entries no manager waited for room and the store peaked at
511; the host's placement, with 512, waited 823 times per manager per round and
filled the store.

**With half the experts running, placing conditional work when decided turns the
loss into a win.** A predicate target of -1 on even experts, on both arms, makes 16
of the 32 experts run. In session K, M0 reads 39.18 ms (window 37.80 ms) and
conditional work placed when decided 36.87 ms (window 35.23 ms): -5.95 % over two
paired reps (sd 0.12 pp; window -6.78 %), where the host's placement measured
+22.8 % (n=6). The core time issued agrees to 0.6 %. Each manager posted 1,032 tasks from
the ring per round, the store peaked at 504 entries, and no manager waited for room.

**Deciding before posting costs nothing here and frees the store.** Every predicated
piece -- 2,064 per controller per round -- was held until `dispatch_meta` had
finished and then posted decided; with every predicate false on aSim, all went
empty. Held back, they no longer sit in the store waiting for an answer: a manager
waited for room 687 times a round, against 4,131 times when the same pieces were
posted up front and answered, and the store peaked at 498 entries rather than all
512. Their ~4,700 entries reach their controller after it has reached their
partition, which the answered design never does.

In session D the manager answered only when its controller asked, through a decision
queue the controller posted ready pieces to: each of the 2,064 predicated pieces per
controller per round waited for its manager once it was ready, ~2.4 us on average,
one at a time, and chains of up to five skipped supertasks paid a round trip per
link. Without a named writer, a predicate is final only once every producer of its
task has finished as the managers see it: answered then, each piece waited ~7.3 us
once it was ready, and the case measured 32.20 ms against 30.36 ms with the writers
named, two rounds each -- a manager sees a producer finished only once every manager
running a piece of it has retired that piece, later than the controllers know it.

Leaving predicated tasks out of the partitions -- no index, the managers reading the
predicate at dispatch and holding back the tasks that name one -- measured -4.2 %
in session F and deadlocks once a predicated task runs:
[investigations](../investigations/2026-10-gq-unnumbered-predicated-tasks.md).

**512 bits, because skipped tasks take indices.** A partition spans ~1,120 indices,
~450 of them tasks that skip; at 256 bits the look-ahead held a piece back 78 times
a round, at 512 never. A skipped task needs its index all the same: the controllers
count it finished like any other, so its consumers can wait for it in the
controller.

**What remains** against the oracle's -10.9 % is not attributed. Two counters
stand out, per controller and round: ~270 releases learned only after the queue's
replay had passed them, ~1.05 ms of lateness in all -- a simulator artifact, under
[Limits](#limits) -- and ~40 records that waited for the watermark to pass their
threshold after their named producers had finished, ~255 us in all. Apart from the
held predicated pieces, every next subpartition was resident before its controller
switched to it.

An earlier build of partitions -- a software barrier at each boundary, positions
numbered per launch in task order, and predicated supertasks left with the managers
-- measured -5.8 % (window -7.5 %) in session C. Its window reads lower against a
slower `device_wall` because it spent 1.8 ms of each round outside the window, where
this build spends 1.5 ms and M0 1.4 ms.

## Limits

- **The partitioning is the host's**, as a stand-in for the compiler: a boundary
  moves within a quarter partition to the narrowest point, and nothing checks that
  the graph has one there.
- **The managers still decide every predicate.** That costs nothing here because
  the writer finishes early; a predicated piece whose writer finishes just before
  the piece would be ready reaches its controller late, and so does one whose writer
  the host cannot name.
- **Every predicate is false on aSim.** The cores compute nothing, so `recv_count`
  stays zero and no routed expert runs: the numbers above are for that case, except
  those with half the experts made to run, by a predicate target of -1.
- **Nothing is kept in the store for conditional work still to come.** A
  conditional task that finds its store full waits for room; if the store is full of
  the next partition's pieces, gated until a switch that needs that task, the run
  stalls. The 1,024 entries leave room to spare here -- at most 512 used -- but not
  by construction: two consecutive partitions can put up to 1,344 entries in one
  group's store on deepseek_v4. A store of 2,048 entries, and a position window to
  match, would rule it out; so would keeping room for the work, at the cost measured
  [above](#conditional-work-is-placed-when-decided).
- **The managers learn of finishes late.** A manager's status covers its queue's
  prefix and 32 positions past it, and partition pieces are posted far ahead, so a
  finish further out is reported only when the prefix reaches it. A task on the
  managers' ordinary path inside a partitioned graph can deadlock on it.
- **Releases learned late.** A queue's model replays only when its manager touches
  it, so a peer's finish can reach it after the replay has passed the instant it
  would have released something; that release is placed late. Here that is ~270
  releases per controller per round, ~1.05 ms of lateness in all. It counts against
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
posted, the predicated pieces held and posted decided, how many of them had no
writer named and how many went empty, the store's peak, the exchanged finishes, the
releases timed and late, the waits on the threshold and the look-ahead, and the
conditional tasks put on the shared ring and posted from it (`routed_out`,
`routed_in`). Half the experts run with `expert_rows_pred.target` set to -1 where
`expert_rows_pred.operand.indices[0]` is even, in the orchestration of both arms.

```bash
H_IN=1.02 H_N=1.32 PRED=0 C=4 SPLIT_MIN=8 PART=372 RESIDENT=1 CUT=narrow LAUNCH_US=1.32 \
  python docs/asim/analyzers/asim_silo_oracle.py <deps.json> <chip_swimlane_records.json> <calib>
```
