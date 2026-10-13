# Partitions: launching the graph a chunk at a time

[silos.md](silos.md) feeds each silo one supertask at a time, in task order, so
whatever stops the head of a silo's feed stops everything behind it. This variant
launches a *partition* -- a chunk of the graph in task order -- whole, and lets the
controllers resolve every edge inside it themselves. The terms are silos.md's: a
*task* is one function execution on one core, a *supertask* everything one
submission launches.

## The design

- **The host cuts the graph.** `rt_declare_partitions(target)` as an
  orchestration's first statement declares silos, as `rt_declare_silos()` does, and
  asks for partitions of about `target` supertasks. Once the graph is built the
  host moves each boundary, within a quarter partition of where it would fall, to
  the point the fewest dependencies cross, and stamps every supertask with its
  partition.
- **A launch hands each controller its whole subpartition.** Every task of the
  partition its silo runs -- its block range of a spread supertask, all of a
  narrower one -- reaches the controller in one launch and is resident from then
  on, waiting on its producers there. Nothing migrates between silos.
- **Producers are named by supertask.** Positions are numbered across the
  partition, so a supertask's pieces in every silo are one range, and a consumer
  names each producer with one comparator, whatever silos run it. A producer in an
  earlier partition has finished by the launch and is not named.
- **Controllers exchange completions.** A controller sees its own piece of a
  supertask finish at once, and the other three learn of it 200 ns later
  (`SIM_GQ_EXCHANGE_NS`) -- the period at which the design has the controllers
  exchange their watermark and look-ahead state. A producer is finished for a
  consumer once every silo running a piece of it has reported.
- **One partition in flight.** The next partition launches once every supertask of
  the one in flight has finished, which the managers count as they retire tasks. Each
  manager builds its share of the next launch while the current partition runs --
  every producer in an earlier partition will have finished by the time it is
  posted -- so a boundary only takes positions and posts.
- **What stays with the manager.** A supertask with a dispatch predicate or a
  sync-start cohort keeps the ordinary path, as with silos. One naming such a
  producer, or more than four producer supertasks, cannot be resolved by its
  controller alone: its manager holds it back from the launch and feeds it once its
  producers allow.

Against M2 this adds a store per controller that holds a subpartition -- 234 tasks
at its peak here, against M2's 32 positions -- the exchange between the controllers,
and comparators that match a producer range in partition numbering. The comparator
count stays M2's four.

## How the simulator models it

- **The store.** Each queue keeps its launched entries resident, keyed by the
  supertask each is a piece of. Every entry of one piece shares one record of the
  producers it waits on, as with silos; in hardware each slot carries its own
  comparators. A ready entry is placed when its core type has room, and a launched
  entry is never stolen by another core.
- **The exchange.** A queue records its piece of a supertask finished once every
  entry of it has, and posts that on a single-producer ring to each other queue,
  stamped on the real clock. The receiver converts it to its own clock, adds the
  exchange, and replays the release together with its own retirements, earliest
  first. One learned only after its replay has passed it is placed where the replay
  stands and counted late.
- **The boundary is the managers'.** The count of finished tasks per partition is
  kept by the scheduler threads, in software; the controllers know nothing of
  partitions beyond the entries they hold.

## deepseek_v4: partitions beat the feed, and the boundary is what remains

**What the oracle expects.** [`analyzers/asim_silo_oracle.py`](analyzers/asim_silo_oracle.py)
prices partitions with `PART`, `RESIDENT`, `CUT` and `LAUNCH_US` (3.7 us), at the
simulator's hops with the 200 ns exchange (`H_IN=1.02 H_N=1.22`), against 30.66 ms
today. With predicates free, and the most blocks one silo holds at once in the last
column:

| partitions | in flight | makespan | vs today | blocks resident |
| ---------- | --------- | -------- | -------- | --------------- |
| the whole graph | 1 | 27.24 ms | -11.2 % | all |
| one layer each, cut evenly | 1 | 28.74 ms | -6.3 % | 174 |
| one layer each, cut at narrow points | 1 | 27.39 ms | -10.7 % | 271 |
| half a layer each, cut at narrow points | 1 | 27.76 ms | -9.4 % | 142 |
| half a layer each, cut evenly | 2 | 27.27 ms | -11.1 % | 168 |

A partition that waits for the one before it pays for whatever the cut splits: work
that would have overlapped the boundary becomes a bubble, about 30 us a boundary at
an even cut. Moving each boundary to where the fewest dependencies cross leaves
little to overlap, so one partition at a time comes within half a point of two in
flight, and that is what is built here: 372 supertasks a partition -- one layer --
cut at narrow points. With predicates left with the manager (`PRED=2.68`) and four
comparators (`C=4`), as here, the oracle gives **27.94 ms, -8.9 %**. Launching the
next partition once only the tasks it depends on have finished (`LAUNCH_ON=cut`)
changes nothing: those are the last to finish.

**Measured.** The expanded orchestration with `rt_declare_partitions(372)` as its
first statement, interleaved with M0 per job, eight rounds a rep with the first two
dropped. Each row is against the M0 of its own session, and the window is the
scheduler's on each thread less that thread's own ledger, the latest of the four:

| arm | session | device_wall | window | vs M0 |
| --- | ------- | ----------- | ------ | ----- |
| M2 | A, n=12 | 33.53 ms | 31.73 ms | +3.7 % (window +2.5 %) |
| M2, silos | A, n=12 | 31.62 ms | 29.90 ms | -2.2 % (window -3.5 %) |
| M2, partitions, each launch built at its boundary | B, n=6 | 31.26 ms | 29.51 ms | -3.0 % (window -4.5 %) |
| **M2, partitions** | C, n=12 | **30.39 ms** | **28.58 ms** | **-5.8 % (window -7.5 %)** |

M0 reads 32.33, 32.23 and 32.27 ms in the three sessions (windows 30.97, 30.89 and
30.90 ms). Every arm completes every task, the core time issued agrees to 0.3 %,
and the two reps of session C agree to 0.1 %.

**The feed's stops are gone.** Per silo per round, 43 launches carry 7,178 tasks,
and nothing waits for room. 7,381 piece completions arrive from the other
controllers and 2,405 releases are timed on one, 6 of them after the controller's
replay had passed it, by 16.5 us in all. Named by supertask, only 43 supertasks have
more producers running than four comparators hold; with the 169 that name a
predicated producer, 212 are held back and fed later -- each holding only itself,
where the feed stopped ~1,700 times a round and held everything behind each stop.

**What remains is the manager's share of the boundary and of the predicated
paths.** Building a silo's share of a launch takes 27 us -- 1.14 ms a round -- and
posting it 7 us of model time: taking positions and accounting the blocks, with the
queue's own submit modelled at 0.5 us and the rest of its cost on the ledger. Built
at the boundary, both sit on the critical path 43 times a round; built while the
partition before it runs, only the post does, and that is the step from -3.0 % to
-5.8 %.

The oracle prices the rest. It charges a launch 3.7 us and the path through a
predicated task one manager hop, 2.68 us, and each microsecond more costs ~42 us a
round at the boundaries and ~210 us on the predicated paths. A 9 us boundary -- the
retirement, the post and the placement after it -- gives 28.15 ms (-8.2 %). The
measured -7.5 % also needs the predicated path about a microsecond dearer than the
oracle's, as it is here: a supertask held back behind a predicated producer waits
for its manager to see the producer finish, then to build and post it.

## Limits

- **The exchange period and the store are design parameters.** 200 ns is the
  design's figure, not a measurement, and the store's size is the model's: 2,048
  resident entries and 4,096 supertask keys per controller, far more than a
  partition here needs.
- **One partition in flight, behind a software count.** The boundary is the
  managers': they count retirements and post the next launch, so it costs a
  retirement's latency plus the post.
- **Predicated supertasks never join a partition**, and one that names a
  predicated producer is held back. A predicate is read when its supertask is
  ready, so the manager dispatches or retires each one, as with silos.
- **The cut is the host's.** A boundary moves within a quarter partition to the
  narrowest point; nothing checks that the graph has one there.
- **The managers run in real time against a device that stands still while the
  simulator overruns**, about 25 ms a round here against M2's 17 ms. `device_wall`
  removes it per thread, as on every arm.
- **qwen and paged_attention have not been run with partitions.**

## Reproducing it

The partition arm is the expanded orchestration with `rt_declare_partitions(372)`
as the entry's first statement; everything else is as in
[validation/deepseek-v4.md](validation/deepseek-v4.md#reproducing-it).
`[GQ_PART thread=N]` in the device log carries the launches, the held-back
supertasks, the boundary time, the build and post time -- with the submit's share
and how much of that is on the ledger -- the store's peak and the exchanged
completions above.

```bash
H_IN=1.02 H_N=1.22 PRED=2.68 C=4 PART=372 RESIDENT=1 CUT=narrow LAUNCH_US=3.7 \
  python docs/asim/analyzers/asim_silo_oracle.py <deps.json> <chip_swimlane_records.json> <calib>
```
