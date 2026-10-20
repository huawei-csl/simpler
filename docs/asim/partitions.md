# Partitions: the controllers run the graph a partition at a time

[silos.md](silos.md) feeds each silo one supertask at a time, in task order, so
whatever stops the head of a silo's feed stops everything behind it. Here the graph
is split into *partitions*, each partition into one *subpartition* per group, and
each controller runs its subpartition from resident entries: every dependency is
resolved by the controllers, without the managers. Each group numbers its share
itself, and the only thing that crosses between groups is the global id of a
finished producer, on a ring.

The terms: a *task* is one function execution on one core, and a *supertask*
everything one submission launches. A *partition* is a subgraph; the partitions are
disjoint, cover the graph, and are numbered `0 .. P-1`. A *group* is one AICPU, its
controller and its cores -- what [silos.md](silos.md) calls a silo -- and its share
of a partition is its *subpartition*. A *local id* numbers the work of a group's
subpartitions, and is used only to track completion in that group.

The design this replaces kept one watermark per partition that the controllers
broadcast to each other; it is recorded in
[investigations](../investigations/2026-10-gq-shared-watermark-partitions.md).

## The design

- **Partitions are declared up front.** `rt_declare_partitions(target)` as an
  orchestration's first statement declares silos and asks for partitions of about
  `target` supertasks. Once the graph is built the host -- standing in for the
  compiler, which would do this ahead of time -- moves each boundary, within a
  quarter partition of where it would fall, to the point the fewest dependencies
  cross, and builds the *partition plan* the managers pass to the controllers. Here
  a partition is one layer: 372 supertasks.
- **Each group numbers its own share.** One local id stream per group runs across
  the partitions, each subpartition in dependency-level order: a piece of a
  supertask takes one id per block and active subtask, so its producers in the group
  sit below it. A supertask of 8 or more blocks is split across the four groups, a
  quarter of its blocks each; a narrower one runs whole in the group of its latest
  producer, or the next group in turn.
- **A producer run elsewhere stands as one local id, its proxy.** It goes just below
  its first dependent in the group, so it holds the group's watermark no longer than
  it has to. Every producer with a proxy anywhere gets a global id from the plan, and
  each group's *map* takes the global id to its proxy, with the number of pieces the
  proxy waits for -- four for a supertask split across the groups, the group's own
  piece included. The proxies an entry with more than four producers needs go where
  each producer's level falls instead, since its threshold covers most of them and
  stays low that way; one a later partition needs goes at the end of its producer's
  subpartition.
- **What an entry waits on.** The entries of a supertask's piece share one record:
  up to four named producers -- each one local range, a piece here or a proxy -- and
  a threshold, where the fifth-latest ends, that the group's watermark must reach.
  An entry with a producer in an earlier partition names none of them: its threshold
  is the subpartition's first id, which the watermark reaches once the group has
  finished every earlier share, proxies included. An entry also waits until its id is
  within the 512-id look-ahead of the watermark, so its finish has a bit to land in.
- **The ring.** Four 64-bit lanes, one per group, pass from each controller to the
  next every 5 ns cycle. When a piece's last entry finishes, its controller writes
  the piece's global id on its own lane in the first free cycle, one per cycle;
  every controller compares each lane, its own included, against its map in the
  cycle the lane reaches it, so a group k places along hears it k cycles later. A
  piece no other group maps is matched against its own map without taking a cycle.
  A proxy lands once every announcement it counts has arrived and the look-ahead
  covers it.
- **Maps go in before work.** A manager installs its group's map for a partition
  once its controller is within one partition of it, so a controller maps at most
  two, and posts the partition's items once all four managers have installed it --
  a counter the managers share in the AICPUs' memory. No announcement can then reach
  a controller that does not map it yet.
- **The managers pass the plan through.** The plan holds every item -- its local ids,
  named ranges, threshold and global id -- and every map; a manager adds positions,
  posts the items in order as its store frees, and holds back only what is not
  decided yet, below.

Against M2 this adds, per controller: a store of 1,024 entries (~20 bytes each); a
threshold and a look-ahead check per record beside four range comparators on local
ids; one watermark and a 512-bit look-ahead; the ring's four lanes, of which a global
id uses 14 bits on deepseek_v4; and the map, at most 323 entries there, each a global
id, a local id and two small counts, compared against four lanes a cycle. Nothing is
broadcast and nothing merged.

## Conditional work

A dispatch predicate lets a task skip itself: deepseek_v4's routed-expert tasks run
only when `recv_count[e] > 0`, the rows dispatched to expert `e`. Submitting to a
controller is committal -- an entry it takes executes -- so the decision is made
before anything is posted.

- **A chain is decided whole.** The plan groups the predicated supertasks narrower
  than 8 blocks into *chains*: those of one partition that share a predicate and
  depend on each other. deepseek_v4 has one per expert per layer, six supertasks
  taking 14 ids when they run. The host names the task that writes the predicate's
  element; once a manager sees it finished, it reads the element.
- **Skipped, a chain finishes at home.** Each of its tasks has one id, its home
  slot, in the group the host gave it; the home manager posts the slot as an empty
  piece, which finishes, without a core, once its producers have.
- **To run, a chain is dealt.** The manager of its first task's home group deals it:
  the k-th chain of a partition to run goes to group `(k + partition) mod 4`, so no
  group takes more than its share of any one partition. That group's manager posts
  the chain whole, into its *dynamic region* of the partition, once the chain's
  producers outside it are complete as the managers see them; its tasks name each
  other by local id, and their global ids complete the home slots and their
  consumers' proxies.
- **A region holds the largest share.** Each group's dynamic region of a partition
  has the ids of the partition's largest chains, a quarter of their number -- 112 on
  deepseek_v4 -- at the level its conditional work starts. Once every chain of the
  partition is decided and every one dealt to the group is posted, its manager marks
  the rest finished.

A chain goes whole because its links' finishes would otherwise have to reach the
managers: a manager learns of a finish through its queue position, which lags
behind the partition pieces posted ahead of it, and a link waiting on that can
deadlock -- the failure recorded in
[investigations](../investigations/2026-10-gq-unnumbered-predicated-tasks.md).

## How the simulator models it

- **The store.** Each queue keeps its entries resident in sets, a set's named
  ranges keyed by their first local id. A range is done once every id of it is known;
  the look-ahead holds a set by its last entry, whose id less 511 becomes part of its
  threshold.
- **The ring.** An announcement is stamped with its lane cycle on the real clock and
  read by the receiver, which converts it to its own clock and adds 5 ns a hop. A
  lane carries one announcement a cycle; one finding its cycle taken takes the next.
- **Mapped ids and unused ids land like sets.** A proxy waits for its announcements
  and the look-ahead, an unused id of a dynamic region for the look-ahead, and each
  is recorded at the instant it lands.
- **Releases are replayed.** A queue's model advances when its manager touches it.
  A release due inside the interval being replayed is timed at its own instant; one
  learned only after the replay has passed it is placed where the replay stands and
  counted late.
- **The manager.** It installs its maps, posts the plan's items in order as its
  store and position window have room, leaving the room an earlier partition's held
  items need, and reads its queue's status whenever the controller may have moved,
  with or without positions out: a proxy landing moves the watermark that lets it
  map and post the next partition.

## deepseek_v4: -8.2 % against M0, 1.4 % ahead of the shared watermark

**What the oracle expects.** [`analyzers/asim_local_oracle.py`](analyzers/asim_local_oracle.py)
models both designs in one machinery at the simulator's hops, one layer a partition.
With the shared watermark's exchange -- 300 ns, after up to 100 ns for a broadcast
slot -- it reads 27.49 ms over ten draws, and local ids 27.14 ms (-1.3 %, sd 0.1 pp).
Almost all of it is the exchange: the shared watermark with a 15 ns exchange reads
27.19 ms, and local ids with the old exchange 27.44 ms, so the local watermark alone
is worth ~0.2 %. With half the experts running it puts the two level, 36.77 against
36.72 ms, dealing each conditional task once the managers see its producers
complete, 3.7 us after they finish.

**Measured.** The expanded orchestration with `rt_declare_partitions(372)` as its
first statement, against M0 and against the shared-watermark design at its checkpoint
(`asim-route3-checkpoint-2026-10-19`), the three interleaved in every job, eight rounds
a rep with the first two dropped, two reps (n=12 rounds an arm):

| arm | as-is | vs M0 | half the experts running | vs M0 |
| --- | ----- | ----- | ------------------------ | ----- |
| M0 | 32.31 ms | | 39.20 ms | |
| shared watermark | 30.10 ms | -6.82 % | 36.69 ms | -6.47 % |
| **local ids** | **29.69 ms** | **-8.15 %** | **36.15 ms** | **-7.79 %** |

Paired rep by rep against the shared watermark, local ids measure -1.42 % as-is
(range -1.59 to -1.26 %; window -1.52 %) and -1.41 % with half the experts running
(range -1.58 to -1.23 %), slightly more than the oracle's 1.3 points. Every arm
completes every task, and the core time issued agrees to 0.2 % as-is and 0.6 % with
experts running.

| per controller and round | local ids, as-is | local ids, experts | shared, as-is | shared, experts |
| ------------------------ | ---------------- | ------------------ | ------------- | --------------- |
| store peak (most of any) | 420 (432) | 420 (432) | 497 (512) | 493 (509) |
| map peak | 322 (323) | 322 (323) | | |
| finishes learned from the others | 8,226 | 10,290 | 12,814 | 12,814 |
| releases learned late, in all | 283, 0.81 ms | 153, 0.17 ms | 272, 1.01 ms | 64, 0.27 ms |
| waits for producers past the fourth | 61, 0.42 ms | 43, 2.0 ms | 40, 0.24 ms | 43, 0.61 ms |

Each controller announces 2,742 global ids a round as-is and 3,430 with experts
running; pieces finishing in the same cycle queue for the lane, ~4,400 cycles of 5 ns
a round as-is and ~2,100 with experts running. With half the experts running, every
manager posts 172 chains a round, four in every partition, and finds a chain waiting
for its producers outside it ~690 times. Producers past the fourth wait longer there
because the combine's threshold, not the four producers it names, is what holds it
for the chains still running. An entry with a producer in an earlier partition waits
at the barrier ~26 times a round, 0.6 to 0.8 ms each: it was posted a partition
ahead, and the barrier is its whole wait. No announcement reached a
controller before its map, and no range was named inconsistently.

## Limits

- **The partitioning and the plan are the host's**, as a stand-in for the compiler:
  a boundary moves within a quarter partition to the narrowest point, and the plan
  rides at the end of the fanin pool -- 1.9 MB on deepseek_v4, uploaded with every
  bind, where a compiler would ship it once.
- **Every predicate is false on aSim.** The cores compute nothing, so `recv_count`
  stays zero and no routed expert runs: the numbers above are for that case, except
  those with half the experts made to run, by a predicate target of -1.
- **Not deadlock-free by construction.** An entry's id must come within the
  look-ahead before it starts: 2 of deepseek_v4's 172 subpartitions span more than
  512 ids, up to 568. No room is kept in the store for chains still to be dealt. And
  the managers still decide every chain, through the writer's completion.
- **The managers learn of finishes late.** A manager's status covers its queue's
  prefix and 32 positions past it, so a dealt chain waits for its producers outside
  it until every manager running a piece of them has retired that piece.
- **Releases learned late.** A queue's model replays only when its manager touches
  it, so an announcement can reach it after the replay has passed the instant it
  would have released something; that release is placed late. It counts against M2
  only, so the gains above are lower bounds.
- **The ring's cycle, a hop per cycle and the store are design parameters**, not
  measurements.
- **The managers run in real time against a device that stands still while the
  simulator overruns**; `device_wall` removes it per thread, as on every arm.
- **qwen and paged_attention have not been run with partitions**, and neither has a
  sync-start cohort: qwen's 40 a round keep the ordinary path.

## Reproducing it

The partition arm is the expanded orchestration with `rt_declare_partitions(372)`
as the entry's first statement; everything else is as in
[validation/deepseek-v4.md](validation/deepseek-v4.md#reproducing-it). The host log
carries one `Partition plan:` line per bind: partitions, plan size, the widest and
mean subpartition, the largest map, global ids and chains. `[GQ_PART thread=N]` in
the device log carries, per manager and round, the pieces and entries posted, the
map entries installed, home slots posted empty and left to the map, items held and
posted later, the passes that waited for room, for the controller, for another
manager's map and for its own, chains decided, dealt, posted and waiting for their
producers, regions filled; and from the queue the store's and the map's peaks, the
announcements sent, received, matched, received early and sent a cycle late, the
releases timed and late, the waits on the watermark for producers past the fourth,
the look-ahead and the barrier, and the local ids the stream used. Half the experts
run with `expert_rows_pred.target` set to -1 where
`expert_rows_pred.operand.indices[0]` is even, in the orchestration of both arms.

```bash
SEEDS=10 PROXY_AT=producer BARRIER=local LOOK=512 \
  python docs/asim/analyzers/asim_local_oracle.py <deps.json> <chip_swimlane_records.json> <calib>
# the shared-watermark design in the same machinery:
SEEDS=10 SHARED=1 RING=0.35 BARRIER=local LOOK=512 ENTER_US=0 \
  python docs/asim/analyzers/asim_local_oracle.py <deps.json> <chip_swimlane_records.json> <calib>
```
