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
  cross, and builds the *partition plan* the managers pass to the controllers. On
  deepseek_v4 and qwen a partition is one layer, 372 and 277 supertasks.
- **Each group numbers its own share.** One local id stream per group runs across
  the partitions, each subpartition in dependency-level order: a piece of a
  supertask takes one id per block and active subtask, so its producers in the group
  sit below it. A supertask of 8 or more blocks, or one that must start all its
  blocks together, is split across the four groups, a quarter of its blocks each. A
  narrower one runs whole in one group. The host stamps it with its latest producer's
  group, or the next in turn; the plan instead deals it, in each partition's level
  order, to the group given the fewest blocks of its core types so far, unless the
  stamp already spreads the work as well -- as few waves of a group's cores per level
  and core type, and on a tie no more blocks in the busiest group. The stamp then
  stays: it keeps a task with its producer, so fewer producers need a global id.
  Conditional work keeps the stamp either way.
- **A producer run elsewhere stands as one local id, its proxy.** It goes just below
  its first dependent in the group, so it holds the group's watermark no longer than
  it has to. Every producer with a proxy anywhere gets a global id from the plan, and
  each group's *map* takes the global id to its proxy, with the number of pieces the
  proxy waits for -- four for a supertask split across the groups, the group's own
  piece included. The proxies an entry with more than four producers needs go where
  each producer's level falls instead, since its threshold covers most of them and
  stays low that way; one the next partition needs goes at the end of its producer's
  subpartition.
- **What an entry waits on.** The entries of a supertask's piece share one record:
  up to four named producers -- each one local range, a piece here or a proxy -- and
  a threshold, where the fifth-latest ends, that the group's watermark must reach.
  A producer in the partition before is named like any other, by its piece here or
  by its proxy at the end of its own subpartition. One further back is not named:
  the entry is posted only once every group maps its partition, which a group does
  once its watermark has passed the partition two before, so the producer has
  finished. An entry also waits until its id is within the 512-id look-ahead of the
  watermark, so its finish has a bit to land in.
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

## Sync-start tasks

A sync-start task's blocks must all start together: each waits on the others, and a
missing peer never arrives. qwen3-14B decode has one a layer, a 270 us mix kernel
over all 24 clusters.

- **Split across the groups as a cohort.** The plan splits it like any wide
  supertask, a quarter of its blocks a group, and flags each piece as a cohort; one
  nothing in its own partition waits on goes last in the partition's level order, so
  no group's watermark waits on it below the partition's other work. Its consumers
  name it through proxies, like any producer split across the groups, so they start
  once its pieces' global ids have gone round the ring.
- **Placed in order, ahead of other work.** A controller places only the entries of
  its current cohort piece -- the oldest it has not released -- so two cohorts can
  never each hold cores the other waits for. A current entry whose producers have
  finished goes to the head of its ring, ahead of the work it drains the cores for;
  the piece's vector entries go only once its cube entries are placed, so they do
  not hold vector cores the work still draining from cube cores may need. Each entry
  takes an idle core and holds it without starting.
- **Released whole by the managers.** The managers release the plan's cohorts one at
  a time, in plan order. Each reads how many of the current cohort's entries its
  queue has placed and publishes the count; the manager that finds the sum complete
  stamps the release; each starts its own share from the stamp, and the last to do so
  moves every manager to the next cohort.

Against the rest of the design this adds, per controller, the sequence of its cohort
pieces and the rule that places only the current one's, a ring-head insertion, and
the count of placed members its status already carried; and, between the managers
in the AICPUs' memory, the current cohort, each manager's placed count, the release
stamp and how many managers have released.

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
- **Cohort pieces hold their cores.** A cohort entry is placed only on an idle
  core, where it sits without a finish; the release starts each member at the stamp,
  or once its core came free if later, and only then counts its finish toward its
  piece, so the piece's global id goes round the ring once the members end. Each
  queue counts the core time its members held: the drain before each cohort.
- **Releases are replayed.** A queue's model advances when its manager touches it.
  A release due inside the interval being replayed is timed at its own instant; one
  learned only after the replay has passed it is placed where the replay stands and
  counted late.
- **The manager.** It installs its maps, posts the plan's items in order as its
  store and position window have room, leaving the room an earlier partition's held
  items need, and reads its queue's status whenever the controller may have moved,
  with or without positions out: a proxy landing moves the watermark that lets it
  map and post the next partition. While a cohort assembles it reads its queue's
  placed count on every pass until its share is placed.

## Measured

[validation/partitions.md](validation/partitions.md) has each case: qwen3-14B decode
2.1 % behind M0, at the design's own floor, deepseek_v4 9.8 % ahead, and
paged_attention 45.6 % ahead. What the design needed to run qwen, and the parts of it
dropped on the way, are in
[investigations](../investigations/2026-10-gq-partitions-qwen.md).

## Limits

- **The partitioning and the plan are the host's**, as a stand-in for the compiler:
  a boundary moves within a quarter partition to the narrowest point, and the plan
  rides at the end of the fanin pool -- 2.0 MB on deepseek_v4, uploaded with every
  bind, where a compiler would ship it once.
- **Every predicate is false on aSim.** The cores compute nothing, so `recv_count`
  stays zero and no routed expert runs: the numbers above are for that case, except
  those with half the experts made to run, by a predicate target of -1.
- **Not deadlock-free by construction.** An entry's id must come within the
  look-ahead before it starts: 2 of deepseek_v4's 172 subpartitions span more than
  512 ids, up to 562. No room is kept in the store for chains still to be dealt. And
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
- **A cohort drains its cores.** Its members hold each cube core from the moment it
  frees until the last one does -- 1.14 ms a cube core a round on qwen, against 1.0 ms
  in the oracle -- and an entry already pipelined behind running work cannot be taken
  back to shorten that: its finish is announced when it is placed.
- **qwen's and paged_attention's partitioned orchestrations are not checked in.**

## Reproducing it

deepseek_v4's partition arm is its expanded orchestration with
`rt_declare_partitions(372)` as the entry's first statement; everything else is as in
[validation/deepseek-v4.md](validation/deepseek-v4.md#reproducing-it). qwen's is the
`host_build_graph` copy of `decode_fwd_layers_expanded.cpp` with
`rt_declare_partitions(277)` first, run as in
[running.md](running.md#case-2--qwen3-14b-decode_fwd-40-layers) on `a2a3asimgq` with
eight rounds and the golden skipped, against the same source unpartitioned on
`a2a3asim` and `a2a3asimgq`. The host log carries one `Partition plan:` line per
bind: partitions, plan size, whether the narrow tasks were dealt or kept the stamp
and the level waves of each, the widest and mean subpartition, the largest map,
global ids and chains. `[GQ_PART thread=N]` in
the device log carries, per manager and round, the pieces and entries posted, the
map entries installed, home slots posted empty and left to the map, items held and
posted later, the passes that waited for room, for the controller, for another
manager's map and for its own, chains decided, dealt, posted and waiting for their
producers, regions filled; and from the queue the store's and the map's peaks, the
announcements sent, received, matched, received early and sent a cycle late, the
releases timed and late, the waits on the watermark for producers past the fourth
and for the look-ahead, the local ids the stream used, and how long items naming a
producer the managers dispatch themselves waited for it. `[GQ_WORK thread=N]` adds
`cohort_hold_us`, the cube and vector core time cohort members held their cores
before their release. Half the experts
run with `expert_rows_pred.target` set to -1 where
`expert_rows_pred.operand.indices[0]` is even, in the orchestration of both arms.

The oracle reads a swimlane only for which tasks ran; qwen has no predicates, so a
JSON listing every kernel task as `{"aicore_tasks": [[0, task_id], ...]}`, built from
its `deps.json`, stands in for one.

```bash
# the design as it stands (add COHORT=sync PART=277 for qwen):
SEEDS=10 PLACE=pick LOOK=512 \
  python docs/asim/analyzers/asim_local_oracle.py <deps.json> <chip_swimlane_records.json> <calib>
# before 2026-10-20 -- the stamp's placement, earlier partitions behind a barrier:
SEEDS=10 PROXY_AT=producer BARRIER=local LOOK=512 \
  python docs/asim/analyzers/asim_local_oracle.py <deps.json> <chip_swimlane_records.json> <calib>
# the shared-watermark design in the same machinery:
SEEDS=10 SHARED=1 RING=0.35 BARRIER=local LOOK=512 ENTER_US=0 \
  python docs/asim/analyzers/asim_local_oracle.py <deps.json> <chip_swimlane_records.json> <calib>
```
