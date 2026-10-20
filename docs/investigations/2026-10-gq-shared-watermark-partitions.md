# GroupQueue partitions on one shared watermark per partition

**Date**: 2026-10-20
**Verdict**: superseded by groups that number their share locally
([docs/asim/partitions.md](../asim/partitions.md))

## Question

Partitions let each controller run its share of a partition from resident entries,
resolving every dependency inside the partition without the managers. The first
design numbered each partition as a whole and had the controllers share its
watermark: is a partition-wide index, exchanged as a watermark and a look-ahead, the
right way for the groups to see each other's work?

## What was built

On `asim-fresh`, from 7a1805fce to the checkpoint `asim-route3-checkpoint-2026-10-19`
(5a3b091d5):

- **One index space per partition.** Every task a group runs took a partition-local
  index, one per block and active subtask, in dependency-level order; an index was
  11 bits, so a partition spanned at most 2,048.
- **One watermark per partition, shared.** Each controller tracked the partition's
  watermark and a 512-bit look-ahead past it, from its own finishes and the others'
  broadcasts. It broadcast its state -- partition, watermark, 512 bits, ~70 bytes --
  when it changed, at most once per 100 ns, and the others had it 300 ns later; a
  receiver merged by maximum and OR.
- **A record named producers by supertask.** Up to four named producer supertasks,
  each one index range whichever groups ran it, the queues running a piece of each,
  and a threshold where the fifth-latest ended; an entry also waited for the
  watermark to come within 512 of it.
- **The controllers moved together.** A controller reached partition `k + 1` once the
  watermark covered `k` whole -- every group's share of it -- so a slow group held
  the others at every boundary, and managers posted no further ahead than the
  partition after their controller's.
- **Predicated pieces posted decided**, and narrow predicated supertasks that run
  placed when decided on a ring every manager shared, ticket `k` going to group
  `k mod 4`, their consumers naming them in any queue
  ([conditional work](2026-10-gq-conditional-work-placement.md)).

Per controller this took a store of 1,024 entries, a watermark and a 512-bit
look-ahead for each partition in flight -- the current one and the next, four
contexts in the model -- the broadcast, three inbound ports to merge the others',
and comparators matching supertask ids and queue masks.

## Result

deepseek_v4 expanded on two dies, eight rounds a rep with the first two dropped,
each arm against the M0 of its own session:

| design | as-is | half the experts running |
| ------ | ----- | ------------------------ |
| predicated pieces answered when the controller asked, 256-bit look-ahead | -3.2 % | |
| predicated pieces posted up front and answered ahead | -6.2 % | |
| predicated pieces posted decided, host placement | -7.16 % (8 pairs, sd 0.19 pp) | +22.8 % (n=6) |
| conditional work placed when decided (the checkpoint) | -6.8 % | -6.0 % |

At 256 bits the look-ahead held a piece back 78 times a round, at 512 never: a
partition spanned ~1,120 indices, ~450 of them tasks that skip, since a skipped task
took its indices like any other. Each controller learned ~270 releases per round only
after its replay had passed them, ~1.05 ms of lateness in all, a simulator artifact
that counts against the GroupQueue.

Groups that number their share locally, in one stream each, and announce only the
global ids of producers another group needs, on a four-lane ring at 5 ns a hop,
measure -8.15 % as-is and -7.79 % with half the experts running against the same
M0, in jobs interleaving all three arms, two reps each -- -1.42 % and -1.41 % against
this design, which measured -6.82 % and -6.47 % in those jobs.

## Why not (now)

The exchange is what costs: the local-id oracle
([asim_local_oracle.py](../asim/analyzers/asim_local_oracle.py)) puts the shared
watermark at 27.49 ms with its 300 ns exchange and 27.19 ms with a 15 ns one, against
27.14 ms for local ids, so ~1.1 of the 1.3 points the local design gains come from
announcing a finish in 15 ns rather than broadcasting state every 100 ns. A
partition-wide watermark cannot be exchanged that fast: it is ~550 bits of state per
controller, merged from three peers, where a ring lane carries one 14-bit id. The
local design also drops the merge ports and two of the four watermark contexts, and
no group waits at a boundary for another group's unrelated work.

## When to reconsider

- If a graph's subpartitions span far more local ids than the look-ahead holds: a
  group's share of a partition then no longer fits its window, and a partition-wide
  index, which packs four groups' work into one space, may span less per group.
- If announcements arrive in bursts a lane cannot carry: a lane takes one global id a
  cycle, and a burst larger than the cycles between its consumers' needs delays them,
  where a broadcast carries any number of finishes at once.
- If a group's map would outgrow what a controller can hold, for a graph whose groups
  depend on many more of each other's producers than deepseek_v4's.

## References

- [docs/asim/partitions.md](../asim/partitions.md) -- the design that replaced it.
- [2026-10-gq-conditional-work-placement.md](2026-10-gq-conditional-work-placement.md)
  and [2026-10-gq-unnumbered-predicated-tasks.md](2026-10-gq-unnumbered-predicated-tasks.md)
  -- alternatives measured on this design.
- Commits 7a1805fce (partitions v2), 51388acc1 (predicated pieces posted decided),
  5a3b091d5 (conditional work placed when decided), on `asim-fresh`.
