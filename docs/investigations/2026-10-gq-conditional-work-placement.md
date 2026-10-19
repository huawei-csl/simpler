# GroupQueue partitions: conditional work placed with room kept, or dealt by quota

**Date**: 2026-10-19
**Verdict**: dropped -- conditional work goes through a shared ring with no room
kept, and the store holds 1,024 entries
([partitions.md](../asim/partitions.md#conditional-work-is-placed-when-decided))

## Question

With partitions, the host places each predicated supertask of fewer than 8 blocks
whole in one group, ahead of time. With half of deepseek_v4's experts made to run --
a predicate target of -1 on even experts -- each expert's chain landing by the
parity of its number put ~78 % of a layer's running expert work in one group, and
the partitions measured +22.8 % against M0 (n=6); with the running experts spread
(`e mod 4 < 2`) the same placement measured -0.2 %. So such a task is placed only
once it is decided. A task placed late reaches a store the next partition's gated
pieces may have filled, so the question is how to place it and what keeps the stores
from deadlocking.

## What was tried

On aSim, deepseek_v4 expanded on two dies, `rt_declare_partitions(372)`, eight
rounds a rep with the first two dropped, each rep paired with M0 in the same job,
two or three reps an arm -- the host placement eight as-is and one with experts
running. *As-is* is the case as aSim runs it, every predicate false; *experts* has
the even experts running.

| design | as-is | experts | room for late work |
| ------ | ----- | ------- | ------------------ |
| host placement, 512 entries | -7.2 % | +22.8 % | not needed |
| ring, room kept, 512 entries | -5.2 % | -4.9 % | guaranteed |
| dealt by quota, 512 entries | -5.9 % | -5.5 % | guaranteed |
| ring, no room kept, 512 entries | -7.0 % | -5.8 % | not guaranteed |
| **ring, no room kept, 1,024 entries** | **-6.8 %** | **-6.0 %** | not guaranteed |

With experts running the busiest store peaked at 468 entries with room kept, 438
dealt, 512 -- full -- with no room kept at 512 entries, and 504 at 1,024.

- **Ring, room kept.** A task decided to run goes on a ring every manager shares,
  the k-th to group `k mod 4`; a skipped one goes empty to the group the host gave
  it. Besides its own held pieces, a later partition leaves each group an even share
  of an earlier partition's undecided conditional work, plus what was put on the
  ring for it and not yet posted. As-is that room is held for work that never comes:
  1.8 pp against the same ring without it.
- **Dealt by quota.** Every decided conditional task, run or empty, is dealt to a
  group under a per-partition quota of `ceil(C/4)` plus the widest task less one
  entries -- runs to the group with the fewest run entries, empties to the fewest in
  all -- through a ring per group; the room kept is the quota not yet posted. It
  holds the guarantee with little room held (~10 waits for room per manager per
  round, against 823 for the host placement), but costs ~1 pp as-is: building takes
  the AICPU 0.43 ms more per manager per round (1.49 to 1.92 ms), and releases learned
  after the queue's replay passed them rise from ~250 to ~990 per manager per round
  (~1.1 to ~3.3 ms), a simulator artifact whose share of the critical path was not
  measured.
- **Splitting** those supertasks from 2 blocks across the groups, a temporary host
  change, measured +3.9 % with experts running (one rep).

## Why not (now)

Keeping room, either way, guarantees the run against a full store and costs 1 to
1.8 pp as the case runs, with no better experts number. Without it the run depends
on the store's size: 1,024 entries, at most 512 used, against a worst case of 1,344
in one group's store for two consecutive partitions of deepseek_v4.

## When to reconsider

For a graph whose two consecutive partitions can overfill a store: keep room then --
the ring's reservation at 1,024 entries has not been measured -- or deal by quota.

## References

- [docs/asim/partitions.md](../asim/partitions.md) -- the adopted design and its
  measurement.
