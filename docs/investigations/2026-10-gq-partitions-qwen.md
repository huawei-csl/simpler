# GroupQueue partitions on qwen: what the first plan got wrong

**Date**: 2026-10-20
**Verdict**: three parts of the partition design as first built, and two
alternatives tried on the way, dropped for what
[docs/asim/partitions.md](../asim/partitions.md) now describes

## Question

Partitions were built and measured on deepseek_v4 and paged_attention, neither of
which has a sync-start task. qwen3-14B decode, 40 layers expanded to 11,087 tasks, has
one a layer: a 270 us mix kernel over all 24 clusters, whose blocks must start
together. Run with `rt_declare_partitions(277)` -- a layer a partition -- what does the
design need to run qwen well, and how close can it come to M0?

qwen is cube-bound: 30.7 ms of cube work per core a round, against M0's 33.2 ms, so
any scheduler's room is a few percent.

## What was tried, and what it measured

All on `a2a3asimgq` with `v3_qwen.calib`, rounds after the second; A/B jobs interleave
M0, plain M2 and the partition arm, two or three reps of eight rounds.

| design | device_wall | vs M0 |
| ------ | ----------- | ----- |
| as first built, once a dropped cohort member no longer stalled it | ~50 ms | +51 % |
| plan places narrow tasks by load | ~38 ms | +15 % |
| cohort members drain the cores ahead of partition work | 35.87 ms | +8.1 % |
| cohort as partition pieces, barrier kept | ~35.8 ms | +7.8 % |
| producers in the partition before named, no barrier | 34.19 ms | +2.9 % |
| members at the head of their ring, the cohort list in the plan | 33.95 ms | +2.1 % |

The rows without an A/B are three-round smokes (rounds two and three) against
M0's 33.2 ms.

- **The host's placement, always.** The silo stamp runs a narrow task in the group
  of its latest stamped producer, so a chain stays where its data is. qwen's layer
  fans out from five producers into 110 tasks and back, and the stamp put 2,281,
  5,247, 2,760 and 6,040 entries in the four groups. The oracle reads 42.8 ms for it
  and 31.8 ms for dealing each level's narrow tasks to the group with the fewest
  blocks of their core types. Dealing everything is no answer either: on
  paged_attention it splits each block's four-task chain over the groups -- 24,576
  global ids a round against the stamp's 16,128 -- and measured 11.7 % slower than
  the stamp, -38.4 % against M0 where the stamp measures -44.9 %. The plan now deals
  only when that spreads the work better: fewer waves of a group's cores per level
  and core type, or as few and fewer blocks in the busiest group. The plan's counts
  put qwen at 1,001 waves dealt against 1,281 stamped, deepseek_v4 at 3,297 either
  way with a lighter busiest group dealt, and paged_attention level on both, so it
  keeps the stamp there. Rules that kept a producer's consumer with it while balancing
  load -- by a slack on the group's load, by the level's share, or for its first
  consumer only -- each lost on qwen or deepseek_v4 (33.1 to 40.7 ms in the oracle
  for qwen) or crossed more groups on paged_attention.
- **The cohort on the managers' path.** A sync-start task kept the ordinary path:
  a manager claims its blocks, hands them over held, and the managers release them
  once every share is placed. Its first run lost six members, refused by a ring full
  of partition work and dropped, and the cohort never assembled; members now wait in
  the hold array. Once running, every cohort was seen finished ~580 us after its
  release, against a 270 us kernel: a manager learns of a finish through its queue's
  prefix and the 32 positions past it, and the cohort's positions sit behind a
  partition of pieces posted ahead. Its consumers, posted only once the managers saw
  it finish, waited ~810 us a partition.
- **The barrier.** An entry with a producer in an earlier partition named none of
  it and waited for its group's watermark to pass the subpartition before. With the
  cohort the last piece of its partition, each layer's first tasks waited for the
  whole layer before: 360 barrier waits per controller a round, ~0.82 ms each. Naming
  a producer in the partition before as any other -- its piece, or the proxy at the
  end of its own subpartition -- and leaving one further back to the order partitions
  are posted in took the window from 34.0 to 32.5 ms.
- **The cohort at the end of its partition.** Holding a cohort until the rest of
  its partition has finished, so it drains cores already idle, reads 33.2 ms in the
  oracle against 31.8 ms for starting it once its producers have: the layer's head
  then waits for the cohort instead of filling the tail.

## Why not

Each dropped part cost qwen more than what replaced it, and what replaced them
costs the cases measured before nothing: deepseek_v4 and paged_attention both run
ahead of the build before
([validation/partitions.md](../asim/validation/partitions.md)). What remains between
the partition arm and M0 is the cohort's
drain: its members hold every cube core until the last frees, ~1.14 ms a core a
round against ~1.0 ms in the oracle, and the oracle's own floor for the design,
31.8 ms, is within 1 % of M0's scheduling window. A pipelined entry behind running
work cannot be taken back to shorten the drain: its finish is announced when it is
placed.

## When to reconsider

- The plan keeps the stamp only where it spreads the work as well. A graph whose
  cross-group edges cost more than the balance dealing buys may want the stamp even
  when it spreads the work worse, which the waves cannot see: paged_attention is
  bound by the hand-over, and dealt, its managers spent 32 % longer posting.
- The barrier holds no fewer local ids than naming does, so it would return only
  with a record format that cannot name an id below its subpartition.
- A cohort's drain shrinks only if partition work is kept off pipeline slots while
  one assembles, which the controller cannot foresee, or if the compiler places the
  cohort where cores idle anyway.

## References

- [docs/asim/partitions.md](../asim/partitions.md) -- the design as it stands.
- [docs/asim/validation/partitions.md](../asim/validation/partitions.md) -- the
  measurements on qwen, deepseek_v4 and paged_attention.
- [docs/asim/analyzers/asim_local_oracle.py](../asim/analyzers/asim_local_oracle.py)
  -- `PLACE`, `COHORT`, `COHORT_AT` and `HOLD` model the alternatives above.
