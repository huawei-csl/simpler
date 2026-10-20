# GroupQueue partitions: predicated tasks left out of the partition, on the managers' path

**Date**: 2026-10-15
**Verdict**: dropped

## Question

Submitting to a controller should be committal: an entry it takes is an entry that
executes. One way to get there keeps predicated tasks out of the partitions
altogether. A predicated task takes no partition-local index; its manager reads the
predicate when the task is ready and dispatches it on the ordinary path, or finishes
it without a core; and a partition task naming one is held back until it has
finished, as for any producer the controllers do not run. The controller would then
need no predicate hardware at all.

## What was tried

`stamp_silo` (`src/common/host_build_graph/host/orchestrator.cpp`) returned
`TASK_SILO_NONE` for every predicated task even with partitions declared, so the task
takes no index -- `finish_partitions` passes levels through it -- and `part_feeds` is
false for it. Nothing else changed: the managers' existing hold path covers the
consumers. Measured on aSim beside M0, the design that posts predicated pieces up
front and answers them, and the design then adopted, which posts them decided
([shared watermark](2026-10-gq-shared-watermark-partitions.md)): deepseek_v4 expanded on two dies,
`rt_declare_partitions(372)`, eight rounds a rep with the first two dropped.

## Result

- **As the case runs on aSim**, where every predicate is false -- the cores compute
  nothing, so `recv_count` stays zero -- it measured 30.95 ms against M0's 32.30 ms,
  -4.2 % (window -5.7 %, n=12), behind the decided design's -7.0 % and the answered
  design's -6.0 % in the same two submissions. The pieces a controller takes per
  round fall from ~4,230 to ~2,170, but the 192 skipped expert supertasks of a layer
  pass through the managers before that layer's combine can be fed.
- **With half the experts made to run** -- a predicate target of -1 on even experts
  -- the case deadlocked in its first round. A manager learns of finishes only through
  its queue's status: the contiguous prefix of its positions and the 32 past it
  (`SimQueueStatus::ahead_bits`). In partition mode a manager posts pieces up to
  `SIM_AHEAD_HORIZON` (1,024) positions ahead, the next partition's included, gated
  until the switch. A predicated task that runs is dispatched on the ordinary path and
  takes a position behind hundreds of outstanding ones, so its finish is never
  reported, the pieces held on it never go, and the prefix never advances.

## Why not (now)

It is slower than both alternatives even when every predicate is false, and it is not
free of controller hardware once predicated tasks run: their completions need a path
outside the position window the partition pieces fill -- positions of their own with
their own status, or managers that learn partition completions from the partition
watermark rather than from positions.

## When to reconsider

If the managers learn partition completions without queue positions, so that the
ordinary path's positions stay within the status window, and for a graph whose
predicated tasks have few consumers: the cost is then one manager hop on every edge
out of a predicated task that runs, each link of a chain included.

## References

- [2026-10-gq-shared-watermark-partitions.md](2026-10-gq-shared-watermark-partitions.md)
  -- the design this was measured against, and its measurement.
- [docs/asim/partitions.md](../asim/partitions.md) -- the design that replaced it.
