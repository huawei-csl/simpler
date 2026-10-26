# More tasks a layer for the partition design: tiles, and the cheaper way

**Date**: 2026-10-21
**Verdict**: the partition design gains from *independently schedulable* tasks, not
from task count as such. Un-batching the gate/up dispatches buys 2.0 points
(+6.8 % -> +4.8 % against M0) for 1.18x the tasks, identical device work, 0.25 % more
silicon time and *less* look-ahead pressure -- the lever to keep. Finer MLP and down
tiles buy 2.8 points for 2.6x the tasks, 4.7 % more cube work and a saturated 512-id
look-ahead. Un-batching out_proj as well buys nothing measured (+5.0 %) although the
oracle's floor falls to 1.35 %: its 24 blocks all wait on the attention cohort and
all feed the residual, so there is nothing to stagger

## Question

qwen3-14B decode is cube-bound, and
[the first partitions investigation](2026-10-gq-partitions-qwen.md) closed with the
design at its own floor: the oracle put it within 1 % of M0's scheduling window
because 277 tasks a layer leave the AICPU off the critical path whatever schedules
it. Tasks per layer are fixed at codegen by pypto-lib's tiling constants, not by any
run-time knob. Does a finer graph -- more, smaller tasks a layer -- give the design
room it can collect?

## What was changed

`MLP_TN` and `DOWN_TN` from 1024 to 512 in pypto-lib's `models/qwen3/14b/decode_fwd.py`.
Both tile dims must divide `gcd(HIDDEN, INTERMEDIATE) = gcd(5120, 17408) = 1024`, and
an assert ties them to each other for the silu/down K-slice alignment, so 512 is the
only step below the upstream value. That gives `MLP_ON` 34, `DOWN_ON` 10 and
`K_SPLITS` 34, and **729 tasks a layer against 277** -- 29,165 against 11,085 over 40
layers, 2.63x.

One structural change comes with it. `dcr_xgamma` is a single `pl.spmd(DOWN_ON)`
dispatch naming every down_proj task, `DOWN_ON * K_SPLITS` of them: 85 at the
upstream tiles, 340 at 512, against a host_build_graph inline fanin cap of
`CHIP_MAX_FANIN = 128` with no spill pool. One `pl.system.task_dummy` funnel a down
block, each carrying that block's `K_SPLITS` tids, puts the dispatch back on 10
named producers and adds 10 tasks a layer. It is the shape the file already uses for
the deferred gate/up tiles. With it the widest dep array in the generated
orchestration is 50.

The regeneration passes the torch golden on silicon at both tilings.

## The baseline had to be re-measured

The published figures for this case (M0 33.25 ms, partitions 33.95 ms, +2.1 %) were
measured with a working calibration carrying no line for func_id 12,
`paged_attention_rope_cce_aiv` -- the AIV half of the 278 us attention kernel, which
the runtime then charges the 1 us default. That is 534 ms of vector core time a
round, 11.1 ms an AIV core, absent from the model. The campaign disclosed the
fallback but not its size.

So both tilings were regenerated from the same pypto, calibrated from their own
silicon chip-swimlane capture with every func_id covered (`M2_DONE` reports
`default=0` in every arm), and run in one interleaved job. The coarse arm on that
footing is **not** the published +2.1 %.

## What it measured

aSim, three arms interleaved in every job -- M0, plain M2, partitions -- eight
rounds a rep with the first two dropped, three reps (n=18 rounds an arm), each
tiling with its own calibration. The window is each thread's scheduler window less
that thread's own simulator ledger, the latest of the four.

| tiling | arm | device_wall | vs M0 | window | vs M0 |
| ------ | --- | ----------- | ----- | ------ | ----- |
| 277 a layer | M0 | 36.18 ms | | 35.02 ms | |
| | M2 | 37.21 ms | +2.8 % | 35.64 ms | +1.8 % |
| | partitions | 38.65 ms | +6.8 % | 37.19 ms | +6.2 % |
| 729 a layer | M0 | 38.31 ms | | 36.64 ms | |
| | M2 | 39.64 ms | +3.5 % | 37.27 ms | +1.7 % |
| | partitions | 39.85 ms | **+4.0 %** | 37.89 ms | **+3.4 %** |

Paired rep by rep, partitions measure +6.79 % on the coarse graph (+6.91, +6.75,
+6.71) and +4.06 % on the fine one (+4.13, +3.97, +4.07). **The finer graph is worth
2.8 points to the design**, on `device_wall` and on the window alike.

The oracle agrees on the direction and says the floor moves further than the
measurement does. With `PLACE=pick LOOK=512 COHORT=sync` and the tiling's own
`PART`, it reads 36.32 ms on the coarse graph -- 3.7 % above M0's window -- and
37.02 ms on the fine one, 1.0 % above. The implementation leaves the same 0.86 ms
on the table against its own model either way.

## Why the design gains while the case gets slower

Finer tiles cost real time. On silicon, same codegen, `device_wall` goes 36.21 ms to
38.62 ms (+6.7 %) and cube core time 780.6 ms to 817.1 ms a round (+4.7 %):

| per round | 277 a layer | 729 a layer |
| --------- | ----------- | ----------- |
| cube busy, hottest core | 33.74 ms | 35.22 ms |
| makespan | 35.68 ms | 38.73 ms |
| slack above the cube floor | 1.94 ms | 3.51 ms |

Two stages pay for it. `MLP_ON` doubling moves the MLP off the batched
`pl.spmd(GATE_UP_SPMD_N)` path and onto the deferred per-tile one -- the spmd
dispatches cover 6 of 34 output blocks instead of 6 of 17, so the deferred share
goes from 65 % to 82 % of the columns, and deferred gate/up core time rises from
180.4 ms to 245.8 ms. down_proj takes 4x the tasks for a task only 3.55x cheaper
(38.16 us to 10.75 us), +16.5 ms. Attention and QKV are untouched.

The design gains anyway because **the extra cube work costs M0 more than it costs the
partition arm**: M0 goes up 2.13 ms between the two graphs, the partition arm 1.20 ms.
M0 was already within 5 % of the cube floor and has nothing to absorb the extra work
with; the partition arm has slack, and the slack above the cube floor grew 81 %.

That is in spite of the design's own overheads growing, per controller a round:

| counter | 277 a layer | 729 a layer |
| ------- | ----------- | ----------- |
| entries | 4,801 | 9,272 |
| map peak | 192 | 708 |
| store peak (of 1,024) | 275 | 499 |
| releases learned late | 2.4, 0.01 ms | 280, 2.27 ms |
| waits for producers past the fourth | 25.8, 1.64 ms | 51.6, 4.19 ms |
| look-ahead waits | 0 | 2.4, 0.81 ms |

Splitting a 4x finer MLP over the same four groups is what multiplies the late
releases: every one of the 280 is a range completed in another group
(`late_peer=2238` of `late=2239` over eight rounds), where the coarse graph's
silu-to-down edges mostly stayed inside a group.

## The same gain, almost free: un-batching gate/up

The tile change confounds two things -- more tasks, and more cube work. `GATE_UP_SPMD_N`
separates them. It splits the gate/up tiles between a batched `pl.spmd(GATE_UP_SPMD_N)`
dispatch carrying the leading `n_out` blocks and a deferred `pl.at` task per remaining
tile; at 6 the spmd covers 6 of 17 output blocks. Lowering it to 1 moves five blocks a
k-split off the batched path onto the deferred one. **Every tile shape, every block and
every kernel is unchanged** -- only how many of them the managers see separately.

277 tasks a layer becomes 327, 1.18x, and the silicon chip swimlane records the same
19,208 AICore tasks either way, with 772.5 ms of cube core time against 780.6 ms
(-1.0 %) and `device_wall` 36.30 ms against 36.21 ms (+0.25 %). The regeneration
passes the torch golden on silicon.

| graph | tasks a layer | AICore tasks | partitions vs M0 | window vs M0 | oracle floor vs M0 window | window - oracle | silicon device_wall |
| ----- | ------------- | ------------ | ---------------- | ------------ | ------------------------- | --------------- | ------------------- |
| `GATE_UP_SPMD_N=6`, 1024 tiles | 277 | 19,208 | +6.8 % | +6.2 % | +3.7 % | 0.86 ms | 36.21 ms |
| **`GATE_UP_SPMD_N=1`, 1024 tiles** | **327** | **19,208** | **+4.8 %** | **+4.3 %** | **+2.6 %** | **0.61 ms** | **36.30 ms** |
| `GATE_UP_SPMD_N=1` + `N_OUT_DIRECT_BLOCKS=1` | 350 | 19,208 | +5.0 % | +4.4 % | +1.35 % | 1.07 ms | 36.65 ms |
| `GATE_UP_SPMD_N=6`, 512 tiles | 729 | 37,088 | +4.0 % | +3.4 % | +1.0 % | 0.87 ms | 38.62 ms |

Paired rep by rep, un-batching measures +4.79 % against M0 (+4.78, +5.01, +4.59)
where the same graph batched measures +6.79 %.

**It costs the design nothing to carry.** Per controller a round the counters are flat
or better than the batched graph's: 4,801 entries either way, store peak 275, map peak
201 against 192, releases learned late 1.5 against 2.4, waits for producers past the
fourth 25.1 against 25.8. The oracle's `la_need` *falls*, 183-187 against 239-242 --
the opposite of the finer tiles, which saturate it. A batched dispatch is one
manager-visible task: its blocks cannot be placed in different groups, and none of them
can start before the dispatch does. Un-batching hands the plan that freedom without
adding an edge, a global id or a byte of plan.

## Un-batching does not keep paying: out_proj is a synchronized wave

`N_OUT_DIRECT_BLOCKS` batches 24 of out_proj's 50 `(n_split, k_split)` tiles into one
dispatch and leaves 26 direct. Set to 1 it leaves 49 direct, taking the graph to 350
tasks a layer on top of `GATE_UP_SPMD_N=1` -- again at the same 19,208 AICore tasks,
and the golden passes on silicon.

**The oracle says this should help, and the implementation does not deliver it.** The
floor falls again, from 2.6 % to 1.35 % above M0's window -- the best of any graph
measured here bar the finer tiles -- while the measured arm goes the other way, 4.8 %
to 5.0 % against M0 (paired +4.91 %: +5.07, +5.03, +4.62, against +4.79 % for
`GATE_UP_SPMD_N=1` alone, ranges that overlap almost entirely). What grows is the
distance between the two: 1.07 ms against 0.61 ms, the largest of the four graphs.

The two batches are not alike, and that is the lesson. The gate/up tiles
`GATE_UP_SPMD_N` releases have five independent k-splits and their own per-k dummy
funnels: they become ready at different times, and the plan can put them in different
groups and start them apart. out_proj's 24 blocks all name the attention cohort as
their producer and all feed the same residual, so they become ready in the same
instant and must all finish before the next stage -- un-batching gives the plan 23
more entries to post over a wave it cannot stagger. The managers duly pay for them:
map peak 201 to 224, waits for producers past the fourth 25.1 to 27.8 a round,
posting 678.8 to 691.3 us and submitting 434.5 to 448.1 us, with silicon
`device_wall` up 0.35 ms from the extra per-task core overhead.

**So the rule is not "more tasks".** It is *more independently schedulable tasks*: a
batch worth splitting is one whose members can start at different times or run in
different groups. That also explains why the remaining batches are not worth touching
-- see below.

## Why 512 is the end of this road

The oracle's `la_need` -- the local-id distance a group must see ahead to start its
next entry -- is 239 to 242 ids on the coarse graph and **512, 511, 512, 512** on the
fine one, against a look-ahead fixed at 512. It is exactly saturated, and the
look-ahead waits that appear in both the oracle (76 waits, 47.1 ms) and the measured
arm (2.4 a round) are that limit binding for the first time.

A further halving is therefore not available: 256-wide tiles would put ~1,500 tasks
a layer and a correspondingly larger span against the same 512 ids, and the
`dcr_xgamma` fanin would need a second funnel tier. The look-ahead is a deliberate
part of the design, fixed at 512 by its author, so the finer graph is the last one
this design reads without changing that.

## When to reconsider

- The three overheads that grew are where the remaining 4 % sits, and they are
  addressable without touching the look-ahead: 2.27 ms of late releases, 4.19 ms of
  threshold waits, 0.81 ms of look-ahead waits a controller a round, against a 1.4 ms
  gap to M0. Keeping a block's silu and its 34 down_proj consumers in one group --
  component placement rather than per-level dealing -- would remove most of the
  cross-group edges the fine graph added, the same idea left untested for
  paged_attention's batch elements.
- The batches that remain cannot be split, or should not be. attention is
  `pl.spmd(..., sync_start=True)`: its 24 blocks must start together, which is what
  makes it the cohort, so un-batching it is a semantic change and not a granularity
  one. `dcr_xgamma` un-batched is a measured regression recorded in pypto-lib's own
  source -- the separate-task form WAW-serialized the sliced writers of `out` at
  ~35 us against ~3 us. copy_hidden and copy_out run once a round, not once a layer.
  `ROPE_CORES` is dead at this revision, rope being fused into the attention op. The
  earlier estimate of "~100 more tasks a layer available" counted all of these and
  was wrong; out_proj's 23 were the only ones, and they did not pay.
- `GATE_UP_SPMD_N=1` on the finer tiles is untried: `MLP_ON` is 34 there, so it would
  add 50 tasks a layer to the fine graph's 729 -- but that graph has no look-ahead
  left to spend.
- qwen stays cube-bound at either tiling. A case whose cube floor is not within 5 %
  of M0's window is where this design has room to win outright, which is what
  deepseek_v4 (-9.8 %) and paged_attention (-45.6 %) already show.

## References

- [docs/asim/partitions.md](../asim/partitions.md) -- the design.
- [docs/asim/validation/partitions.md](../asim/validation/partitions.md) -- the
  published measurements, whose qwen row uses the calibration without func_id 12.
- [2026-10-gq-partitions-qwen.md](2026-10-gq-partitions-qwen.md) -- the design changes
  that brought this case from +51 % to its floor.
- [docs/asim/calibration.md](../asim/calibration.md) -- calibration is per
  (case, func_id); a missing line is charged the 1 us default.
