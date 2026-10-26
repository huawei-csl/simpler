# Partitions measured: qwen, deepseek_v4 and paged_attention

[partitions.md](../partitions.md) describes the design. Every number here is an
interleaved job: M0 and the partition arm -- and where named, a second M2 arm -- run
back to back in one lock, rounds after the second only, `device_wall` paired rep by
rep. The window is each thread's scheduler window less its own ledger, the latest of
the four.

| case | M0 | partitions | vs M0 |
| ---- | -- | ---------- | ----- |
| qwen3-14B decode, 40 layers | 36.18 ms | 38.65 ms | +6.8 % |
| deepseek_v4, as-is | 32.29 ms | 29.15 ms | -9.8 % |
| deepseek_v4, half the experts running | 39.21 ms | 35.94 ms | -8.4 % |
| paged_attention Case1 | 22.24 ms | 12.10 ms | -45.6 % |

## qwen3-14B decode: 6.8 % behind M0, on a cube-bound case

All 40 layers expanded -- 11,085 tasks, one sync-start cohort a layer -- with
`rt_declare_partitions(277)` as the entry's first statement: a layer a partition, 41
partitions. M0, plain M2 and partitions interleaved in every job, eight rounds a rep
with the first two dropped, three reps (n=18 rounds an arm):

| arm | device_wall | vs M0 | window | vs M0 |
| --- | ----------- | ----- | ------ | ----- |
| M0 | 36.18 ms | | 35.02 ms | |
| M2 | 37.21 ms | +2.8 % | 35.64 ms | +1.8 % |
| M2, partitions | 38.65 ms | +6.8 % (+6.71 to +6.91) | 37.19 ms | +6.2 % |

**These figures replace an earlier set** -- M0 33.25 ms, partitions 33.95 ms, +2.1 %
-- measured with a calibration carrying no line for func_id 12,
`paged_attention_rope_cce_aiv`: the AIV half of the 278 us attention kernel, which
the runtime then charges the 1 us default. That is 534 ms of vector core time a
round, 11.1 ms an AIV core, absent from the model, and it flattered every arm
unequally. The arms above are a regeneration of the same orchestration from pypto
`6beb9763`, each calibrated from its own silicon chip-swimlane capture with every
func_id covered: `M2_DONE` reports `default=0`, so no task fell back.

**The case is cube-bound.** On silicon a round carries 32.5 ms of cube work a core
and 11.3 ms of vector, in cube kernels of 18 to 278 us, so the AICPU is off the
critical path whatever schedules it: the hottest cube core is busy 33.74 ms of a
35.68 ms makespan, and M0's window is within 5 % of that floor. The oracle
(`PLACE=pick LOOK=512 COHORT=sync PART=277`) puts this design at 36.32 ms, 3.7 %
above M0's window, so the partition arm cannot draw level here; it measures 6.2 %
behind on the window, 0.86 ms above its own model.

| per controller and round | partitions |
| ------------------------ | ---------- |
| entries | 4,801 |
| store peak (of 1,024) | 275 |
| map peak | 192 |
| releases learned late | 2.4, 0.01 ms |
| waits for producers past the fourth | 25.8, 1.64 ms |
| look-ahead waits | 0 |

Every arm completes every task. The oracle's `la_need` -- the local-id distance a
group must see ahead to start its next entry -- is 239 to 242 ids against a
look-ahead of 512, so the window is not binding at this task count.

**What it took.** As first built the partitions ran qwen at ~50 ms: the host's
placement put 2.6 times as much in two groups as in the other two, a cohort member
refused by a full ring was dropped, the managers saw each cohort finish ~310 us
late, and every layer's first tasks waited at the barrier for the whole layer
before. [investigations](../../investigations/2026-10-gq-partitions-qwen.md) records
each step and what it measured.

**More tasks a layer helps, and the cheap way is to un-batch.** Lowering
`GATE_UP_SPMD_N` from 6 to 1 splits the batched gate/up dispatches into one task a
tile: 327 tasks a layer at byte-identical device work -- the same 19,208 AICore
tasks -- and the partition arm goes to 4.8 % behind M0 (window 4.3 %, oracle floor
2.6 % above M0's window) for 0.25 % more silicon time, with its counters flat and
`la_need` falling to 185. It is independent schedulability that pays, not the count:
un-batching out_proj as well (350 a layer) measures 5.0 %, no better, because its 24
blocks all wait on the attention cohort and all feed the residual. Halving the MLP and
down tiles instead reaches 729 tasks a layer and 4.0 % behind M0, but costs 4.7 % more
cube work and saturates the 512-id look-ahead:
[investigations](../../investigations/2026-10-gq-partitions-qwen-finer-tiles.md).

## deepseek_v4: -9.8 % against M0

**Measured.** The expanded orchestration with `rt_declare_partitions(372)` as its
first statement, on the build these pages describe and on the one before the qwen
work (`bba029704`), against M0, the three interleaved in every job, eight rounds a rep
with the first two dropped, two reps (n=12 rounds an arm):

| arm | as-is | vs M0 | half the experts running | vs M0 |
| --- | ----- | ----- | ------------------------ | ----- |
| M0 | 32.29 ms | | 39.21 ms | |
| before qwen | 29.75 ms | -7.76 % | 36.13 ms | -7.80 % |
| **now** | **29.15 ms** | **-9.80 %** | **35.94 ms** | **-8.40 %** |

Paired against the build before, -2.21 % as-is (range -2.33 to -2.08 %; window
-2.20 %) and -0.65 % with half the experts running (range -0.94 to -0.36 %; window
-0.77 %). Two changes
reach deepseek_v4, which has no sync-start task: the plan deals its narrow tasks
(3,297 level waves either way, a lighter busiest group dealt), and an entry names a
producer of the partition before rather than
waiting for its group to finish that partition, which held such an entry ~26 times a
round for 0.6 to 0.8 ms each. The busiest group's share drops from 9,450 entries to
9,332, against a mean of 9,282, the map's peak rises from 323 to 335 global ids, and
the releases learned late fall from 0.85 to 0.58 ms a round as-is.

### Against the shared watermark

On the build before qwen, the design against the one it replaced.

**What the oracle expects.** [`analyzers/asim_local_oracle.py`](../analyzers/asim_local_oracle.py)
models both designs in one machinery at the simulator's hops, one layer a partition,
no entry starting before the hand-over has reached it.
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

## paged_attention: -45.6 % against M0

**Measured.** The partitioned orchestration below, on the build these pages describe
and on the one before the qwen work, against M0, the three rotated in every job, 14
rounds a rep with the first two dropped, four reps (n=48 rounds an arm):

| arm | device_wall | vs M0 | window |
| --- | ----------- | ----- | ------ |
| M0 | 22.24 ms | | 18.96 ms |
| before qwen | 12.31 ms | -44.8 % | 9.65 ms |
| **now** | **12.10 ms** | **-45.6 %** | 9.37 ms |

Paired against the build before, -1.45 % (range -2.5 to -0.6 %; window -2.57 %).
paged_attention has no sync-start task and no edge between partitions, so none of the
cohort and cross-partition changes reach it, and the plan keeps the host's stamp: it
needs 5,888 level waves either way and puts 16,384 entries in every group. Dealing
instead splits each block's four-task chain over the groups -- 24,576 global ids a
round against 16,128, the map's peak from 126 to 192 -- and measured 13.68 ms, -38.4 %
against M0, in a job of its own with the stamp at 12.26 ms.

### Against declared groups

On the build before qwen.

The GroupQueue's paged_attention orchestration with its group declarations removed
and `rt_declare_partitions(1028)` as the entry's first statement -- four of its 256
disjoint batch elements a partition -- against M0, and against the same orchestration
declaring one group per batch element. Case1, the three interleaved, 14 rounds a rep
with the first two dropped, four reps (n=48 rounds an arm):

| arm | device_wall | vs M0 | window |
| --- | ----------- | ----- | ------ |
| M0 | 22.21 ms | | 18.96 ms |
| M2, declared groups | 11.14 ms | -49.8 % | 8.94 ms |
| M2, partitions | 12.26 ms | -44.8 % | 9.60 ms |

Partitions trail the groups by 9.8 % (7.7 to 11.0 % a rep). The plan has 64
partitions; a group's share spans 320 ids and maps 64 global ids. The host deals each
block's first task to the next group in turn, so a batch element's update chain
crosses groups at every block: 16,128 global ids a round, ~4,000 announced by each
controller.

**The oracle puts the two level.** With every release floored at its hand-over, it
reads 11.08 ms for partitions dealt by block and for a batch element per group alike,
and 13.08 ms for the shared watermark: the case is bound by the hand-over, at the 6
tasks/us calibrated on the groups' arm.

**What the partitions lose is the simulator's.** All of the ~674 releases per
controller per round learned late are ranges completed from another group, 7.5 ms of
lateness in all, and the partition model's status read costs the simulator 601 ns
against 287 ns for the groups' (a ledger of 13.2 against 5.2 ms a thread a round).
The core time issued is the same, 12.7 ms of cube and 10.1 ms of vector a thread.
Keeping each batch element in one group would remove the edges that cross: not
measured.
