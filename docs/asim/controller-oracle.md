# What a different controller would buy: the dataflow oracle

deepseek_v4 is the case the GroupQueue loses (+4.7 % against M0,
[validation/deepseek-v4.md](validation/deepseek-v4.md)). Before changing the
controller, this asks how much any controller could win there, and which change
would capture it.

## The oracle

[`analyzers/asim_oracle.py`](analyzers/asim_oracle.py) list-schedules one round's
task graph on the simulator's own core model: 24 clusters of one cube and two
vector cores, a mix block taking a whole cluster, and per-kernel compute drawn
from the calibration as aSim draws it. A task is released when its last producer
finishes plus a hop cost, and its blocks take the first free cores of their type.
No manager stands on any hop. The manager still has to hand the fabric every task
once, so tasks enter in submission order at a fixed rate.

The GroupQueue's structure can be switched on: four controllers of six clusters
(cluster `c` belongs to controller `c % 4`, as in M2), and a task confined to the
controller its last producer finished on unless its blocks may span controllers. A
block may start on a controller `H_IN` after a producer that finished there and
`H_X` after one that finished anywhere else, so a block free to span weighs a
nearby core against an earlier one. A task whose blocks ran on more than one
controller finishes in none, so every consumer pays `H_X`.

Inputs are the round's `deps.json` and a swimlane naming the tasks the simulator
actually ran -- dispatch predicates retire the rest inline. The numbers below are
the mean of three draws.

## It agrees with the simulator where it can be checked

| check | oracle | aSim |
| ----- | ------ | ---- |
| paged_attention Case1, hand-over at 6 tasks/us | 11.04 ms | M2 11.50 ms |
| deepseek_v4, whole chip, every hop 3.7 us (silicon's median hop gap) | 30.66 ms | M0 window 30.98 ms |
| deepseek_v4, four controllers, wide tasks confined, every hop 3.7 us | 55.62 ms | one-group probe: M2 window 57.88 ms |
| core time a round, deepseek_v4 | 348.1 ms | 348.7 ms |

paged_attention calibrates the hand-over rate: it is entirely submission-bound,
so its makespan is the rate, and 6 tasks/us reproduces M2. deepseek_v4 is not
submission-bound at any rate from 4 to 6 tasks/us, so the rate does not move its
numbers.

## deepseek_v4: what each change is worth

Hand-over at 6 tasks/us, in-controller hop 0.45 us (the link plus the core's
startup):

| design | makespan | vs today |
| ------ | -------- | -------- |
| today's hop on the whole chip (~ M0) | 30.66 ms | |
| 4 controllers, wide tasks confined (either kind of cross hop) | 51.61 ms | +68 % |
| wide tasks span controllers, cross hops via the manager | 28.73 ms | -6.3 % |
| wide tasks span, cross hops in hardware at 0.8 us | 26.34 ms | **-14.1 %** |
| ideal fabric: whole chip, every hop 0.45 us | 26.05 ms | -15.0 % |

**Both changes are needed.** Confinement is the dominant term: 330 of the
critical path's tasks are 16-64 blocks wide, and one controller's six clusters
run them in up to six times the waves, so a hardware cross hop alone changes
nothing. Letting a wide task span controllers removes that, but its consumers
then wait for a completion no single controller sees, so most of the hop saving
needs a chip-wide completion signal as well. With both, the GroupQueue gets within
1.1 % of the ideal fabric.

**The cross-controller hop has a budget.** With both changes, a cross hop of
0.45, 0.8, 1.5 and 2.5 us gives 26.05, 26.34, 26.89 and 27.73 ms: under ~1.5 us
it keeps most of the win (-12 % or better), and by 2.5 us it is down to -10 %.

**Four comparators are enough; leaving predicates with the manager is not
free.** With both changes, a held task that may name at most four unfinished
producers -- the manager resolving the rest -- gives 26.44 ms (-13.8 %). A task
its dispatch predicate retires costs nothing above; if the manager still
evaluates every predicate, each path through one pays a manager hop, and the win
falls to 27.01 ms (-11.9 %), or 27.14 ms (-11.5 %) with four comparators as well.

**paged_attention is indifferent.** Every variant lands at 11.0-11.3 ms: it is
bound by how fast the manager hands tasks over, which none of these changes touch.

## Limits

- Placement is greedy with perfect knowledge of every core, and blocks of one task
  start in parallel; a real controller pays something for both.
- Every task is handed over in submission order and held without limit. That order
  is costly to honour: a task submitted late but needed early forces everything
  before it in, so on deepseek_v4 the fabric would hold ~3,000 tasks (~23,000
  blocks) at once. Fed one dependency level ahead -- a task enters when its last
  producer starts -- it holds at most 13 tasks (99 blocks) waiting, and at most 7
  of a task's producers are unfinished when it enters. Both figures come from the
  start and end times of one draw of the both-changes schedule.
- A task's controller is where its last producer finished. A different affinity
  rule changes how many hops cross controllers, and so the value of the second
  change.
- The rank asymmetry silicon shows is lost here as in aSim: both ranks run the
  same tasks, because the model computes no dispatch predicate.
- qwen has not been run: no expanded `deps.json` of it is on disk.
- The hops above are below the simulator's own: a hop inside a controller there is
  1.02 us -- most of it the core's entry poll and startup, which every arm pays --
  and 1.33 us across one with a 310 ns link. At those hops the ideal fabric is
  -12.5 %. [silos.md](silos.md) and [partitions.md](partitions.md) measure the designs
  built on this.

## Reproducing it

```bash
python docs/asim/analyzers/asim_oracle.py <deps.json> <chip_swimlane_records.json> <calib>
# DRAW=0 uses plain means, RATE the hand-over rate, VARIANTS a JSON list of
# [name, whole_chip, h_in_us, h_x_us, spread] with optionally the producers a
# held task can name (null = any) and the hop a predicate-retired task adds
```

deepseek_v4's `deps.json` comes from a dep-gen round of the expanded
orchestration and the swimlane from an M0 round (`--enable-chip-swimlane 1`);
paged_attention's the same way from `tests/st/a2a3/host_build_graph/paged_attention`.
