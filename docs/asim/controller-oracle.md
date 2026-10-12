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
(cluster `c` belongs to controller `c % 4`, as in M2), a hop costing `H_IN` inside
a controller and `H_X` across one, and a task confined to the controller its last
producer finished on unless its blocks may span controllers. A task whose blocks
ran on more than one controller finishes in none, so every consumer pays `H_X`.

Inputs are the round's `deps.json` and a swimlane naming the tasks the simulator
actually ran -- dispatch predicates retire the rest inline. The numbers below are
the mean of three draws.

## It agrees with the simulator where it can be checked

| check | oracle | aSim |
| ----- | ------ | ---- |
| paged_attention Case1, hand-over at 6 tasks/us | 11.04 ms | M2 11.50 ms |
| deepseek_v4, whole chip, every hop 3.7 us (silicon's median hop gap) | 30.66 ms | M0 window 30.93 ms |
| deepseek_v4, four controllers, wide tasks confined, every hop 3.7 us | 55.62 ms | one-group probe: M2 window 57.38 ms |
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
| wide tasks span controllers, cross hops via the manager | 29.25 ms | -4.6 % |
| wide tasks span, cross hops in hardware at 0.8 us | 26.40 ms | **-13.9 %** |
| ideal fabric: whole chip, every hop 0.45 us | 26.05 ms | -15.0 % |

**Both changes are needed.** Confinement is the dominant term: 330 of the
critical path's tasks are 16-64 blocks wide, and one controller's six clusters
run them in up to six times the waves, so a hardware cross hop alone changes
nothing. Letting a wide task span controllers removes that, but its consumers
then wait for a completion no single controller sees, so most of the hop saving
needs a chip-wide completion signal as well. With both, the GroupQueue gets within
1.5 % of the ideal fabric.

**The cross-controller hop has a budget.** With both changes, a cross hop of
0.45, 0.8, 1.5 and 2.5 us gives 26.05, 26.40, 27.08 and 28.07 ms: under ~1.5 us
it keeps most of the win (-12 % or better), and by 2.5 us it is down to -8 %.

**paged_attention is indifferent.** Every variant lands at 11.0-11.3 ms: it is
bound by how fast the manager hands tasks over, which none of these changes touch.

## Limits

- Placement is greedy with perfect knowledge of every core, and blocks of one task
  start in parallel; a real controller pays something for both.
- A task's controller is where its last producer finished. A different affinity
  rule changes how many hops cross controllers, and so the value of the second
  change.
- The rank asymmetry silicon shows is lost here as in aSim: both ranks run the
  same tasks, because the model computes no dispatch predicate.
- qwen has not been run: no expanded `deps.json` of it is on disk.

## Reproducing it

```bash
python docs/asim/analyzers/asim_oracle.py <deps.json> <chip_swimlane_records.json> <calib>
# DRAW=0 uses plain means, RATE the hand-over rate, VARIANTS a JSON list of
# [name, whole_chip, h_in_us, h_x_us, spread]
```

deepseek_v4's `deps.json` comes from a dep-gen round of the expanded
orchestration and the swimlane from an M0 round (`--enable-chip-swimlane 1`);
paged_attention's the same way from `tests/st/a2a3/host_build_graph/paged_attention`.
