# Why qwen loses under GroupQueue partitions, and why the plan is not where to fix it

## Question

qwen3-14B decode, 40 layers, is the one case in the suite where the partition
design loses to M0. deepseek_v4 gains 9.8 % and paged_attention 45.6 %; qwen is
6.8 % behind. A loss of 0 % would be acceptable on a cube-bound case — 6.8 %
needs an explanation, and the explanation has to name something that can be
changed.

Three mechanisms were plausible going in: the per-layer sync-start cohort
draining the cores before it runs, core confinement (24 cube cores split 6 to a
group, so no group can borrow a neighbour's idle cores), and threshold waits for
producers past the fourth a record can name. The first two are wrong.

## What the figure actually is

The published +2.1 % is not the number to explain. It was measured with
`asim_calib/v3_qwen.calib`, which has no line for func_id 12 — the attention AIV
kernel, 278 µs, charged the 1 µs default. Re-measured on `v4_qwen_coarse.calib`,
the same build and the same graph read:

| arm | device_wall | vs M0 | window | vs M0 |
| --- | ----------- | ----- | ------ | ----- |
| M0 | 36.185 ms | | 34.997 ms | |
| GroupQueue, no partitions | 37.203 ms | +2.8 % | 35.620 ms | +1.8 % |
| GroupQueue partitions | 38.673 ms | +6.9 % | 37.215 ms | +6.3 % |

n = 18 rounds an arm, three paired reps, arms interleaved in one lock.

Two things bound the question. A third of the loss — 0.62 ms of window — is the
GroupQueue transport itself, which the un-partitioned arm pays too and which has
nothing to do with partitions. And the case has almost no headroom: on silicon
the hottest cube core is busy 33.74 ms of a 35.68 ms makespan, so M0 is already
within 3.8 % of a floor no scheduler can beat.

## What the oracle says costs the rest

[`asim_local_oracle.py`](../asim/analyzers/asim_local_oracle.py) models the
design at the simulator's hops. Run on the same graph and calibration
(`SEEDS=10 PLACE=pick LOOK=512 COHORT=sync PART=277`) it reads 36.329 ms, and
each knob turned off one at a time:

| ablation | makespan | Δ |
| -------- | -------- | - |
| the design as built | 36.329 ms | |
| `NAMED=64` — a record may name every producer | 35.185 ms | **−1.144** |
| `LOOK=inf` — no look-ahead limit | 36.329 ms | 0.000 |
| `RING=0` — a free ring exchange | 36.327 ms | −0.002 |
| `ENTER_US=0` — free subpartition entry | 36.328 ms | −0.001 |
| `GATE=0` — every subpartition resident at once | 36.327 ms | −0.002 |
| `COHORT=spread` — no managed cohort | 36.486 ms | **+0.157** |
| `PART=∞` — one partition, no boundaries | 37.317 ms | **+0.988** |

**The cohort drain and core confinement are not the cause.** Running the
attention cohort on the managers' path is *better* than treating it as an
ordinary spread supertask, and removing partition boundaries altogether is
0.99 ms *worse*. The ring, the entry cost, the residency gate and the 512-id
look-ahead are all exactly zero on this graph.

With unlimited comparators the oracle puts the design at 35.185 ms — 0.17 ms
above M0's measured window. **The design itself is free on qwen; the
four-named-producer rule is what makes it cost.**

### It is one kernel

Attributing each threshold wait to the consumer that paid it (a patched copy of
the oracle, recording the consumer's kernel signature and its producer count):

| consumer | direct producers | share of threshold wait time |
| -------- | ---------------- | ---------------------------- |
| `silu` (func 33, 680 tasks a round) | 11 | 85 % |
| `dcr_xgamma` (35) | 87 | 3 % |
| `post_rms_reduce` (20) | 27 | 3 % |
| `residual_rms_cast` ×3 (15/16/17) | 10, 10, 7 | 5 % |

`silu`'s eleven producers are the five `gate_proj` k-splits, the five `up_proj`
k-splits and the `post_rms_reduce` — `K_SPLITS_MLP = 5`, doubled by gate and up,
plus the scale. A record names four and covers the rest with a watermark
threshold, so each of the 680 waits for its group to retire **every local id
below the fifth-highest producer**, not just its own producers. Exempting
func 33 alone is worth 0.60 ms of the 1.14; the five smaller fan-ins are the
rest.

A funnel of dependency-only tasks does **not** help: both the oracle and
`partition_plan.cpp` resolve producers *through* them, which is why the
finer-tile graph still shows `dcr_xgamma` with 340 producers despite the funnel
added there for `CHIP_MAX_FANIN`. Making a funnel count would be a plan change,
not a hardware one.

### Two priced levers

Raising the comparator count is hardware: `NAMED` 4 → 8 recovers 1.04 ms of the
1.14, 4 → 10 recovers all of it, and the curve is flat past 10. The other lever
is free. `PLACE=list` — a list schedule placing each narrow task where it would
finish earliest — reads 35.610 ms against `pick`'s 36.329, and holds across all
four qwen graphs (coarse −0.72, `GATE_UP_SPMD_N=1` −0.78, out_proj un-batched
−0.39, 512-tiles −0.71 ms) at the *same* threshold-wait count. At `NAMED=64` it
is 0.05 ms *worse* than `pick`, so its whole advantage is threshold damage.

## The list placement, built and measured: dropped

The oracle's prediction was worth testing, so the plan builder got it: a list
schedule over each group's cores, a task placed where it finishes earliest given
when its producers finish, the hop one run in another group costs, and when that
group's cores come free.

It needs an estimate of each kernel's duration, and that is not a detail — with
unit durations the same schedule reads 38.222 ms, far *worse* than what the plan
does today. The build read the run's own calibration as the compiler's
stand-in, so the measurement below is the upper bound a perfect cost model would
give.

The plan it produces is leaner in every dimension:

| plan statistic | as built | list placement |
| -------------- | -------- | -------------- |
| plan size | 241,421 words | 234,170 (−3.0 %) |
| subpartition span, max / mean | 307 / 194 | 287 / 179 |
| map, max per subpartition | 144 | 125 (−13 %) |
| global ids | 8,604 | 7,721 (−10 %) |

And on the device it does what it was built to do, consistently across three
reps — per controller a round, threshold wait time 11,623 → 8,524 µs (−27 %),
`map_peak` 195 → 137, announcements sent 2,301 → 2,080, with core busy time
unchanged (aic 194.8 against 192.4 ms a thread, so nothing was traded away).

**It is worth 0.52 points of the predicted 2.0.** Paired rep by rep against the
as-built plan: +0.03, −1.08, −0.51, mean −0.52 pp — 38.449 ms against 38.673
(window 37.085 against 37.215), with one rep a wash.

| arm | device_wall | vs M0 | window | vs M0 |
| --- | ----------- | ----- | ------ | ----- |
| GroupQueue partitions, as built | 38.673 ms | +6.9 % | 37.215 ms | +6.3 % |
| GroupQueue partitions, list placement | 38.449 ms | +6.3 % | 37.085 ms | +6.0 % |

Dropped: a quarter of the predicted gain, a rep spread that straddles zero, and
a dependency on a cost model the compiler would have to supply on silicon.

## What this changes about the diagnosis

A third of the threshold waiting disappeared and the makespan moved 0.2 ms, so
**on the device those waits are overwhelmingly off the critical path** — they
overlap with controller work that is itself what binds the round. The oracle's
ladder is right about the design and wrong about where the device's time goes,
because it models hops and not the controllers' own cost: posting (668 µs a
controller a round), submitting (410 µs), installing maps, and the announcement
traffic.

So the loss decomposes as 0.62 ms GroupQueue transport, ~0.2 ms the comparator
limit as it actually bites, and the remaining ~1.4 ms the implementation's own
cost above the design's model. **The next lever is controller throughput, not
the plan.**

## When to reconsider

- If controller throughput improves enough that the arm becomes
  dependency-bound, re-run the list placement — its predicted 0.72 ms is still
  in the oracle, waiting for the waits to reach the critical path.
- If the comparator count is ever revisited as hardware, 8 is where the knee is
  on qwen, and `silu` is the shape that sets it: a reduction joining 2 × k-split
  partials plus a scale.
- If the plan stops resolving producers through dependency-only tasks, a funnel
  under `silu` becomes a zero-hardware way to reach the same place.

## References

- [docs/asim/validation/partitions.md](../asim/validation/partitions.md) — the
  three cases' measurements
- [2026-10-gq-partitions-qwen-finer-tiles.md](2026-10-gq-partitions-qwen-finer-tiles.md)
  — the task-count levers on the same case, and the calibration correction
- [2026-10-gq-partitions-qwen.md](2026-10-gq-partitions-qwen.md) — the four
  parts of the first plan that cost qwen most
- `docs/asim/analyzers/asim_local_oracle.py` — `PLACE`, `NAMED`, `LOOK`,
  `COHORT`, `GATE`, `RING`, `ENTER_US`, `PART`
