# a5 hand-off latency, measured

Everything here was measured on one chip, **`Ascend950DT_9592`**, on both of its
device ids. Nothing was measured on `Ascend950PR` — the layout is probably the
same, but treat every number below as unverified there until someone runs the
tools on one.

Both figures are **corroboration, not the source**: every number in them is also
written out below, so this file can be read as plain text.

## The chip

![a5 hand-off latency: two dies, each carrying 4 AICPU cores in 2 clusters, 18
AICore clusters of 1 AIC + 2 AIV, and its own GM, with the arena drawn in die
0's GM and every measured round trip drawn as two legs meeting at that
arena](figures/a5-topology.png)

```text
die 0                                  die 1
  AICPU cluster 0: cpu0 (OS)  cpu1       AICPU cluster 2: cpu4  cpu5
  AICPU cluster 1: cpu2       cpu3       AICPU cluster 3: cpu6  cpu7
  L3 4 MB per AICPU cluster              L3 4 MB per AICPU cluster
  ----- GM -----------------------       ----- GM -----------------------
  18 AICore clusters, 1 AIC + 2 AIV      18 AICore clusters, 1 AIC + 2 AIV
    = 18 AIC + 36 AIV                      = 18 AIC + 36 AIV
    L2 + 3 KiB SSBUF per cluster           L2 + 3 KiB SSBUF per cluster

  GM is drawn between the two tiers because every AICPU<->AICore hand-off
  passes through it.  That is the data path; the physical floorplan is not
  established here.
```

**Two dies, and each one carries a full set of everything**: 4 AICPU cores (2
clusters of 2), 18 AICore clusters of 1 AIC + 2 AIV, and its own GM. It is not
an AICPU die pair plus an AICore die pair, and there is no memory pool sitting
between them.

`cpu0` is owned by the AICPU OS, so 7 cores are user-schedulable — 3 on die 0 and
4 on die 1. `cluster_id = phy_cpu_id / 2`, `die_id = phy_cpu_id / 4`.

## AICPU ↔ AICPU: two tiers, and the die boundary is free

![AICPU ping-pong matrix: 7x7 heatmap of round-trip latency, showing ~191 ns
inside a cluster and ~1113 ns for every other pair regardless of
die](figures/aicpu-pingpong-matrix.png)

```text
Round-trip floor over 6 sweeps, ns, device 0.  Upper triangle only: a round trip
contains both directions, so just a < b was measured.  cpu0 is OS-owned and not
measured.  cluster = cpu / 2,  die = cpu / 4.

 cpu |     2       3       4       5       6       7
-----+------------------------------------------------
   1 |  1113.0  1113.3  1327.5  1329.0  1128.1  1128.7
   2 |     -     190.7  1170.0  1169.1  1114.2  1114.1
   3 |             -    1170.0  1170.0  1114.1  1114.3
   4 |                     -     189.8  1114.2  1114.2
   5 |                             -    1114.2  1114.2
   6 |                                     -     190.3
```

| Pair | Round trip |
| ---- | ---------- |
| Inside a cluster | **~191 ns**, stable to 1% over 6 sweeps |
| Anything else | **~1113 ns**, about 6× |

There is no third tier for the die boundary. `cpu2,3 ↔ cpu6,7` crosses it and
costs 1114 ns; `cpu4,5 ↔ cpu6,7` stays inside die 1 and costs the same 1114 ns.

Two things in that matrix are *not* explained:

- **`cpu1` reads high everywhere.** It is the cluster-mate of the OS core.
- **Every cell above the 1113 floor involves cluster 2 (`cpu4,5`) paired with a
  die-0 cluster** — 1170 against cluster 1, 1328 against `cpu1`. We first put
  this down to DVFS, but the clock has since been measured directly and does not
  move (below), so the explanation is open.

## AICPU ↔ AICore: one floor plus two independent crossings

```text
round trip = 975 ns  + ~170 ns  if the AICPU  is off the home die   (cpu_term)
                     + ~225 ns  if the AICore is off the home die   (core_term)
```

The two terms are independent and additive, so with the arena homed on die 0:

| AICPU | AICore | Crossings | Round trip |
| ----- | ------ | --------- | ---------- |
| die 0 | die 0 | none | **975 ns** |
| die 1 | die 0 | AICPU off home | **~1145 ns** |
| die 0 | die 1 | AICore off home | **~1200 ns** |
| die 1 | die 1 | both | **~1370 ns** |

Each round trip is two legs meeting at the arena — AICPU to GM, GM to AICore —
and the cost is set by how many of those legs cross the boundary. The two
one-crossing cases are **not** the same price: the AICore leg is the dearer of
the two, so "one die boundary" is not a single number.

AIC and AIV are indistinguishable here — medians 1196.3 vs 1198.2 ns over 78k
round trips, same floor — so the cube/vector choice does not enter this path.

**"Home die" is a property of the GM allocation, not of the cores, the launch or
the clock.** It is drawn per `aclrtMalloc`: eight arenas allocated inside a
single launch split `0 0 0 1 1 1 0 0` across the two dies. This is why the same
binary looks like it favours die 0 in one launch and die 1 in the next — the
arena landed on the other side. **Never quote one die as the fast one.**

## The boundary costs writes, not reads

| Across the die boundary | Penalty |
| ----------------------- | ------- |
| An AICore GM load (`ld_dev` bypass, 86 ns) | **+0.16 ns** |
| A hand-off becoming visible | **+176 ns** |

Checked at every walked-region size from a single cache line to 8 MiB — twice
the per-cluster L3, so the reads genuinely reach GM — and flat every time, over
789 samples split evenly across the two dies. Loads really are served uniformly from either die; the entire die
penalty is the time for a store to become visible on the other side. A compute
kernel streaming a tensor does not care which die its data homes on. A scheduler
publishing signals does.

### What one GM access costs

Dependent pointer-chase over a random cycle, 256 steps, same 1 GHz counter. The
AICore reads with the `ld_dev` bypass; the AICPU reads through its cache.

| Walked region | AICPU ns/step | AICore ns/step |
| ------------- | ------------- | -------------- |
| 64 B (1 line) | 10 | 86 |
| 1 KiB | 89 | 86 |
| 16 KiB | 159 | 86 |
| 64 KiB | 193 | 86 |
| 512 KiB | 182 | 86 |
| 8 MiB | 194 | 86 |
| 16 MiB | 192 | — |

**AICore: 86 ns, flat across five orders of magnitude of region size**, because
the bypass makes every step a real GM access. About 9% of samples instead land
at 300–400 ns; those are first touches of a large region, they appear equally on
both dies, and they disappear once the arena is warm.

**AICPU: 10 ns when the line is L1-resident, plateauing at ~190 ns** by 64 KiB
and staying there out to 16 MiB — four times the 4 MB per-cluster L3, so ~190 ns
is its true cold-GM read.

Neither side's *store* visibility is measured; that is the term the die boundary
actually charges, and getting it needs one-way timestamps rather than round
trips.

Supporting measurement: the core clock is **identical on both dies (ratio
1.0000)** and does not move between launches, and `CNTFRQ_EL0` is a fixed 1 GHz,
so none of the above is a DVFS artefact and counting cycles would change nothing.

## AICore ↔ AICore

![AICore to AICore round trip by relation, against the AICPU to AICore path on
the same binary: 376 ns inside a cluster, 385 ns across a die, 730 ns across the
die boundary, versus 1114 and 1370 ns for AICPU to
AICore](figures/aicore-aicore.png)

24 slots over 8 clusters, every unordered pair, 3 launches × 4 arenas = 6624
round trips, zero failures.

| Pair | Round trip |
| ---- | ---------- |
| Same cluster (AIC↔AIV or AIV↔AIV) | **376 ns** |
| Same die, different clusters | **385 ns** |
| Across the die boundary | **730 ns** |

Three things worth keeping:

- **An AICore pair hands off 2.9× faster than an AICPU does** — 385 ns against
  the 1114 ns floor on the same binary. The AICPU is the slow party, not the
  fabric.
- **Sharing a cluster buys 9 ns, which is nothing.** The 1 AIC + 2 AIV of a
  cluster share a 3 KiB SSBUF and an L2, but this protocol hands off through GM,
  so that path is never on it. Measuring what SSBUF is worth needs a kernel that
  actually uses `__ssbuf__`; this result says only that cluster locality is not
  free money for a GM-mediated hand-off.
- **AIC and AIV are interchangeable** here: 378–386 ns same die, 727–731 ns
  across, whichever kinds the pair mixes.

The arena's home die still shows up — the cheaper same-die group alternates by
arena index and flips between launches, the same signature as everywhere else —
but it is worth only **~31 ns**, swamped by the ~347 ns cost of the two cores
being on opposite dies. So for an AICore pair, *where the cores are* matters an
order of magnitude more than *where the memory is*.

## Caveats

- **The floors are not portable between builds.** The same tool, rebuilt with a
  larger arena, moved the AICPU↔AICore floor from 975 to 1114 ns and the terms to
  +80 / +176. The *contrasts* held across both builds; the absolute floor did
  not. Quote the structure, not the constant.
- **`Ascend950PR` is unmeasured**, as is any a5 SKU other than `_9592`. CANN's a5
  ini files span `ai_core_cnt` from 8 to ~28 per die, so even the 18 is per-SKU.
- The AICPU↔AICPU matrix is a floor over 6 sweeps on device 0; the AICPU↔AICore
  numbers are medians over 16 arena sweeps across two launches.

## How it was measured

Two standalone CANN tools, each a host `.so` + AICPU `.so` + AICore `.o` launched
on ACL streams with no runtime dependency:

- `../aicpu-pingpong/` — AICPU core-to-core, atomic ping-pong over GM.
- `../aicpu-aicore-pingpong/` — AICPU↔AICore, raw GM hand-off. Launches
  `blockDim=36` so every one of the 108 AIC/AIV slots reports in, filters to one
  target per round, and shuffles the pair order so position in the schedule
  cannot be mistaken for position on the chip.

Both use bypass primitives on the AICore side (`ld_dev` / `st_dev` + `dsb`) and
volatile + `rmb()` / `wmb()` on the AICPU side, one writer per 64-byte line, and
time against the fixed 1 GHz system counter.
