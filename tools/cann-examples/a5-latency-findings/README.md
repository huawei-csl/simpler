# a5 hand-off latency, measured

Everything here was measured on **`Ascend950DT_9592`**, on both of its device
ids, unless a section says otherwise. The AICore matrix and the AICPU↔AICPU
section have since been re-run on an **`Ascend950PR_957b`** (28 AICore clusters,
3 user AICPUs) and say so where the two parts differ; everything else is still
unverified on a PR part.

All four figures are **corroboration, not the source**: every number in them is
also written out below, so this file can be read as plain text.

## The chip

![a5 hand-off latency: two dies, each carrying 4 AICPU cores in 2 clusters and
18 AICore clusters of 1 AIC + 2 AIV, sitting over one GM that spans both, with
the arena homed on die 0 and every measured round trip drawn as two legs
meeting at that arena](figures/a5-topology.png)

```text
die 0                                  die 1
  AICPU cluster 0: cpu0 (OS)  cpu1       AICPU cluster 2: cpu4  cpu5
  AICPU cluster 1: cpu2       cpu3       AICPU cluster 3: cpu6  cpu7
  L3 4 MB per AICPU cluster              L3 4 MB per AICPU cluster

  ===== GM, 96 GiB, one pool spanning both dies =======================

  18 AICore clusters, 1 AIC + 2 AIV      18 AICore clusters, 1 AIC + 2 AIV
    = 18 AIC + 36 AIV                      = 18 AIC + 36 AIV
    L2 + 3 KiB SSBUF per cluster           L2 + 3 KiB SSBUF per cluster

  The die boundary runs between the cores, not through the memory.  GM is
  drawn between the two tiers because every hand-off passes through it:
  that is the data path, and the physical floorplan is not established
  here.
```

**Two dies, and each one carries a full set of cores**: 4 AICPU cores (2
clusters of 2) and 18 AICore clusters of 1 AIC + 2 AIV. It is not an AICPU die
pair plus an AICore die pair.

**The memory is not divided the same way.** The two dies share one GM — see
[Two dies, one GM](#two-dies-one-gm) for the measurement and the vendor-side
evidence.

`cpu0` is owned by the AICPU OS, so 7 cores are user-schedulable — 3 on die 0 and
4 on die 1. `cluster_id = phy_cpu_id / 2`, `die_id = phy_cpu_id / 4`.

**None of that structure is visible to software.** The AICPU OS publishes one
package (`physical_package_id = 0` for every core), no `cluster_id`, no
`die_id`, no SMT (each core is its own thread sibling), a last-level cache
shared by *all* cores (`L2u@0-7` on `_9592`, `L2u@0-3` on `_957b`), and **one
CPU NUMA node holding every core** — `node0` with ~1.5 GiB beside 8-10
**cpu-less** memory nodes that add up to the chip's HBM. So the 2-core grouping
below is a timing result, not a topology query, and no scheduler can read it
from the OS.

## AICPU ↔ AICPU: 192 ns inside a cluster, a lottery between them

![AICPU ping-pong matrix: 7x7 heatmap of round-trip latency floors, ~191 ns
inside a cluster and 1113 ns or more for every other
pair](figures/aicpu-pingpong-matrix.png)

```text
Round-trip FLOOR over 6 sweeps, ns, device 0.  Upper triangle only: a round trip
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

Read that as a floor and nothing more. Every off-diagonal cell is the best of 6
draws of a quantity that spans 500 ns, and the diagonal blocks are the only
numbers in it a single run can be expected to reproduce.

**Inside a cluster the hand-off is 192 ns and nothing moves it.** Over 204
counter addresses in 12 processes, `cpu2↔cpu3`, `cpu4↔cpu5` and `cpu6↔cpu7` each
span under 5 ns. Not the address, not the process, not the launch.

**Across clusters the cost is drawn per allocation.** One pair, measured on 64
counters 128 B apart within a single launch, costs anything from 1101 to
1627 ns:

| cluster pair | min | median | max | range |
| --- | ---: | ---: | ---: | ---: |
| c0–c1  `cpu1 ↔ cpu2,3` | 1101 | 1116 | 1627 | **526** |
| c0–c2  `cpu1 ↔ cpu4,5` | 1317 | 1366 | 1420 | 104 |
| c0–c3  `cpu1 ↔ cpu6,7` | 1115 | 1301 | 1562 | 447 |
| c1–c2  `cpu2,3 ↔ cpu4,5` | 1116 | 1301 | 1562 | 446 |
| c1–c3  `cpu2,3 ↔ cpu6,7` | 1110 | 1225 | 1327 | 217 |
| c2–c3  `cpu4,5 ↔ cpu6,7` | 1066 | 1322 | 1607 | 542 |
| any same-cluster pair | 192 | 193 | 197 | 5 |

ns, device 1, 204 samples per row = 12 processes × 16 aligned 512-byte blocks.

**The selector is the line's address, at 512-byte granularity.** Counters inside
one aligned 512 B block agree to 1.7 ns (median over every block measured); the
next block along differs by up to 334. The *virtual* address is not the
selector: the AICPU OS loads the SO at the same base in every process
(`0x3fff0032100` on the DT, `0x3fff80f2300` for the bank), yet the cost is
redrawn each process. What changes is the physical page the BSS lands on.

**One hidden variable reorders the whole matrix.** Over those 204 samples
`corr(c0–c1, c2–c3) = −0.92`. With c0–c1 at its floor, c2–c3 reads 1493 ns
median; with c0–c1 above 1400, c2–c3 reads 1116. A line that is cheap for one
cluster pair is expensive for the opposite one, which is what a coherence home
does: it sits near one pair of clusters and far from the other.

**It is not DVFS, and this is now measured on the AICPU too.** Each role times a
dependent-ALU chain against the 1 GHz counter inside the measured launch:
**800.1 µs ± 0.3 over every launch on both parts**, while the hand-off moves by a
third. The cores run at a fixed clock.

**c0–c2 is the one pair with a floor of its own.** `cpu1 ↔ cpu4,5` never drops
below 1317 ns at any address, and spans only 104 ns. Every other cross-cluster
pair reaches ~1115 on a good draw. So there is one genuine placement term in
this matrix, and it is the cluster-0-to-cluster-2 one.

**There is still no die tier.** `cpu2,3 ↔ cpu6,7` crosses the die boundary and
`cpu4,5 ↔ cpu6,7` does not; both floor at ~1115 and both span ~450 ns. The
boundary that matters to an AICPU hand-off is the cluster, not the die.

### `Ascend950PR_957b`: same law, fewer cores

3 user-schedulable AICPUs (`present` = 0-3, cpu0 OS-owned), so one cluster pair.
**Which die(s) those two clusters sit on is not established** — the AICPU side
of every measurement here resolves clusters, not dies (above), so `die =
phy_cpu_id / 4` is an untested extrapolation on a part with half the cores:

| | min | median | max | range |
| --- | ---: | ---: | ---: | ---: |
| `cpu2 ↔ cpu3` same cluster | 189.9 | 191.0 | 193.4 | 3.6 |
| `cpu1 ↔ cpu2,3` across | 1161 | 1315 | 1564 | 403 |

Same 512-byte granularity, same ~10 discrete levels, same fixed ALU probe. The
part has fewer cores and the same hand-off behaviour.

### What this corrects

An earlier revision read the floor matrix above as the cost — "**~1113 ns** for
anything not inside a cluster" — and listed two cells it could not explain:
`cpu1` reading high everywhere, and cluster 2 against a die-0 cluster reading
1170 instead of 1113. Both resolve:

- 1113 ns is the best draw, not the cost. The same pair pays 1101–1627 ns
  depending on where its counter landed.
- `cpu1` reading high against `cpu4,5` is the c0–c2 floor, which is real.
- The 1170 cells were one home draw, not a property of cluster 2. Over 204
  draws that pair's median is 1301.

The DVFS explanation that revision left open is closed: the clock does not move.

**What to do with it.** A cross-cluster AICPU handshake — a ticket lock, a shared
ready-queue index, a scheduler notification — pays a cost set by where its line
landed, with a 1.5× spread and no API that reports or chooses it. Allocate the
structure once and keep it; reallocating re-rolls the draw. Putting both ends in
one cluster removes the question altogether, at 192 ns.

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

Note, though, that the home die here was identified by *which configuration ran
faster*, and the AICore↔AICore matrix finds no home effect at all. See
[Two dies, one GM](#two-dies-one-gm) — the effect is real, its cause is not
established.

**`cpu_term` is keyed to the AICPU's cluster, not its die.** Re-reading the
per-arena sweep with the cpus kept separate, the four AICPU clusters take four
distinct values, and the split is not the die line — ns against the cpu-mean of
each arena:

```text
  cpu           1       2       3       4       5       6       7
  cluster      c0      c1      c1      c2      c2      c3      c3
  die          d0      d0      d0      d1      d1      d1      d1
  arena 0   +118.6   -10.1   -10.1   +15.6   +16.9   -64.6   -66.3
  arena 2    +98.2    -8.6   -17.2   +57.8   +65.1   -97.7   -97.6
  arena 7    +44.3   -18.6   -18.5   +55.4   +58.7   -59.9   -61.4
```

Cluster-mates always agree to a few ns. `cpu1` does **not** track `cpu2,3`
despite sharing die 0 with them, and in arena 0 it is further from them than
`cpu4,5` are from across the die boundary. So the +170 ns figure is the span of
a four-level per-cluster term collapsed onto two, and the "off the home die"
reading of it does not survive. The table above should be read as a floor plus
two crossings whose AICPU half has finer structure than stated; re-deriving the
term per cluster has not been done.

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
the same binary: 366 ns inside a cluster, 384 ns across a die, 723 ns across the
die boundary, versus 1114 and 1370 ns for AICPU to
AICore](figures/aicore-aicore.png)

**All 108 AIC/AIV sub-cores, all 5778 unordered pairs, 12 repeats each** —
69 336 measured round trips over 24 arena-runs, zero failures, with all 108
blocks serving in every run.

| Pair | Round trip |
| ---- | ---------- |
| Same cluster (AIC↔AIV or AIV↔AIV) | **366 ns** |
| Same die, different clusters | **384 ns** |
| Across the die boundary | **723 ns** |

- **An AICore pair hands off 2.9× faster than an AICPU does** — 384 ns against
  the 1114 ns floor on the same binary. The AICPU is the slow party, not the
  fabric.
- **Sharing a cluster buys 18 ns.** The 1 AIC + 2 AIV of a cluster share a 3 KiB
  SSBUF and an L2, but this protocol hands off through GM, so that path is never
  on it. Measuring what SSBUF is worth needs a kernel that actually uses
  `__ssbuf__`; this result says only that cluster locality is not free money for
  a GM-mediated hand-off.
- **AIC and AIV are interchangeable**: 379.2 ns AIC↔AIC, 383.8 AIC↔AIV, 385.3
  AIV↔AIV, all same-die cross-cluster. Six nanoseconds across the three.
- **Inside a die the fabric is flat.** Cluster-index distance *looks* correlated
  with latency (r = +0.72), but that is an artefact of which clusters sit at the
  extremes: remove additive per-cluster offsets and it collapses to **r = −0.09**
  with 7.9 ns of residual spread. What is left is ~20 ns of per-cluster
  idiosyncrasy with no spatial pattern — and the dies do not even share it, die
  0's slowest cluster being 1 and die 1's being 17. There is no NoC distance
  gradient to schedule around.

### The allocation matters ~20 ns, and the die it lands on does not

The eight arenas of a launch get the same eight addresses every time, 18 MiB
apart, and each one's cost is reproducible to 1–2 ns across launches:

| arena | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| ----- | - | - | - | - | - | - | - | - |
| median ns | 396 | 383 | 385 | 380 | 387 | 381 | 377 | 388 |

A 19.8 ns span, stable, and not monotone in address. **But both dies move
together**: over all 24 arena-runs the median gap between die-0 pairs and
die-1 pairs in the same arena is **1.0 ns** (worst case 3.8 ns).

This **corrects an earlier figure in this document**. The 8-cluster subset
reported a ~31 ns arena-home effect for AICore pairs. It does not survive full
sampling: that subset drew 4 clusters per die whose intrinsic offsets differ
between the dies, and the apparent home effect was that imbalance. For an
AICore pair there is no measurable home die.

## Two dies, one GM

![The full 108x108 matrix of AICore-to-AICore round trips beside the same 5778
numbers as a histogram. The matrix shows two pale blocks on the diagonal and two
dark off-diagonal blocks; the histogram shows two peaks separated by 241 ns of
completely empty space](figures/aicore-matrix.png)

```text
                          same die      across dies
pairs                         2862             2916
slowest / fastest         433.9 ns         675.0 ns
                   --- 241.1 ns with nothing in it ---
```

**The dies are real, and latency alone proves it.** Threshold the 108×108 matrix
at its widest empty band, keep the fast edges, take connected components — a
procedure that never reads a slot number. It returns exactly two groups of 54,
each holding 18 AIC and 36 AIV, and they agree with `slot / 54` on all 108
cores. This is not a statistical tendency: **no same-die pair is slower than any
cross-die pair**, and 241 ns separates the two populations.

**The memory is not divided the same way.** Four independent lines say one pool:

| Evidence | What it shows |
| -------- | ------------- |
| Allocations vary ~20 ns by address, but identically for both dies (1.0 ns asymmetry over 24 arenas) | an allocation has no die affinity a core can feel |
| An AICore GM load costs 86 ns to either home, **+0.16 ns** across the boundary | reaching the far side of the chip for data is free |
| `Ascend950DT_9592.ini` declares `io_die_num=2` beside a single `memory_size` of 96 GiB | the vendor models two dies and still reports one memory |
| No ACL, runtime, HAL or DSMI call reports per-die memory or accepts a die when allocating; `drv_mem_side` has `MEM_HOST_NUMA_SIDE` and no device equivalent | there is no interface to place, or even observe, an allocation per die |

CANN is not merely silent about dies. It uses the per-die idiom freely — CCU
pools ("单 die 容量有限"), task dispatch ("die IDs must alternate between die 0
and die 1"), RDMA `die_index`, and `dieId` in every error string. It never uses
it for GM.

**What the boundary costs is visibility between cores, not distance to memory:**

| Change | Cost |
| ------ | ---- |
| Move the memory to another allocation, both cores staying together | **~20 ns**, and the same for both dies |
| Move one core to the other die | **+339 ns** |

If a hand-off were two cores doing GM round trips to a word at address X, the
cost would track where X is. It does not. And +339 ns is about twice the ~176 ns
one-way store-visibility penalty measured separately — what a round trip that
crosses the boundary twice should cost.

**So: two dies of cores over one GM.** Draw the boundary between the cores.

### `Ascend950PR_957b`: the bipartition reproduces, the allocation row does not

84 sub-cores (28 AIC + 56 AIV, 14 clusters per die, die 1 cluster 4 absent), all
3486 pairs × 16 repeats, same parameters as the DT run:

| | 950DT_9592 | 950PR_957b |
| --- | ---: | ---: |
| same cluster | 366.9 | 390.6 |
| same die, other cluster | 384.0 | 389.7 |
| across dies | 722.9 | 791.3 |
| empty band at the split | 241.1 | 328.1 |
| blind bipartition | 54/54 exact | 42/42 exact |
| GM read, cross-die delta | +0.16 | −0.57 |
| **per-arena \|die0 − die1\|** | **1.0** | **20.9** |

ns. The bipartition is as clean on the PR as on the DT — 42/42, matching
`slot / 54` on all 84 cores, with nothing in the band. Reads stay uniform. But
the first row of the evidence table above does **not** carry over: on the PR an
allocation's cost differs by ~21 ns between the two dies, and the polarity flips
between launches at the same device virtual address. So *"an allocation has no
die affinity a core can feel"* is a statement about `_9592`, not about a5. The
conclusion it supports still holds — 21 ns sits an order of magnitude under the
~400 ns a core placement costs, and no API exposes it — but quote it per part.

One thing this does **not** settle. The AICPU↔AICore law above has two terms
keyed to the arena's home die, worth +170 and +225 ns, and that home was
identified by which configuration ran faster. The AICore-pair data finds no home
effect at all. Both were measured on this chip and they are in tension; the
one-way timestamp experiment is what would resolve it. Until then, treat the
AICPU-path home terms as describing a real effect whose cause is not established.

## Caveats

- **The floors are not portable between builds.** The same tool, rebuilt with a
  larger arena, moved the AICPU↔AICore floor from 975 to 1114 ns and the terms to
  +80 / +176. The *contrasts* held across both builds; the absolute floor did
  not. Quote the structure, not the constant.
- **Only two SKUs are measured**, `_9592` and `_957b`. CANN's a5 ini files span
  `ai_core_cnt` from 4 to 36 and `ai_cpu_cnt` from 5 to 7, so no count here is an
  a5 constant. The AICPU↔AICore section and the GM-read table are DT-only.
- The AICPU↔AICPU floor matrix is the best of 6 sweeps on device 0 and is
  superseded by the per-address tables beside it (12 processes × 16 blocks,
  device 1); the AICPU↔AICore numbers are medians over 16 arena sweeps across
  two launches; the AICore↔AICore matrix is the median of 12 repeats per pair
  over 24 arena-runs on device 1.
- **A subset can invent structure that full sampling removes.** Two claims in
  earlier drafts came from an 8-cluster subset and did not survive measuring all
  108 sub-cores: a ~31 ns arena-home effect for AICore pairs (actually 1.0 ns),
  and an apparent cluster-distance gradient (actually per-cluster offsets). Both
  are corrected above. Prefer the full matrix wherever the two disagree.

## How it was measured

Two standalone CANN tools, each a host `.so` + AICPU `.so` + AICore `.o` launched
on ACL streams with no runtime dependency:

- `../aicpu-pingpong/` — AICPU core-to-core, atomic ping-pong over a counter
  in the device SO's **BSS, not GM**: GM-side atomics trap from the AICPU's
  libc. Its numbers are therefore not directly comparable with the GM
  hand-offs measured by the other tool, which is why no table mixes them.
- `../aicpu-aicore-pingpong/` — AICPU↔AICore and AICore↔AICore, raw GM
  hand-off. Launches `blockDim=36`, which on a5 is a bijection onto the 36
  clusters, so every one of the 108 AIC/AIV slots reports in and a slot is
  chosen by coreid filter rather than by placement. It filters to one pair per
  round and shuffles the pair order, so position in the schedule cannot be
  mistaken for position on the chip. A schedule longer than the arena's
  `pair_list` is split into equal chunks across arenas, which keeps the arena
  layout — and therefore the comparability of the numbers — unchanged.

Both use bypass primitives on the AICore side (`ld_dev` / `st_dev` + `dsb`) and
volatile + `rmb()` / `wmb()` on the AICPU side, one writer per 64-byte line, and
time against the fixed 1 GHz system counter.
