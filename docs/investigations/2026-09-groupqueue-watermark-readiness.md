# Resolving readiness from the watermark alone (2026-09-23)

**Verdict: keep the current rule.** The controller names up to
`SIM_HELD_MAX_DEPS` (4) unmet producer positions per entry, and the manager holds
a task back until its unmet count fits. Three cheaper-looking rules were modelled
against the recorded graphs of the two in-scope cases; one is exact but buys
nothing the current rule lacks, and two over-serialise badly.

## The proposals

The GroupQueue already publishes a **watermark** — the contiguous completed
prefix of its index space — and one read of it retires everything it covers. That
invites using it for readiness as well, since `position <= watermark` proves a
producer has finished:

1. **Watermark only.** A task carries one value, the highest index among its
   producers, and is ready when the watermark reaches it. Deletes the comparators
   and the hold array outright, and imposes no fan-in limit.
2. **Watermark + 4 indices.** A task carries its four highest producer positions.
   Fast path: the highest is at or below the watermark, so all of them are, and
   one compare admits it. Slow path: each carried position above the watermark is
   checked in the completion array.
3. **Watermark-dependency.** As (2), plus a fifth value — the highest index among
   the producers the entry could not carry — which the watermark must reach.

## What they cost

Offline replay of each rule against `deps.json` for both cases: greedy schedule on
72 cores, per-kernel durations from the run's own calibration, measuring only when
tasks become ready. Dispatch latency and controller occupancy are not modelled, so
read the ratios and not the absolute figures.

| rule | PA Case1 | qwen 40L |
| ---- | -------- | -------- |
| exact readiness | 1,286.9 us | 5,208.2 us |
| 1 · watermark only | +5,040 % | +328 % |
| 2 · watermark + 4 indices | **+0.00 %** | +255.8 % |
| 3 · plus a watermark-dependency | +0.00 % | +255.8 % |
| current rule (4 indices, manager holds the tail) | **+0.00 %** | **+0.00 %** |

**The watermark is sufficient but not necessary, and the gap is enormous.** The
prefix stalls on the oldest unfinished position, so a task ends up waiting for
everything submitted before its last producer rather than for its producers. PA
has 81,664 real producer edges; rule 1 implies 1.61 billion. qwen has 23,520 and
implies 60.2 million.

**PA's collapse is false serialisation of its 256 disjoint batch elements.** They
are submitted one after another, so element *g* occupies a contiguous index run
and cannot start until elements *0..g-1* have drained. Giving each component its
own index space and watermark recovers all of it (+0.1 %) — within one element the
rule is nearly exact, because a batch element really is a chain of joins. qwen is
one connected component, so that partition does not exist for it and per-component
measures the same +328 %.

**Rule 2 is exact for any task whose producers fit in its four slots**, with no
conservatism at all: every uncarried producer is below the fourth carried one, so
if that one is at or below the watermark the prefix covers them. PA's maximum
fan-in is 3, so nothing ever spills and rule 2 reproduces exact readiness. qwen's
maximum is 86.

**A spill is a full barrier, which is why 39 tasks cost 256 %.** Only 39 of qwen's
11,085 tasks ever waited on the prefix, but each waits for every older position to
drain and its whole downstream cone waits behind it. Raising the carried count
confirms the diagnosis: 8 gives +37.8 %, 16 gives +36.6 %, and 128 — above qwen's
maximum fan-in — gives +0.00 %.

**Rule 3 is sound but inert on these graphs.** Requiring the prefix to reach the
highest *uncarried* producer is strictly weaker than requiring it to reach the
fourth carried one, but across qwen's 920 high-fan-in tasks the gap between those
two positions is a median of 1 index and a maximum of 3: their producers are
contiguous, being reductions over adjacent blocks. The barrier is unmoved.

## Why the current rule already wins

`group_deps` skips producers that have retired and counts only unmet ones,
returning `kDepsTooMany` only when *those* exceed the comparators — so the manager
submits a task the moment its unmet count falls to four, not when it reaches zero.
That makes the current path exact by construction, and it is the reason rule 2 plus
the existing manager fallback measures +0.00 % on both cases: the four named
positions are all the unmet ones.

The tail that takes the manager route is 8.09 % of qwen's tasks and 0 % of PA's.

## Worth keeping from this

The **fast path in rule 2 is real and free**: when the highest carried position is
at or below the watermark the completion array need not be read at all. That holds
for 25.1 % of PA's admissions and 50.2 % of qwen's. It is an access-count
reduction, not a scheduling change, so it does not appear in the table above — and
it does not remove the array, which the other half to three-quarters still reads.

## When to re-open

- A graph whose high-fan-in tasks depend on a few *recent* producers plus a long
  tail of *old* ones. Rule 3 pays there, where the gap between the carried and
  uncarried positions is wide; it is worthless when producers are contiguous.
- A workload of wholly disjoint components that are also **declared**, where each
  could carry its own index space and watermark. Rule 1 then costs +0.1 % on a
  PA-shaped graph. It would make grouping load-bearing for performance rather
  than an optimisation: an undeclared graph would get one index space and the
  +328 % penalty, turning qwen's measured −33 % into roughly a 3x loss.
