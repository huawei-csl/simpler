# Why the GroupQueue wins even against free transport

**The claim this page carries:** the GroupQueue's advantage over the AICPU
scheduler is not the cost of moving data between the AICPU and the AICores. Give
M0 a zero-latency connection -- free task send, free status read, instant
completion notice -- and it still loses to an unmodified M2 by **2.0x on
paged_attention and 1.4x on qwen**. Free transport closes **27 %** of the gap on
the first case and **2 %** on the second.

The reason is where the scheduler sits, not how fast it talks. In M0 the AICPU is
on the path between every producer finishing and its consumer starting, and that
hop costs scheduler software that no interconnect improvement touches. The
GroupQueue takes the AICPU out of that hop.

Method and run commands are in [running.md](../running.md); the simulated
baseline's own error against silicon is in [fidelity.md](fidelity.md).

## The ablation (2026-10-06)

M0's injected latencies are calibration inputs -- `LAT push read ack notice`, in
ns -- so the experiment changes only the calibration file and needs no rebuild:

| arm | what is free | paged_attention `LAT` | qwen `LAT` |
| --- | ------------ | --------------------- | ---------- |
| `m0_base` | nothing: the calibrated model | 5 195 1119 80 | 5 195 1209 140 |
| `m0_nosend` | sending: the task write and the push-to-ack delay | **1** 195 **1** 80 | **1** 195 **1** 140 |
| `m0_nopoll` | polling: the status read and the completion becoming observable | 5 **1** 1119 **1** | 5 **1** 1209 **1** |
| `m0_ideal` | all transport | **1 1 1 1** | **1 1 1 1** |
| M2 | -- | unmodified | unmodified |

**M2 is deliberately left at its real modelled latencies.** Comparing an idealised
M0 against a non-idealised M2 is the steelman: if the GroupQueue still wins there,
the win cannot be attributed to transport.

### The headline

Real silicon, M0, idealised M0 and M2, interleaved in one device lock -- 4 reps of
14 rounds on paged_attention and 3 on qwen, first two rounds of each run dropped,
giving 48 and 36 samples per arm. qwen is the in-repo expanded 40-layer
orchestration (11,087 tasks). M2 includes `6244769b`, which made the controller
offer both of a package's vector cores rather than only the first.

| arm | paged_attention Case1 | qwen 40L |
| --- | --------------------- | -------- |
| real silicon (`a2a3`) | 21.173 ms | 34.632 ms |
| `m0_base` | 22.282 ms | 33.147 ms |
| `m0_ideal` | 18.867 ms (**-15.3 %**) | 32.986 ms (**-0.5 %**) |
| M2, unmodified | 9.430 ms | 23.779 ms |
| **idealised M0 / M2** | **2.00x** (+100.1 %) | **1.39x** (+38.7 %) |
| **share of the M0-to-M2 gap free transport closes** | **26.6 %** (3.42 of 12.85 ms) | **1.7 %** (0.16 of 9.37 ms) |

Against **real silicon running today's scheduler**, idealised M0 is only 10.9 %
faster on paged_attention and 4.8 % on qwen -- and on qwen most of that is the
model's own error, which reads M0 4.3 % fast ([fidelity.md](fidelity.md)) --
while M2 is 55.5 % and 31.3 % faster than the same silicon. Transport is not
where the difference lives.

### Where the transport cost sits

From an earlier session, before `6244769b`; that commit touches only M2, and the
M0 arms repeated in the headline run reproduce these to 0.2 %:

| arm | paged_attention Case1 | qwen 40L |
| --- | --------------------- | -------- |
| `m0_base` | 22.233 ms | 33.096 ms |
| `m0_nosend` | 21.331 ms (-4.1 %) | 33.049 ms (-0.1 %) |
| `m0_nopoll` | 19.599 ms (-11.8 %) | 33.047 ms (-0.1 %) |
| `m0_ideal` | 18.881 ms (-15.1 %) | 32.984 ms (-0.3 %) |

On paged_attention polling is worth 11.8 % and sending 4.1 %, roughly additive.
The expensive part of the transport is *asking*, not *telling* -- and even all of
it together is about a quarter of what the GroupQueue buys.

**Work was identical across every arm.** Compute issued agrees between M0 and M2
to 0.065 % on paged_attention (91.132 against 91.072 ms) and 0.034 % on qwen
(746.582 against 746.324 ms), every task completed on both, and it holds at
22.5-23.1 ms and 186.3-187.3 ms per thread across all four M0 arms. Nothing but
transport moved.

### Both corrections run in the thesis's favour

- **The idealised arms are inflated, by a bounded amount.** aSim charges a
  modelled latency by spinning to a deadline; below the simulator's own per-call
  work (~15-27 ns) it overruns, and the excess lands inside the window. The
  baselines overrun nothing. The ideal arms overrun 347 us/thread on
  paged_attention and 1.14 ms/thread on qwen. Subtracting it, idealised M0 sits
  near 18.52 and 31.84 ms -- still **1.96x and 1.34x** M2 -- so free transport
  closes at most **29 %** and **14 %** of the gap.
- **M2 is inflated far more.** aSim models the controller in software on the
  manager's own thread, and that work overruns its modelled latency by
  7.30 ms/thread on paged_attention and 14.71 ms/thread on qwen, inside windows of
  9.43 and 23.78 ms. On silicon that work is the controller's. So M2 is faster
  than quoted, and the ratios above are floors.

One reading not to take from the headline: paged_attention's M2 is lower here
(9.430 ms) than before `6244769b` (9.750 and 9.985 ms in two earlier sessions),
where that commit's own paired A/B -- binaries swapped inside one session --
measured the fix as costing M2 about 2.9 %. Both are right. PA's M2 arm spans
14.9 % across reps, so a comparison across sessions cannot resolve a 3 % effect;
the paired figure is the estimate of the fix's cost.

## Why the GroupQueue still wins

### paged_attention: the AICPU runs out of throughput

With transport at 1 ns, each M0 thread still spends **953 ns of wall time per
task** -- a 15.68 ms scheduling window over 16,448 tasks -- against 1,160 ns at the
calibrated latencies. That residue is real scheduler code: aSim replaces the
AICores, not the scheduler, so `host_build_graph`'s dispatch loop executes at the
AICPU's genuine speed. It is not a simulator artefact.

Work is not what is short. 16,384 AIC tasks sit ready at once
(`QPROBE rq[AIC] maxocc=16384`), while the cores are ~8 % busy -- 91 ms of
compute per round across 72 cores in a 15.7 ms window. To keep a thread's 18
cores fed with ~1.4 us kernels it would have to turn a task around every ~80 ns;
it takes ~950. Ready work waits behind an AICPU that is busy with bookkeeping.

### qwen: the AICPU has spare throughput, and still loses

Here the per-task AICPU wall is 11.5 us and does not move with free transport
(11,533 -> 11,497 ns): the threads are mostly waiting, not working. So the 28 % M2
wins is not a throughput result. qwen is a single connected graph of dependent
layers, so its makespan is its critical path, and in M0 every hop on that path
routes through the AICPU -- notice that the producer finished, decrement its
consumers' fan-in, enqueue, dequeue, choose a core, dispatch. That round trip is
software, and no MMIO speed-up reaches it.

**This mechanism is established by elimination, not measured directly** -- see
[What is not yet measured](#what-is-not-yet-measured).

### What the GroupQueue changes

Three structural differences, each present by design and independent of any
latency constant:

1. **Readiness is resolved next to the cores.** The manager submits a task while
   up to `SIM_HELD_MAX_DEPS` (4) of its producers are still running; the
   controller holds it and releases it the moment the watermark passes them. The
   AICPU never returns to a consumer after its producers finish. M0 must return
   to every consumer after every producer.
2. **Completion is learned in bulk.** One watermark read retires a contiguous run
   of finished positions -- 16,448 completions from 6.3-7.3k reads per thread on
   paged_attention -- where M0 learns them core by core. At zero latency the
   saving is not the read itself but the per-completion bookkeeping that goes
   with it.
3. **Nothing is shared.** Each manager owns its queue privately, with no atomics
   on the hot path. M0's four threads route every one of paged_attention's 32,768
   AIC tasks through one shared ready queue (`QPROBE rq[AIC] pushes=32768`). An
   earlier real-silicon measurement is consistent with that costing throughput:
   raising `host_build_graph` from 3 to 4 dispatch threads made small cases 5-12 %
   *slower*, which is the signature of contention rather than of work. That
   measurement was on small cases, so it supports the point rather than proving
   it here.

Together they change the AICPU's job. In M0 it is a **per-task dispatcher on the
critical path**; with the GroupQueue it is a **feeder that keeps a 32-slot window
full** -- a throughput job done off the critical path, which is why the window M2
leaves behind is mostly the simulator modelling the controller.

**One counter shows the distinction directly.** With polling made free, M0 polls
*more* -- 21.1k -> 23.8k reads per thread on paged_attention, 46k -> 64k on qwen --
and goes no faster. Cheap asking only means asking more often. A controller that
signals instead of being asked has no such loop to speed up.

## What this does and does not establish

- **It separates transport latency from everything else, not signal from poll.**
  The ablation leaves M0's structure intact: one thread serially walking 18
  cores, a shared ready queue, software dependency resolution. Isolating
  "signalling instead of polling" from "locality" and from "parallel completion"
  needs an M0 variant whose completion path is both free and parallel, which is a
  build rather than a calibration change.
- **Two cases, both far from core saturation.** A core-bound case could shrink
  every gap here, since there is less idle silicon for any scheduler to fill.

### What is not yet measured

How paged_attention's 953 ns per task splits between dependency resolution,
shared-queue contention and dispatch logic, and the per-hop mechanism on qwen.
Both are identified by elimination. The decisive measurement is a **critical-path
trace**: the gap between a producer's end and its consumer's start, M0-ideal
against M2, from a swimlane capture of each arm.
