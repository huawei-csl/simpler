# Why the GroupQueue wins even against free transport

**The claim this page carries, on paged_attention:** the GroupQueue's advantage
over the AICPU scheduler is not the cost of moving data between the AICPU and the
AICores. Give M0 a zero-latency connection -- free task send, free status read,
instant completion notice -- and it still takes **1.63x** the time of an
unmodified M2. Free transport closes **30 %** of the gap.

The reason is where the scheduler sits, not how fast it talks. In M0 the AICPU is
on the path between every producer finishing and its consumer starting, and on
paged_attention that hop costs scheduler software that no interconnect
improvement touches. The GroupQueue takes the AICPU out of that hop.

**On qwen there is no gain to explain.** qwen is bound by its cube cores'
compute, M0 already runs within ~4 % of that floor, and M2 is 13 % *slower* than
M0 ([validation.md](../validation.md#what-the-groupqueue-is-worth)). The claim
here is about a scheduler-bound graph, which is what paged_attention is.

Method and run commands are in [running.md](../running.md); the simulated
baseline's own error against silicon is in [fidelity.md](fidelity.md). All
figures are on the model clock ([DESIGN.md](../DESIGN.md) §3), measured
2026-10-07.

## The ablation

M0's injected latencies are calibration inputs -- `LAT push read ack notice`, in
ns -- so the experiment changes only the calibration file and needs no rebuild:

| arm | what is free | `LAT` |
| --- | ------------ | ----- |
| `m0_base` | nothing: the calibrated model | 5 195 1119 80 |
| `m0_nosend` | sending: the task write and the push-to-ack delay | **1** 195 **1** 80 |
| `m0_nopoll` | polling: the status read and the completion becoming observable | 5 **1** 1119 **1** |
| `m0_ideal` | all transport | **1 1 1 1** |
| M2 | -- | unmodified |

**M2 is deliberately left at its real modelled latencies.** Comparing an idealised
M0 against a non-idealised M2 is the steelman: if the GroupQueue still wins there,
the win cannot be attributed to transport.

All five arms interleaved in one device lock, 3 reps of 14 rounds, first two
rounds of each run dropped -- 36 samples per arm.

| arm | paged_attention Case1 | vs `m0_base` |
| --- | --------------------- | ------------ |
| `m0_base` | 21.722 ms | |
| `m0_nosend` | 20.754 ms | -4.5 % |
| `m0_nopoll` | 19.350 ms | -10.9 % |
| `m0_ideal` | 18.649 ms | **-14.1 %** |
| M2, unmodified | 11.432 ms | -47.4 % |
| **idealised M0 / M2** | **1.63x** | |
| **share of the M0-to-M2 gap free transport closes** | **29.9 %** (3.07 of 10.29 ms) | |

Polling is worth 10.9 % and sending 4.5 %, roughly additive. The expensive part
of the transport is *asking*, not *telling* -- and even all of it together is
under a third of what the GroupQueue buys.

**The window is the modelled design, not the simulator.** Each arm's simulator
work beyond the latency it charges goes on its thread's ledger, during which its
cores stand still, and the same ledger is removed from `device_wall`. The free
arms charge 1 ns for work that costs ~15-27 ns to simulate, so their ledgers are
larger -- 1.27 ms per thread in `m0_ideal` against 0.80 ms at baseline -- but they
are removed exactly rather than inflating the arm. Within a round M0's threads'
ledgers diverge by at most 0.02 ms and M2's by 0.23 ms, which bounds the one
residue a per-thread clock cannot remove.

**Work was identical across every arm**: compute issued agrees to 0.04 % between
M0 and M2, and every task completes in every run.

## Why the GroupQueue still wins

### The AICPU runs out of throughput

With transport at 1 ns, each M0 thread still spends **940 ns of wall time per
task** -- a 15.46 ms scheduling window over 16,448 tasks -- against 1,128 ns at the
calibrated latencies. That residue is real scheduler code: aSim replaces the
AICores, not the scheduler, so `host_build_graph`'s dispatch loop executes at the
AICPU's genuine speed.

Work is not what is short. 16,384 AIC tasks sit ready at once
(`QPROBE rq[AIC] maxocc=16384`), while the cores are ~8 % busy -- 91 ms of compute
per round across 72 cores in a 15.5 ms window. To keep a thread's 18 cores fed
with ~1.4 us kernels it would have to turn a task around every ~80 ns; it takes
~940.

**A critical-path trace shows the same thing directly.** aSim writes the per-task
core records a real AICore writes, so `simpler_setup.tools.critical_path` runs on
it unchanged. On `m0_ideal` the as-executed critical path is **88.1 % stall and
11.9 % compute**, against a dependency floor (static CPM) of 0.08 ms in a 17.0 ms
makespan. With transport free, every one of those stalls is the scheduler: in
81.1 % of the makespan the task was ready and its core idle, and in the other
7.0 % its producer had finished and it had not been dispatched. `m0_base` reads
87.5 % stall.

### What the GroupQueue changes

Three structural differences, each present by design and independent of any
latency constant:

1. **Readiness is resolved next to the cores.** The manager submits a task while
   up to `SIM_HELD_MAX_DEPS` (4) of its producers are still running; the
   controller holds it and releases it the moment the watermark passes them. The
   AICPU never returns to a consumer after its producers finish. M0 must return
   to every consumer after every producer.
2. **Completion is learned in bulk.** One watermark read retires a contiguous run
   of finished positions, where M0 learns them core by core. At zero latency the
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
full**, a throughput job done off the critical path. M2's manager spends 544 ns
of wall per task on paged_attention, against M0's 1,128.

**One counter shows the distinction directly.** With polling made free, M0 polls
*more* -- 21.1k -> 23.5k reads per thread. Cheap asking only means asking more
often, and the free-transport arm still holds ready tasks beside idle cores for
81 % of its critical path. A controller that signals instead of being asked has
no such loop to speed up.

## What this does and does not establish

- **It separates transport latency from everything else, not signal from poll.**
  The ablation leaves M0's structure intact: one thread serially walking 18
  cores, a shared ready queue, software dependency resolution. Isolating
  "signalling instead of polling" from "locality" and from "parallel completion"
  needs an M0 variant whose completion path is both free and parallel, which is a
  build rather than a calibration change.
- **It is about a scheduler-bound graph.** paged_attention leaves its cores ~90 %
  idle under M0. On a compute-bound graph there is little for any scheduler to
  recover, and qwen shows M2 can lose there: why it spends 13 % more than M0 on
  qwen has not been traced yet.

### What is not yet measured

How paged_attention's 940 ns per task splits between dependency resolution,
shared-queue contention and dispatch logic. Splitting each critical-path stall at
the scheduler's dispatch stamp would separate the AICPU's share from transport,
but the stamp is on the real clock and the core records on the model clock, so
the stamp has to be moved to the model clock first.
