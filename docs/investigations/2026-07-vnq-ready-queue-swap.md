# Replacing the tmr ready-task queue with a VNQueues concurrent queue

**Date**: 2026-07-22
**Verdict**: dropped as a general change — one variant is a real but
**workload-specific** win (decode_fwd 40L −0.5%, decode_tq 40L +0.2%); the
rest are neutral/no-op. Not worth landing unconditionally.

## Question

`huawei-csl/queues` (VNQueues) is a header-only library of hand-tuned
lock-free ring buffers (SPSC, atomic-element MPMC with cacheline
reindexing, ticket-locked MPMC, and single-owner steal/donate variants).
The tmr polling scheduler's global ready queues (`PTO2ReadyQueue`, one
per resource shape) are a generic sequence-based Vyukov MPMC. Could
swapping in a more specialized VNQ queue buy measurable device time?

The intuition: with 3 scheduler threads pushing/popping one shared queue
per shape, cross-thread contention on the cursors and false-sharing on
the slot array might be a real cost worth removing.

## What was tried

Four experiments on the polling runtime (`src/a2a3/runtime/tensormap_and_ringbuffer`),
each a clean drop-in preserving the `push/push_batch/pop/pop_batch/size/
reset` + arena-layout surface. A/B was whole-`.so`-swap, device time
(`Avg Device` = on-NPU `device_wall` from `[STRACE]` markers,
`strace_timing --rounds-table`), against baseline `c349446d`.

1. **`PTO2ReadyQueue` → VNQ `MPMC_AtomicRingBuffer`** (Egorushkin
   atomic-element: the slot pointer is itself the sync variable; consecutive
   logical positions reindexed onto physically distant slots, stride 9,
   coprime to capacity, to kill slot false-sharing). Arena-backed, runtime
   capacity. Branch carries this change.
2. **`AICoreCompletionMailbox` batched push** — reserve all of a task's
   deferred completion conditions with one head-CAS instead of one per
   condition (`try_push_conditions_batch`).
3. **`PTO2ReadyQueue` → VNQ ticket-spinlock MPMC** (`TicketedSpinLock`
   guarding a plain ring) — the locked counterpart, to bracket the
   synchronization-discipline axis.
4. **SPCMT/SPCMD work-stealing** — analysis only (see "Result").

Workloads: DSv4 component suite (26 kernels, phased 4-device multi-round),
qwen `decode_fwd` 40L (batch-16, `--decode-steps`, pypto-lib 9931cac),
qwen `decode_tq` 40L (`--num-layers 40`, pypto-lib a085c15), the in-repo
`qwen3_14b_decode` scene, and `benchmark_bgemm`. Significance via **paired
interleaved** V/D runs (back-to-back same device so slow drift cancels)
with a paired t-test (`tmp/polling_ab/paired_stats.py`).

## Result

**Phase breakdown first (why the ceiling is low).** Baseline
`strace --tree`: `device_wall` is **95-99% `graph_build`** — qwen decode
2283/2307 µs, bgemm 380/399 µs. `graph_build` is the concurrent
orchestrate+dispatch+compute phase, bounded by orchestration
(`submit_task` rate). There is **no separate scheduler span**; dispatch
and its queue ops are fully overlapped inside `graph_build`, off the
critical path. So for light / orchestration-bound workloads a queue swap
cannot move the wall, by construction.

**1 — atomic-element + reindex.** Neutral where queue traffic is low, but
a **real win where it is high**:

| workload | Δ device (V vs baseline) | evidence |
| --- | ---: | --- |
| decode_fwd 40L | **−0.50%** | paired n=11, t=−15.96, 95% CI [−0.56%, −0.43%], p≈1e-8, zero overlap (V 36256-36385 < D 36423-36570) |
| decode_tq 40L | **+0.19%** (slower) | paired, 3 clean pairs +0.19/+0.08/+0.35% (median +0.19%); robust to drift (V ran first yet slower) |
| DSv4 suite (26) | ~0 (−0.2%/+0.26%, 5-round) | spreads overlap on every row |

The decode_fwd win was invisible in early contended runs (measured −0.2%
within noise) and only resolved under a **low-contention window** — under
load the device is core-bound, hiding the scheduler delta. Mechanism: the
reindex stride spreads consecutive ready-queue slots across cache lines,
cutting false-sharing among the 3 scheduler threads *when* enough tasks
are concurrently ready (40-layer forward decode). decode_tq's turboquant
graph has a lighter ready-queue pattern where the reindex multiply is pure
overhead with no false-sharing to relieve → marginally slower.

**2 — mailbox batched push.** Correct (golden + all 3 async demos pass),
but a **no-op on every available workload**: no production model calls
`register_completion_condition` (it is absent from the pypto-lib codegen),
so `cond_count==0` and the mailbox is never touched; the only workloads
that use it are async demos, each with `cond_count≈1`, so the batch
reserves one slot = identical to a single push. No workload in-tree has
`cond_count>1`, so the batching never batches.

**3 — ticket-lock MPMC.** Neutral: DSv4 3-round mean −0.47% (spreads
overlap), decode_fwd 40L +0.33% (marginally slower, as expected for a lock
vs lock-free). Brackets the design space — algorithm *and* synchronization
discipline are both irrelevant on the light/orchestration-bound workloads.

**4 — SPCMT/SPCMD work-stealing.** Not applicable as a swap: the polling
scheduler's per-thread `PTO2LocalReadyBuffer` + excess-donation machinery
is **dead code** — `try_push` is never called anywhere, `local_bufs` are
created empty and only drained, so the donation loop never fires and every
ready task already flows through the global MPMC. There is no live
donation to improve. A real local-first + steal design would be a
from-scratch rewrite, and with only **3 scheduler threads**
(`aicpu_thread_num=4`) VNQ's contention advantage (which needs 8-16+
threads) barely applies.

## Why not (now)

The atomic-element variant is the only tangible result, and it is a
**workload-specific trade**, not a free win: −0.5% on decode_fwd, +0.2% on
decode_tq, ~0 elsewhere. Landing it unconditionally would regress
decode_tq to speed up decode_fwd. The other three variants are
neutral / no-op / not-applicable. The root cause is structural: the ready
queue is off the critical path (95-99% `graph_build`) on all but the
narrowest dispatch-heavy workloads, and the box only runs 3 scheduler
threads, so there is little cursor contention for a specialized queue to
remove.

## When to reconsider

- **A dispatch-bound workload becomes a flagship** — many concurrent
  ready tasks, high per-thread imbalance, scheduler on the critical path.
  Then the atomic-element reindex (and possibly per-thread sharding) could
  matter more than ±0.5% on one workload.
- **Scheduler thread count rises** well above 3 — VNQ's atomic-element and
  steal/donate designs pay off at higher concurrency.
- **Completer-local sharded ready queues** are the one untested shot with
  a real mechanism (keep the common completion→dispatch path off any shared
  cursor). It needs `thread_idx` plumbed through the fanout path
  (`register_wake` / `drain_wiring_queue` / async completion) — moderate
  surgery, and the phase data predicts it is invisible in the wall for
  current workloads, so only worth it alongside the first bullet.
- If the atomic-element win is wanted for decode_fwd specifically, it must
  first be shown not to regress decode_tq under a clean (low-contention)
  measurement, and the a5 / host_build_graph copies of `PTO2ReadyQueue`
  must be ported for a mergeable change.

## References

- The atomic-element swap itself is the commit this branch carries
  (`perf(tmr): swap ready queue to VNQ atomic-element + reindexing MPMC`).
- Other experiment branches (local, not carried here):
  `polling-mbox-batch` (mailbox batching), `polling-sharded-rq`
  (ticket-lock).
- A/B harnesses: `tmp/polling_ab/vnq_sig_decode40.sh`,
  `vnq_sig_dtq.sh`, `paired_stats.py`, `vnq_dsv4_multi.sh`.
- Source: `huawei-csl/queues` (VNQueues), header-only C++17.
- Related: [`.claude/rules/discipline.md`](../../.claude/rules/discipline.md)
  §4 (check investigations before re-proposing), the orchestration-bound
  finding for paged_attention (memory: `reference-paged-attention-orch-bound`).
