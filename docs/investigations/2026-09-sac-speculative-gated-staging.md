# scan_and_claim: speculative gated staging of guaranteed-ready consumers

**Date**: 2026-09-08
**Verdict**: implemented, correct, and **measured as a regression** — qwen3
45.28 → 47.47 ms (**+4.8%**), PA Case2 10.16 → 10.32 (+1.6%), Case1
19.05 → 19.23 (+0.9%). Reverted. Speculation pre-commits cores and the
scheduling freedom it gives up is worth more than the dispatch latency it buys.

## Question

During qwen3's long full-chip tasks (~282 µs) the scheduler only polls for
FINs. If a running task's consumer is *guaranteed* to become ready the moment
that task retires, park it on a core behind the doorbell gate now, so the
producer's FIN costs one 64-bit store instead of a claim + prepare + publish
round-trip — and the AICore fills the staged task's `args[]` during its gate
wait, off the AICPU entirely
([2026-07](2026-07-aicore-fills-all-args-ready-path.md)).

## The readiness predicate is exact

Reach consumer `C` by walking running task `T`'s fanout CSR. `T` is therefore a
producer of `C`, and `T` has not retired, so it has not decremented. If
`C.fanin_remaining == 1`, that one remaining decrement **must** be `T`'s — no
other producer can race it to zero, because they have all already decremented.
`T`'s own fanout count never enters the argument, which matters: on qwen3
`fanin == 1` covers 9,883 / 11,085 tasks (89.2%), while the stricter "producer
also has fanout == 1" covers **1 task** (single-fanout producers there feed the
*high*-fanin consumers, 11/27/87).

## What was built

`stage_guaranteed_consumers` in the existing `try_early_dispatch` slot (its two
queue-driven paths are dead under polling — nothing pushes
`early_dispatch_queues` since `propagate_dispatch_fanin` became a stub). It
dedups the distinct tasks running on the thread's cores, keeps producers with
`allow_early_resolve`, asks the ring for guaranteed-ready consumers, and reuses
`stage_consumer_blocks` unchanged — which already force-gates, does one `wmb()`,
records `early_dispatch_doorbell_table`, ORs `staged_core_mask`, and handles the
producer-retires-mid-staging race. Release is `release_staged_consumer` at the
zero-crossing inside `on_mixed_task_complete`.

Correctness held: the 8-scene gate passed at `SAC_THREADS` 1 and 4, plus
`runtime_fatal_codes` at N=1/2. The whole hbg gating/doorbell apparatus works
in sac; it simply had no candidate source.

## Why it loses

Staging **commits the core**. The AICore defers its ACK until *after* the gate,
so `pending_occupied` stays set for the whole gated window and nothing else can
run there.

At each qwen3 layer boundary *two* consumers become ready and compete for the
same 24 AIC cores: a `bn=24` cube task, and a dummy whose retirement unlocks 26
single-core cube tasks. Staging pre-commits all 24 AIC cores to the first before
the MIX producer has retired, and each core is locked from the moment it
finishes *its own* MIX block until the *last* block lands. Cores that finish
early sit gated instead of running the 26 siblings, and the scheduler loses the
freedom to interleave.

The PA numbers isolate the two costs. **No PA task sets
`allow_early_resolve`** (0 of 32,768 Case2, 0 of 65,536 Case1), so staging
cannot fire there at all: PA's +0.9–1.6% is pure discovery overhead — walking up
to 18 running cores per loop pass to find no eligible producer. Subtracting it,
~3–4% of qwen3's +4.8% is the staging itself.

## Trap worth keeping

The first build faulted the AICPU (`errcode=0x2a`) on **round 4** of qwen3,
rounds 1–3 clean. The orchestrator submits progressively, so the fanout CSR
names consumers whose slots the round has not reached yet; their `task`/`payload`
pointers survive from the *previous* round, so a null check only catches them in
round 1 (where slots are still zeroed), and their fanin counter can read 1 while
belonging to the older window. **Any new path that reads slots by id needs the
`current_task_index` bound that `scan_ready_tasks_batch` carries**, alongside the
`fanin_pending` cross-round test the deferred-ready FIFO uses.

## If revisited

The ceiling is small: the addressable prize is the ~4.6 ms sac loses to hbg on
qwen3, since half that graph's wall is full-chip MIX with nothing to gain (see
[2026-09 qwen3 characterization](2026-09-qwen3-scheduler-headroom.md)). A version
that caps staged blocks so it never commits more cores than would otherwise idle
might avoid the flexibility loss, but it must beat a 4.8% deficit first.
