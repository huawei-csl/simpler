/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#ifndef SCHEDULER_CONTEXT_H
#define SCHEDULER_CONTEXT_H

#include "common/l2_perf_profiling.h"
#include "common/unified_log.h"
#include "scheduler_types.h"
#include "callable.h"

#include "scheduler/pto_scheduler.h"

#include "aicore_completion_mailbox.h"
#include "pto2_dispatch_payload.h"

// These macros are defined in runtime.h, but we cannot include it here
// (it pulls in Handshake which we only forward-declare).  Mirror the
// authoritative values so the class layout compiles standalone.
#ifndef RUNTIME_MAX_WORKER
#define RUNTIME_MAX_WORKER 72
#endif
#ifndef RUNTIME_MAX_FUNC_ID
#define RUNTIME_MAX_FUNC_ID 1024
#endif

namespace {
inline constexpr int32_t PTO2_DEFERRED_RELEASE_CAP = 256;
}


// Forward declarations — avoid pulling in full headers for pointer/reference params.
class Runtime;
struct Handshake;
struct PTO2Runtime;

/**
 * SchedulerContext: owns all scheduler-side state and methods.
 *
 * Held as a member of AicpuExecutor (sched_ctx_).  The single public entry
 * point is resolve_and_dispatch(), called once per scheduler thread.
 *
 * All dispatch/completion/drain/cold-path logic is implemented as private
 * member methods, split across three .cpp files by responsibility:
 *   - scheduler_completion.cpp  (completion polling, drain protocol)
 *   - scheduler_cold_path.cpp   (exit checks, stall diagnostics, profiling)
 *   - scheduler_dispatch.cpp    (task dispatch loop and helpers)
 */
class SchedulerContext {
public:
    // =========================================================================
    // Lifecycle
    // =========================================================================

    // Initialize scheduler state from the given runtime and thread layout.
    // - Discovers cores via handshake_all_cores()
    // - Assigns cores to scheduler threads
    // - Resets task counters, payloads, per-core GlobalContext
    // - Binds func_id_to_addr_ / initial sched_ (if rt is already known)
    // - Captures AICore-register base (consumed by handshake_all_cores())
    // Returns 0 on success, negative on failure (handshake / assignment error).
    int32_t
    init(Runtime *runtime, int32_t aicpu_thread_num, int32_t sched_thread_num, bool orch_to_sched, uint64_t regs_base);

    // Reset all SchedulerContext-owned state to its post-construction defaults.
    // Called by AicpuExecutor::deinit() during per-run teardown.
    void deinit();

    // =========================================================================
    // Per-thread execution entry points (called by AicpuExecutor::run)
    // =========================================================================

    // Main scheduler thread entry: poll completion + dispatch ready tasks.
    int32_t resolve_and_dispatch(Runtime *runtime, int32_t thread_idx);

    // Shutdown AICore registers for this thread's assigned cores.
    // Also runs PMU finalize (PTO2_PROFILING) before deinit when enabled.
    // Orchestrator threads (core_trackers_[thread_idx].core_num() == 0) are a no-op.
    int32_t shutdown(int32_t thread_idx);

    // Run all post-orchestration scheduler bookkeeping:
    //  - publishes core assignments to the perf collector (PTO2_PROFILING)
    //  - latches submitted task count from PTO2 shared memory
    //  - folds inline_completed_tasks into completed_tasks_
    //  - flips orchestrator_done_ and triggers core transition
    //    (skipped on fatal error — emergency_shutdown runs instead)
    // Callers must invoke rt_orchestration_done(rt) before this — that
    // step belongs to the orchestrator lifecycle, not the scheduler.
    void on_orchestration_done(Runtime *runtime, PTO2Runtime *rt, int32_t thread_idx, int32_t total_tasks);

    // Bind the PTO2Runtime scheduler pointer. Required in device-orchestration
    // mode where rt is created by the orchestrator thread after init().
    void bind_runtime(PTO2Runtime *rt);

    // =========================================================================
    // State queries / external synchronization points
    // =========================================================================

    int32_t aic_count() const { return aic_count_; }
    int32_t aiv_count() const { return aiv_count_; }
    bool is_completed() const { return completed_.load(std::memory_order_acquire); }
    int32_t completed_tasks_count() const { return completed_tasks_.load(std::memory_order_acquire); }

    // Block until the first scheduler thread has finished one-time PTO2 init.
    // Called by the orchestrator thread in device-orch mode.
    void wait_pto2_init_complete() const;

private:
    // =========================================================================
    // State
    // =========================================================================

    // --- Scheduler binding & per-core runtime state ---
    alignas(64) PTO2SchedulerState *sched_{nullptr};
    PTO2Runtime *rt_{nullptr};

    // Per-core execution state, indexed by core_id (= worker_id)
    CoreExecState core_exec_states_[RUNTIME_MAX_WORKER];

    // Cluster-ordered core trackers, one per scheduler thread
    CoreTracker core_trackers_[MAX_AICPU_THREADS];

    // Per-core dispatch payload storage: dual-buffer for pipelining.
    // buf_idx = reg_task_id & 1; adjacent dispatches alternate automatically.
    PTO2DispatchPayload payload_per_core_[RUNTIME_MAX_WORKER][2];

    // Per-core deferred-completion software registration storage.  This has
    // the same runtime lifetime as payload_per_core_, but is kept out of the
    // dispatch payload so normal task dispatch layout and cache footprint stay
    // unchanged.
    DeferredCompletionSlab deferred_slab_per_core_[RUNTIME_MAX_WORKER][2];

    // sync_start drain coordination
    SyncStartDrainState drain_state_;

    // --- Task-execution tracking ---
    std::atomic<int32_t> completed_tasks_{0};
    int32_t total_tasks_{0};
    // Device orchestration: set by last orchestrator when graph is built; schedulers poll it.
    // volatile prevents the compiler from hoisting the load out of spin loops.
    volatile bool orchestrator_done_{false};
    std::atomic<bool> completed_{false};
    uint64_t *func_id_to_addr_{nullptr};

    // --- Core-transition coordination ---
    std::atomic<bool> transition_requested_{false};
    std::atomic<int32_t> wait_reassign_{0};
    std::atomic<bool> reassigned_{false};

    // --- Thread/core configuration ---
    int32_t active_sched_threads_{0};
    int32_t sched_thread_num_{0};
    bool orch_to_sched_{false};
    int32_t aicpu_thread_num_{0};
    int32_t cores_total_num_{0};

    // Cluster-ordered worker_id lists, populated by handshake_all_cores().
    int32_t aic_worker_ids_[RUNTIME_MAX_WORKER]{};
    int32_t aiv_worker_ids_[RUNTIME_MAX_WORKER]{};
    int32_t aic_count_{0};
    int32_t aiv_count_{0};

    // Platform AICore-register base array (set by AicpuExecutor before init()).
    uint64_t regs_{0};

    // --- One-time init coordination ---
    std::atomic<bool> pto2_init_done_{false};
    std::atomic<bool> pto2_init_complete_{false};

    // =========================================================================
    // Core management (scheduler_cold_path.cpp)
    // =========================================================================

    // Handshake with all AICore workers; populates core_exec_states_, worker id lists.
    int32_t handshake_all_cores(Runtime *runtime);

    // Assign discovered cores (cluster = 1 AIC + 2 AIV) round-robin across scheduler threads.
    bool assign_cores_to_threads();

    // Re-distribute all cores across all threads after orchestration completes.
    void reassign_cores_for_all_threads();

    // Emergency shutdown: broadcast exit signal to every handshake'd core and
    // deinit their AICore register blocks. Idempotent.
    void emergency_shutdown(Runtime *runtime);

    // =========================================================================
    // Dispatch (scheduler_dispatch.cpp)
    // =========================================================================

    static const char *shape_name(PTO2ResourceShape shape) {
        switch (shape) {
        case PTO2ResourceShape::AIC:
            return "AIC";
        case PTO2ResourceShape::AIV:
            return "AIV";
        case PTO2ResourceShape::MIX:
            return "MIX";
        case PTO2ResourceShape::DUMMY:
            return "DUMMY";
        }
        return "UNKNOWN";
    }

    // Lower-case rendering of PTO2SubtaskSlot, used by dispatch and stall logs.
    // Kept lower-case to match the `kernels=[aic:N aiv0:N aiv1:N]` field
    // convention already established in the stall log family.
    static inline const char *subslot_name(PTO2SubtaskSlot s) {
        switch (s) {
        case PTO2SubtaskSlot::AIC:
            return "aic";
        case PTO2SubtaskSlot::AIV0:
            return "aiv0";
        case PTO2SubtaskSlot::AIV1:
            return "aiv1";
        }
        return "?";
    }

    int pop_ready_tasks_batch(
        PTO2ResourceShape shape, int32_t thread_idx, PTO2LocalReadyBuffer &local_buf, PTO2TaskSlotState **out,
        int max_count
    )
    {
        (void)thread_idx;
        int count = sched_->get_ready_tasks_batch(shape, local_buf, out, max_count);
        return count;
    }

    void build_payload(
        PTO2DispatchPayload &dispatch_payload, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot,
        const AsyncCtx &async_ctx, int32_t block_idx
    )
    {
        int32_t slot_idx = static_cast<int32_t>(subslot);
        uint64_t callable_addr = get_function_bin_addr(slot_state.task->kernel_id[slot_idx]);
        const CoreCallable *callable = reinterpret_cast<const CoreCallable *>(callable_addr);
        dispatch_payload.function_bin_addr = callable->resolved_addr();
        auto &payload = *slot_state.payload;
        int n = 0;
        for (int32_t i = 0; i < payload.tensor_count; i++) {
            dispatch_payload.args[n++] = reinterpret_cast<uint64_t>(&payload.tensors[i]);
        }
        for (int32_t i = 0; i < payload.scalar_count; i++) {
            dispatch_payload.args[n++] = payload.scalars[i];
        }
        dispatch_payload.local_context.block_idx = block_idx;
        dispatch_payload.local_context.block_num = slot_state.logical_block_num;
        dispatch_payload.local_context.async_ctx = async_ctx;
        dispatch_payload.args[PAYLOAD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&dispatch_payload.local_context);
        dispatch_payload.args[PAYLOAD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&dispatch_payload.global_context);
    }

    void dispatch_subtask_to_core(
        int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot,
        bool to_pending, int32_t block_idx
    )
    {
        CoreTracker &tracker = core_trackers_[thread_idx];
        auto core_id = tracker.get_core_id_by_offset(core_offset);
        CoreExecState &core_exec_state = core_exec_states_[core_id];
        core_exec_state.dispatch_seq++;
        uint32_t reg_task_id = core_exec_state.dispatch_seq & TASK_ID_MASK;
        static_assert(
            (TASK_ID_MASK - AICORE_EXIT_SIGNAL + 1) % 2 == 0, "Sentinel skip must be even to preserve dual-buffer parity"
        );
        if (reg_task_id >= AICORE_EXIT_SIGNAL) {
            core_exec_state.dispatch_seq += (TASK_ID_MASK - reg_task_id + 1);
            reg_task_id = core_exec_state.dispatch_seq & TASK_ID_MASK;
        }

        uint32_t buf_idx = reg_task_id & 1u;
        PTO2DispatchPayload &payload = payload_per_core_[core_id][buf_idx];
        DeferredCompletionSlab *deferred_slab = &deferred_slab_per_core_[core_id][buf_idx];
        deferred_slab->count = 0;
        deferred_slab->error_code = PTO2_ERROR_NONE;
        AsyncCtx async_ctx = AsyncCtx::make(slot_state.task->task_id, deferred_slab);
        build_payload(payload, slot_state, subslot, async_ctx, block_idx);

        if (to_pending) {
            core_exec_state.pending_subslot = subslot;
            core_exec_state.pending_slot_state = &slot_state;
            core_exec_state.pending_reg_task_id = static_cast<int32_t>(reg_task_id);
        } else {
            core_exec_state.running_subslot = subslot;
            core_exec_state.running_slot_state = &slot_state;
            core_exec_state.running_reg_task_id = static_cast<int32_t>(reg_task_id);
            tracker.change_core_state(core_offset);
        }

        LOG_DEBUG(
            "Thread %d: Dispatched %s %s task %" PRId64 " kernel_id=[%d,%d,%d] block_idx=%d/total_blocks=%d to"
            " core_offset=%d core_id=%d reg_task_id=%u",
            thread_idx, to_pending ? "pending" : "idle", subslot_name(subslot),
            static_cast<int64_t>(slot_state.task->task_id.raw), slot_state.task->kernel_id[0],
            slot_state.task->kernel_id[1], slot_state.task->kernel_id[2], block_idx, slot_state.logical_block_num,
            core_offset, core_id, reg_task_id
        );

        write_reg(core_exec_state.reg_addr, RegId::DATA_MAIN_BASE, static_cast<uint64_t>(reg_task_id));
        tracker.set_pending_occupied(core_offset);
    }

    void dispatch_mix_block_to_cluster(
        int32_t thread_idx, int32_t cluster_offset, PTO2TaskSlotState &slot_state, bool to_pending, int32_t block_idx
    )
    {
        CoreTracker &tracker = core_trackers_[thread_idx];
        uint8_t cmask = slot_state.active_mask.core_mask();
        if (cmask & PTO2_SUBTASK_MASK_AIC) {
            bool aic_to_pending = to_pending && !tracker.is_aic_core_idle(cluster_offset);
            dispatch_subtask_to_core(
                thread_idx, tracker.get_aic_core_offset(cluster_offset), slot_state, PTO2SubtaskSlot::AIC, aic_to_pending,
                block_idx
            );
        }
        if (cmask & PTO2_SUBTASK_MASK_AIV0) {
            bool aiv0_to_pending = to_pending && !tracker.is_aiv0_core_idle(cluster_offset);
            dispatch_subtask_to_core(
                thread_idx, tracker.get_aiv0_core_offset(cluster_offset), slot_state, PTO2SubtaskSlot::AIV0,
                aiv0_to_pending, block_idx
            );
        }
        if (cmask & PTO2_SUBTASK_MASK_AIV1) {
            bool aiv1_to_pending = to_pending && !tracker.is_aiv1_core_idle(cluster_offset);
            dispatch_subtask_to_core(
                thread_idx, tracker.get_aiv1_core_offset(cluster_offset), slot_state, PTO2SubtaskSlot::AIV1,
                aiv1_to_pending, block_idx
            );
        }
    }

    void dispatch_block(
        int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state, PTO2ResourceShape shape,
        bool to_pending, int32_t block_idx
    )
    {
        if (shape == PTO2ResourceShape::MIX) {
            dispatch_mix_block_to_cluster(thread_idx, core_offset, slot_state, to_pending, block_idx);
        } else if (shape == PTO2ResourceShape::AIC) {
            dispatch_subtask_to_core(thread_idx, core_offset, slot_state, PTO2SubtaskSlot::AIC, to_pending, block_idx);
        } else {
            dispatch_subtask_to_core(thread_idx, core_offset, slot_state, PTO2SubtaskSlot::AIV0, to_pending, block_idx);
        }
    }

    void dispatch_shape(
        int32_t thread_idx, PTO2ResourceShape shape, CoreTracker::DispatchPhase phase, PTO2LocalReadyBuffer &local_buf,
        CoreTracker &tracker, bool &entered_drain, bool &made_progress, bool &try_pushed
    )
    {
        if (entered_drain) return;

        bool is_pending = (phase == CoreTracker::DispatchPhase::PENDING);
        auto cores = tracker.get_dispatchable_cores(shape, phase);
        if (!cores.has_value()) return;

        while (cores.has_value() && !entered_drain) {
            int want = cores.count();
            PTO2TaskSlotState *batch[CoreTracker::MAX_CLUSTERS * 3];
            int got = pop_ready_tasks_batch(shape, thread_idx, local_buf, batch, want);
            if (got == 0) break;

            bool dispatched_any = false;
            for (int bi = 0; bi < got; bi++) {
                PTO2TaskSlotState *slot_state = batch[bi];

                if (slot_state->active_mask.requires_sync_start()) {
                    if (is_pending) {
                        sched_->ready_queues[static_cast<int32_t>(shape)].push(slot_state);
                        continue;
                    }
                    int32_t available = cores.count();
                    if (available < slot_state->logical_block_num) {
                        if (!enter_drain_mode(slot_state, slot_state->logical_block_num)) {
                            sched_->ready_queues[static_cast<int32_t>(shape)].push(slot_state);
                        }
                        for (int rem = bi + 1; rem < got; rem++) {
                            sched_->ready_queues[static_cast<int32_t>(shape)].push(batch[rem]);
                        }
                        entered_drain = true;
                        break;
                    }
                }

                if (!cores.has_value()) {
                    sched_->ready_queues[static_cast<int32_t>(shape)].push_batch(&batch[bi], got - bi);
                    break;
                }

                dispatched_any = true;
                try_pushed = true;
                // Claim a contiguous range of blocks, hand the slot back to the
                // ready queue immediately, then perform the expensive dispatches.
                // This lets other schedulers concurrently claim and dispatch the
                // remaining blocks of the same SPMD task instead of spinning while
                // this thread fills all its own cores.  Only local `start + b` is
                // read after the push -- `next_block_idx` may already be advanced
                // by another scheduler that popped the slot.
                int32_t remaining = slot_state->logical_block_num - slot_state->next_block_idx;
                int32_t claim = std::min(cores.count(), remaining);
                int32_t start = slot_state->next_block_idx;
                slot_state->next_block_idx += claim;

                if (slot_state->next_block_idx < slot_state->logical_block_num) {
                    sched_->ready_queues[static_cast<int32_t>(shape)].push(slot_state);
                }

                for (int32_t b = 0; b < claim; b++) {
                    auto core_offset = cores.pop_first();
                    dispatch_block(thread_idx, core_offset, *slot_state, shape, is_pending, start + b);
                }
                made_progress = true;
            }

            if (!dispatched_any) break;

            if (!cores.has_value()) {
                cores = tracker.get_dispatchable_cores(shape, phase);
            }
        }
    }

    // One pass of "Phase 4" in the resolve_and_dispatch loop: IDLE-stage dispatch
    // for MIX then (if no mix residual) AIC/AIV; mid-flush of local buffers; then
    // PENDING-stage dispatch with cross-thread idle gating. MIX is strictly
    // prioritized — when mix residual is detected after MIX-IDLE, AIC/AIV are
    // skipped for the whole pass but MIX-PENDING still runs.
    //
    // Forward-progress argument for AIC/AIV: skip_aic_aiv is sticky for the
    // current pass only. The next loop iteration re-evaluates after Phase 1
    // completion polling and the global MIX queue draining (here or on any
    // peer thread). AIC/AIV starvation is therefore bounded by MIX throughput,
    // not unbounded — once mix completes on at least one cluster, the next
    // pass either drains the residual or admits AIC/AIV.
    void dispatch_ready_tasks(
        int32_t thread_idx, CoreTracker &tracker, PTO2LocalReadyBuffer (&local_bufs)[PTO2_NUM_RESOURCE_SHAPES],
        bool pmu_active, bool &made_progress, bool &try_pushed
    )
    {
        using Phase = CoreTracker::DispatchPhase;
        constexpr int32_t MIX_I = static_cast<int32_t>(PTO2ResourceShape::MIX);

        // MIX is handled explicitly at the top of each stage; only AIC/AIV cycle
        // through this 2-elem array, with order toggled by thread parity for
        // shape-level load balancing across threads.
        static constexpr PTO2ResourceShape kAicAivOrder[2][2] = {
            {PTO2ResourceShape::AIC, PTO2ResourceShape::AIV},
            {PTO2ResourceShape::AIV, PTO2ResourceShape::AIC},
        };
        const PTO2ResourceShape *aic_aiv = kAicAivOrder[thread_idx & 1];

        // Note: flush_local_bufs is invoked multiple times per pass (mid-function
        // flush + RAII tail flush). local_overflow_count accumulates each batch
        // separately — each entry is counted exactly once (count is zeroed after
        // push_batch). The total reflects "entries this pass pushed to the global
        // queue", which is slightly larger than the pre-refactor "buf residual at
        // pass end" semantics — comparing PTO2_SCHED_PROFILING traces across
        // commits, expect the post-refactor number to be greater-or-equal.
        auto flush_local_bufs = [&]() {
            for (int32_t s = 0; s < PTO2_NUM_RESOURCE_SHAPES; s++) {
                auto &lb = local_bufs[s];
                if (lb.count > 0) {
                    sched_->ready_queues[s].push_batch(lb.slot_states, lb.count);
                    lb.count = 0;
                }
            }
        };
        // Every return path below must flush; wrap in RAII so we cannot forget.
        // The mid-function flush between IDLE and PENDING is still called
        // explicitly — guard only covers exit.
        struct FlushGuard {
            decltype(flush_local_bufs) &flush_fn;
            ~FlushGuard() { flush_fn(); }
        } flush_guard{flush_local_bufs};

        bool entered_drain = false;

        // ===== IDLE stage =====
        dispatch_shape(
            thread_idx, PTO2ResourceShape::MIX, Phase::IDLE, local_bufs[MIX_I], tracker, entered_drain, made_progress,
            try_pushed
        );
        if (entered_drain) return;

        // MIX-IDLE residual: AIC/AIV (both IDLE and PENDING) yield for this pass.
        // MIX-PENDING below still runs — that is the core of "mix strict priority":
        // pending slots are spent on mix before AIC/AIV get any chance.
        bool skip_aic_aiv = has_residual_mix(local_bufs[MIX_I]);

        if (!skip_aic_aiv) {
            for (int i = 0; i < 2; i++) {
                PTO2ResourceShape s = aic_aiv[i];
                dispatch_shape(
                    thread_idx, s, Phase::IDLE, local_bufs[static_cast<int32_t>(s)], tracker, entered_drain, made_progress,
                    try_pushed
                );
                if (entered_drain) return;
            }
        }

        // Flush between IDLE and PENDING so PENDING-stage queue-size checks and any
        // peer-thread reads see the IDLE-stage release_fanin output.
        flush_local_bufs();

        if (pmu_active) return;

        // ===== PENDING stage =====
        // MIX-PENDING gate: skip when a peer has an idle MIX-capable cluster — that
        // peer's next IDLE-MIX iteration will pull the mix task from the global
        // queue (already flushed above) at lower latency than us pre-loading a
        // pending slot here. Forward progress for MIX is preserved: at least one
        // thread will run MIX-IDLE next pass and consume the residual.
        //
        // The gate is NOT subject to skip_aic_aiv — residual mix continues to drain
        // via pending slots on this thread when no peer is idle.
        if (!has_idle_in_other_threads(thread_idx, PTO2ResourceShape::MIX)) {
            dispatch_shape(
                thread_idx, PTO2ResourceShape::MIX, Phase::PENDING, local_bufs[MIX_I], tracker, entered_drain,
                made_progress, try_pushed
            );
            if (entered_drain) return;
        }

        // Re-check after MIX-PENDING. If MIX-IDLE already set skip_aic_aiv, leave
        // it set; otherwise, escalate iff PENDING-MIX left residual.
        if (!skip_aic_aiv && has_residual_mix(local_bufs[MIX_I])) {
            skip_aic_aiv = true;
        }

        // PENDING-MIX may have re-populated AIC/AIV local_bufs via release_fanin
        // during in-flight completions; flush_guard ensures these don't carry
        // across to the next iteration's IDLE stage.
        if (skip_aic_aiv) return;

        // AIC/AIV-PENDING gate: a peer-idle skip is a delay, not a loss — the peer
        // will pull from the global queue on its next IDLE pass.
        for (int i = 0; i < 2; i++) {
            PTO2ResourceShape s = aic_aiv[i];
            if (has_idle_in_other_threads(thread_idx, s)) continue;
            dispatch_shape(
                thread_idx, s, Phase::PENDING, local_bufs[static_cast<int32_t>(s)], tracker, entered_drain, made_progress,
                try_pushed
            );
            if (entered_drain) return;
        }
    }

    // Returns true if any *other* scheduler thread currently has an idle core
    // matching `shape`. Used as a scheduling hint on the PENDING dispatch path
    // — see the implementation in scheduler_dispatch.cpp for the hint-semantics
    // rationale and the safety argument against the drain worker.
    bool has_idle_in_other_threads(int32_t self_thread_idx, PTO2ResourceShape shape) const
    {
        // Cross-thread read of peer trackers without explicit synchronization. The
        // backing `core_states_` is a naturally aligned uint64_t; aarch64 guarantees
        // single-copy atomicity for an 8-byte aligned load, so no torn read. The
        // value is consumed only as a scheduling *hint* — a stale read at worst
        // causes one missed/extra pending dispatch, corrected on the next iteration.
        // Drain-mode cross-thread writes are serialized by handle_drain_mode's ack
        // barrier (all peers spin out of the dispatch path before any tracker
        // mutation), so this routine is never racing the drain worker.
        for (int32_t t = 0; t < active_sched_threads_; t++) {
            if (t == self_thread_idx) continue;
            if (core_trackers_[t].get_idle_core_offset_states(shape).has_value()) {
                return true;
            }
        }
        return false;
    }

    // True if mix tasks remain anywhere this thread could see them: the caller's
    // MIX local LIFO stack or the global MIX ready queue. Approximate —
    // PTO2ReadyQueue::size() (see pto_scheduler.h) snapshots its enqueue/dequeue
    // positions with std::memory_order_relaxed and may interleave with concurrent
    // push/pop. Don't confuse with PTO2SpscQueue::size(), which uses acquire
    // loads — that one isn't on this path. A stale read here causes at most one
    // extra/missed AIC/AIV skip and self-corrects on the next loop iteration.
    bool has_residual_mix(const PTO2LocalReadyBuffer &mix_local_buf) const {
        return mix_local_buf.count > 0 || sched_->ready_queues[static_cast<int32_t>(PTO2ResourceShape::MIX)].size() > 0;
    }

    // =========================================================================
    // Completion & drain (scheduler_completion.cpp)
    // =========================================================================

    static SlotTransition
    decide_slot_transition(int32_t reg_task_id, int32_t reg_state, int32_t running_id, int32_t pending_id)
    {
        SlotTransition t;
        if (pending_id != AICPU_TASK_INVALID && reg_task_id == pending_id) {
            t.matched = true;
            t.running_done = true;  // Serial execution: pending event implies running done
            t.running_freed = true;
            t.pending_freed = true;
            if (reg_state == TASK_FIN_STATE) {
                t.pending_done = true;  // Case 1: pending FIN
            }
            // else: Case 2: pending ACK (pending_done stays false)
        } else if (reg_task_id == running_id) {
            if (reg_state == TASK_FIN_STATE) {
                if (pending_id == AICPU_TASK_INVALID) {
                    // Case 3.2: running FIN, no pending -> core goes idle
                    t.matched = true;
                    t.running_done = true;
                    t.running_freed = true;
                }
                // Case 3.1: running FIN, pending exists -> skip (transient state).
                // Case 1/2 (pending ACK/FIN) will complete running implicitly via running_done=true.
            } else {
                // Case 4: running ACK -- only pending_freed (slot now hardware-latched)
                t.matched = true;
                t.pending_freed = true;
            }
        }
        return t;
    }

    void complete_slot_task(
        PTO2TaskSlotState &slot_state, int32_t expected_reg_task_id, [[maybe_unused]] PTO2SubtaskSlot subslot, int32_t thread_idx,
        int32_t core_id, [[maybe_unused]] Handshake *hank, int32_t &completed_this_turn,
        PTO2TaskSlotState *deferred_release_slot_states[], int32_t &deferred_release_count,
        PTO2LocalReadyBuffer *local_bufs
    )
    {
        bool mixed_complete = sched_->on_subtask_complete(slot_state);
        if (slot_state.payload != nullptr) {
            int32_t reg_err = PTO2_ERROR_NONE;
            AsyncWaitList::RegisterResult reg_result;
            volatile DeferredCompletionSlab *deferred_slab = &deferred_slab_per_core_[core_id][expected_reg_task_id & 1];
            AsyncCtx async_ctx = AsyncCtx::make(slot_state.task->task_id, deferred_slab);
            do {
                reg_result = sched_->async_wait_list.register_deferred(slot_state, async_ctx, mixed_complete, reg_err);
                if (reg_result == AsyncWaitList::RegisterResult::Skipped) {
                    SPIN_WAIT_HINT();
                }
            } while (reg_result == AsyncWaitList::RegisterResult::Skipped);

            if (reg_result == AsyncWaitList::RegisterResult::Error) {
                int32_t expected = PTO2_ERROR_NONE;
                sched_->sm_header->sched_error_code.compare_exchange_strong(
                    expected, reg_err, std::memory_order_acq_rel, std::memory_order_acquire
                );
                completed_.store(true, std::memory_order_release);
                return;
            }

            if (mixed_complete && reg_result == AsyncWaitList::RegisterResult::Registered) {
                return;
            }
        }
        if (mixed_complete) {
            sched_->on_mixed_task_complete(slot_state, local_bufs);
            if (deferred_release_count < PTO2_DEFERRED_RELEASE_CAP) {
                deferred_release_slot_states[deferred_release_count++] = &slot_state;
            } else {
                LOG_INFO_V9("Thread %d: release", thread_idx);
                while (deferred_release_count > 0) {
                    sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count]);
                }
                deferred_release_slot_states[deferred_release_count++] = &slot_state;
            }
            completed_this_turn++;
        }
    }

    static void promote_pending_to_running(CoreExecState &core)
    {
        core.running_slot_state = core.pending_slot_state;
        core.running_reg_task_id = core.pending_reg_task_id;
        core.running_subslot = core.pending_subslot;
        core.pending_slot_state = nullptr;
        core.pending_reg_task_id = AICPU_TASK_INVALID;
    }

    static void clear_running_slot(CoreExecState &core)
    {
        core.running_slot_state = nullptr;
        core.running_reg_task_id = AICPU_TASK_INVALID;
    }

    void check_running_cores_for_completion(
        int32_t thread_idx, Handshake *hank, int32_t &completed_this_turn, int32_t &cur_thread_completed,
        bool &made_progress, PTO2TaskSlotState *deferred_release_slot_states[], int32_t &deferred_release_count,
        PTO2LocalReadyBuffer *local_bufs
    )
    {
        CoreTracker &tracker = core_trackers_[thread_idx];
        auto running_core_states = tracker.get_all_running_cores();
        while (running_core_states.has_value()) {
            int32_t bit_pos = running_core_states.pop_first();
            int32_t core_id = tracker.get_core_id_by_offset(bit_pos);
            CoreExecState &core = core_exec_states_[core_id];

            // --- Judgment phase: read register, derive transition ---
            uint64_t reg_val = read_reg(core.reg_addr, RegId::COND);
            int32_t reg_task_id = EXTRACT_TASK_ID(reg_val);
            int32_t reg_state = EXTRACT_TASK_STATE(reg_val);

            SlotTransition t =
                decide_slot_transition(reg_task_id, reg_state, core.running_reg_task_id, core.pending_reg_task_id);
            if (!t.matched) continue;

            // --- Apply phase: execute actions based on transition ---

            // 1. Complete finished tasks (capture pointers before modifying core state)
            if (t.pending_done) {
                complete_slot_task(
                    *core.pending_slot_state, core.pending_reg_task_id, core.pending_subslot, thread_idx, core_id, hank,
                    completed_this_turn, deferred_release_slot_states, deferred_release_count, local_bufs
                );
                cur_thread_completed++;
            }
            if (t.running_done) {
                complete_slot_task(
                    *core.running_slot_state, core.running_reg_task_id, core.running_subslot, thread_idx, core_id, hank,
                    completed_this_turn, deferred_release_slot_states, deferred_release_count, local_bufs
                );
                cur_thread_completed++;
            }

            // 2. Update slot data
            if (t.running_freed) {
                if (core.pending_slot_state != nullptr && !t.pending_done) {
                    promote_pending_to_running(core);  // Case 2 or Case 3 (with pending)
                } else {
                    clear_running_slot(core);  // Case 1 or Case 3 (no pending)
                    if (t.pending_done) {
                        // Case 1: pending FIN observed directly -- clear stale pending fields.
                        // Without this, pending_reg_task_id retains a stale value that blocks
                        // clear_pending_occupied and permanently degrades pipelining.
                        core.pending_slot_state = nullptr;
                        core.pending_reg_task_id = AICPU_TASK_INVALID;
                    }
                }
            }

            // 3. Update tracker bitmap
            bool is_idle = (core.running_reg_task_id == AICPU_TASK_INVALID);
            if (is_idle) {
                tracker.change_core_state(bit_pos);       // Mark idle
                tracker.clear_pending_occupied(bit_pos);  // Idle safeguard: no payload to protect
            } else if (t.pending_freed && core.pending_reg_task_id == AICPU_TASK_INVALID) {
                // Case 4 (running ACK) or Case 2 (pending ACK): clear pending_occupied only
                // when no pending task is currently held. Otherwise pending slot is occupied
                // by a pre-loaded task and must stay protected.
                tracker.clear_pending_occupied(bit_pos);
            }

            // 4. Progress signal (only when running task completes)
            if (t.running_done) {
                made_progress = true;
            }
        }
    }

    bool enter_drain_mode(PTO2TaskSlotState *slot_state, int32_t block_num)
    {
        int32_t expected = 0;
        if (!drain_state_.sync_start_pending.compare_exchange_strong(
                expected, -1, std::memory_order_relaxed, std::memory_order_relaxed
            )) {
            return false;  // Another thread already holds the drain slot.
        }
        // We own the drain slot.  Store the task and reset election flag before making it visible.
        drain_state_.pending_task = slot_state;
        drain_state_.drain_ack_mask.store(0, std::memory_order_relaxed);
        drain_state_.drain_worker_elected.store(0, std::memory_order_relaxed);
        // Release store: all stores above are now visible to any thread that
        // acquire-loads sync_start_pending and sees block_num > 0.
        drain_state_.sync_start_pending.store(block_num, std::memory_order_release);
        return true;
    }

    int32_t count_global_available(PTO2ResourceShape shape)
    {
        int32_t total = 0;
        for (int32_t t = 0; t < active_sched_threads_; t++) {
            total += core_trackers_[t].get_idle_core_offset_states(shape).count();
        }
        return total;
    }

    void drain_worker_dispatch(int32_t block_num)
    {
        PTO2TaskSlotState *slot_state = drain_state_.pending_task;
        if (!slot_state) {
            drain_state_.sync_start_pending.store(0, std::memory_order_release);
            return;
        }
        PTO2ResourceShape shape = slot_state->active_mask.to_shape();

        for (int32_t t = 0; t < active_sched_threads_ && slot_state->next_block_idx < block_num; t++) {
            auto valid = core_trackers_[t].get_idle_core_offset_states(shape);
            while (valid.has_value() && slot_state->next_block_idx < block_num) {
                dispatch_block(t, valid.pop_first(), *slot_state, shape, false, slot_state->next_block_idx);
                slot_state->next_block_idx++;
            }
        }

        // All blocks dispatched -- clear drain state.
        // Release fence ensures tracker mutations are visible to threads that
        // acquire-load sync_start_pending == 0 and resume normal operation.
        std::atomic_thread_fence(std::memory_order_release);
        drain_state_.pending_task = nullptr;
        drain_state_.drain_ack_mask.store(0, std::memory_order_relaxed);
        drain_state_.drain_worker_elected.store(0, std::memory_order_relaxed);
        drain_state_.sync_start_pending.store(0, std::memory_order_release);
    }

    void handle_drain_mode(int32_t thread_idx)
    {
        // Spin until drain is fully initialized (sentinel -1 -> block_num > 0).
        int32_t block_num;
        do {
            block_num = drain_state_.sync_start_pending.load(std::memory_order_acquire);
        } while (block_num < 0);
        if (block_num == 0) return;

        uint32_t all_acked = (1u << active_sched_threads_) - 1;

        // Ack barrier -- signal this thread has stopped dispatch.
        drain_state_.drain_ack_mask.fetch_or(1u << thread_idx, std::memory_order_release);

        // Spin until all threads have acked.
        // If our bit is cleared while waiting, elected reset due to insufficient resources.
        while (true) {
            uint32_t ack = drain_state_.drain_ack_mask.load(std::memory_order_acquire);
            if ((ack & all_acked) == all_acked) break;
            if ((ack & (1u << thread_idx)) == 0) return;
            SPIN_WAIT_HINT();
        }

        // Election -- exactly one thread wins the CAS.
        int32_t expected = 0;
        drain_state_.drain_worker_elected.compare_exchange_strong(
            expected, thread_idx + 1, std::memory_order_acquire, std::memory_order_relaxed
        );

        if (drain_state_.drain_worker_elected.load(std::memory_order_relaxed) != thread_idx + 1) {
            // Non-elected: spin-wait for drain completion or resource-insufficient reset.
            while (drain_state_.sync_start_pending.load(std::memory_order_acquire) != 0) {
                if (drain_state_.drain_worker_elected.load(std::memory_order_acquire) == 0) return;
                SPIN_WAIT_HINT();
            }
            return;
        }

        // Elected: check if global resources are sufficient.
        PTO2TaskSlotState *slot_state = drain_state_.pending_task;
        PTO2ResourceShape shape = slot_state->active_mask.to_shape();
        int32_t available = count_global_available(shape);

        if (available < block_num) {
            // Insufficient resources -- reset drain fields so threads can resume
            // completion polling to free running cores, then retry.
            drain_state_.drain_ack_mask.store(0, std::memory_order_release);
            drain_state_.drain_worker_elected.store(0, std::memory_order_release);
            return;
        }

        // Dispatch -- all other threads are spinning, elected thread has exclusive tracker access.
        drain_worker_dispatch(block_num);
    }


    // =========================================================================
    // Cold path: exit checks, stall diagnostics, profiling (scheduler_cold_path.cpp)
    // =========================================================================

    __attribute__((noinline, cold)) LoopAction
    handle_orchestrator_exit(int32_t thread_idx, PTO2SharedMemoryHeader *header, Runtime *runtime, int32_t &task_count);

    __attribute__((noinline, cold)) LoopAction handle_core_transition(bool &cores_released);

    __attribute__((noinline, cold)) LoopAction
    check_idle_fatal_error(int32_t thread_idx, PTO2SharedMemoryHeader *header, Runtime *runtime);

    __attribute__((noinline, cold)) void
    log_stall_diagnostics(int32_t thread_idx, int32_t task_count, int32_t idle_iterations, int32_t last_progress_count);

    // Reverse lookup: given a global core_id, find which scheduler thread's
    // tracker owns it. Returns -1 if not found. Linear scan — only used on
    // the cold diagnostic path.
    int32_t find_core_owner_thread(int32_t core_id) const;

    __attribute__((noinline, cold)) int32_t handle_timeout_exit(
        int32_t thread_idx, PTO2SharedMemoryHeader *header, Runtime *runtime, int32_t idle_iterations
    );

    // =========================================================================
    // Small inline helpers
    // =========================================================================

    uint64_t get_function_bin_addr(int func_id) const {
        if (!func_id_to_addr_ || func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) {
            LOG_ERROR("func_id=%d is out of range [0, %d) or map is null", func_id, RUNTIME_MAX_FUNC_ID);
            return 0;
        }
        return func_id_to_addr_[func_id];
    }
};

#endif  // SCHEDULER_CONTEXT_H
