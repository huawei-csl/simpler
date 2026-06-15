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

#include "aicpu/platform_regs.h"
#include "common/l2_swimlane_profiling.h"
#include "scheduler_types.h"

#include "scheduler/pto_scheduler.h"

#include "aicore_completion_mailbox.h"
#include "pto2_dispatch_payload.h"

#ifndef RUNTIME_MAX_WORKER
#define RUNTIME_MAX_WORKER 72
#endif
#ifndef RUNTIME_MAX_FUNC_ID
#define RUNTIME_MAX_FUNC_ID 1024
#endif

// Forward declarations — avoid pulling in full headers for pointer/reference params.
class Runtime;
struct Handshake;
struct PTO2Runtime;

class SchedulerContext
{
public:
    int32_t init(Runtime *runtime, int32_t aicpu_thread_num, int32_t sched_thread_num, bool orch_to_sched, uint64_t regs_base);

    // Reset all SchedulerContext-owned state to its post-construction defaults.
    // Called by AicpuExecutor::deinit() during per-run teardown.
    void deinit();

    // Main scheduler thread entry: poll completion + dispatch ready tasks.
    int32_t resolve_and_dispatch(Runtime *runtime, int32_t thread_idx);

    int32_t shutdown(int32_t thread_idx);

    void on_orchestration_done(Runtime *runtime, PTO2Runtime *rt, int32_t total_tasks);

    // Bind the PTO2Runtime scheduler pointer. Required in device-orchestration
    // mode where rt is created by the orchestrator thread after init().
    void bind_runtime(PTO2Runtime *rt);

    int32_t aic_count() const
    {
        return aic_count_;
    }
    int32_t aiv_count() const
    {
        return aiv_count_;
    }
    bool is_completed() const
    {
        return completed_.load(std::memory_order_acquire);
    }
    int32_t completed_tasks_count() const
    {
        return completed_tasks_.load(std::memory_order_acquire);
    }

    // Block until the first scheduler thread has finished one-time PTO2 init.
    // Called by the orchestrator thread in device-orch mode.
    void wait_pto2_init_complete() const;

private:
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

    // Handshake with all AICore workers; populates core_exec_states_, worker id lists.
    int32_t handshake_all_cores(Runtime *runtime);

    // Assign discovered cores (cluster = 1 AIC + 2 AIV) round-robin across scheduler threads.
    bool assign_cores_to_threads();

    // Re-distribute all cores across all threads after orchestration completes.
    void reassign_cores_for_all_threads();

    // Emergency shutdown: broadcast exit signal to every handshake'd core and
    // deinit their AICore register blocks. Idempotent.
    void emergency_shutdown(Runtime *runtime);

    static const char *shape_name(PTO2ResourceShape shape);

    static inline const char *subslot_name(PTO2SubtaskSlot s)
    {
        switch (s)
        {
        case PTO2SubtaskSlot::AIC:
            return "aic";
        case PTO2SubtaskSlot::AIV0:
            return "aiv0";
        case PTO2SubtaskSlot::AIV1:
            return "aiv1";
        }
        return "?";
    }

    int pop_ready_tasks_batch(PTO2ResourceShape shape, int32_t thread_idx, PTO2LocalReadyBuffer &local_buf, PTO2TaskSlotState **out, int max_count);

    void build_payload(PTO2DispatchPayload &dispatch_payload, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot, const AsyncCtx &async_ctx, int32_t block_idx);

    struct PublishHandle
    {
        uint64_t reg_addr;
        uint32_t reg_task_id;
        int32_t core_offset;
        uint64_t *dispatch_timestamp_slot;
    };

    PublishHandle prepare_subtask_to_core(int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot, bool to_pending, int32_t block_idx);

    inline void publish_subtask_to_core(const PublishHandle &h, uint64_t dispatch_ts)
    {
        if (h.dispatch_timestamp_slot != nullptr) *h.dispatch_timestamp_slot = dispatch_ts;
        write_reg(h.reg_addr, RegId::DATA_MAIN_BASE, static_cast<uint64_t>(h.reg_task_id));
    }

    // Fan out one block's subtasks (1 for AIC/AIV, 1-3 for MIX) into the
    // caller-supplied handles buffer. Returns the number of handles written.
    int prepare_block_for_dispatch(int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state, PTO2ResourceShape shape, bool to_pending, int32_t block_idx, PublishHandle *out_handles);

    void dispatch_shape(int32_t thread_idx, PTO2ResourceShape shape, CoreTracker::DispatchPhase phase, PTO2LocalReadyBuffer &local_buf, CoreTracker &tracker, bool &entered_drain, bool &made_progress, bool &try_pushed);

    void dispatch_ready_tasks(int32_t thread_idx, CoreTracker &tracker, PTO2LocalReadyBuffer (&local_bufs)[PTO2_NUM_RESOURCE_SHAPES], bool pmu_active, bool &made_progress, bool &try_pushed);

    bool has_idle_in_other_threads(int32_t self_thread_idx, PTO2ResourceShape shape) const;

    bool has_residual_mix(const PTO2LocalReadyBuffer &mix_local_buf) const
    {
        return mix_local_buf.count > 0 || sched_->ready_queues[static_cast<int32_t>(PTO2ResourceShape::MIX)].size() > 0;
    }

    static SlotTransition decide_slot_transition(int32_t reg_task_id, int32_t reg_state, int32_t running_id, int32_t pending_id);

    void complete_slot_task(PTO2TaskSlotState &slot_state, int32_t expected_reg_task_id, int32_t core_id, Handshake *hank, int32_t &completed_this_turn, PTO2TaskSlotState *deferred_release_slot_states[], int32_t &deferred_release_count, PTO2LocalReadyBuffer *local_bufs);

    static void promote_pending_to_running(CoreExecState &core);
    static void clear_running_slot(CoreExecState &core);

    void check_running_cores_for_completion(int32_t thread_idx, Handshake *hank, int32_t &completed_this_turn, int32_t &cur_thread_completed, bool &made_progress, PTO2TaskSlotState *deferred_release_slot_states[], int32_t &deferred_release_count, PTO2LocalReadyBuffer *local_bufs);

    bool enter_drain_mode(PTO2TaskSlotState *slot_state, int32_t block_num);
    int32_t count_global_available(PTO2ResourceShape shape);
    void drain_worker_dispatch(int32_t block_num);
    void handle_drain_mode(int32_t thread_idx);

    __attribute__((noinline, cold)) LoopAction handle_orchestrator_exit(PTO2SharedMemoryHeader *header, Runtime *runtime, int32_t &task_count);

    __attribute__((noinline, cold)) LoopAction handle_core_transition(bool &cores_released);

    __attribute__((noinline, cold)) LoopAction check_idle_fatal_error(PTO2SharedMemoryHeader *header, Runtime *runtime);

    __attribute__((noinline, cold)) void log_stall_diagnostics(int32_t thread_idx, int32_t task_count);

    __attribute__((noinline, cold)) void log_shutdown_stall_snapshot(int32_t trigger_idle_iterations, int32_t trigger_last_progress_count);

    int32_t find_core_owner_thread(int32_t core_id) const;

    bool self_owns_running_task(int32_t thread_idx) const;

    bool no_thread_owns_running_task() const;

    __attribute__((noinline, cold)) int32_t handle_timeout_exit(int32_t thread_idx, PTO2SharedMemoryHeader *header, Runtime *runtime, int32_t idle_iterations, int32_t last_progress_count);

    uint64_t get_function_bin_addr(int func_id) const
    {
        if (!func_id_to_addr_ || func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) return 0;
        return func_id_to_addr_[func_id];
    }
};

#endif  // SCHEDULER_CONTEXT_H
