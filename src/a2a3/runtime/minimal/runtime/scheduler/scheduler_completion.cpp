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
#include "scheduler_context.h"

#include "common/unified_log.h"
#include "aicpu/device_time.h"
#include "aicpu/platform_regs.h"
#include "common/l2_perf_profiling.h"
#include "common/platform_config.h"
#include "pto_runtime2.h"
#include "runtime.h"
#include "spin_hint.h"

// Performance profiling headers
#include "aicpu/l2_perf_collector_aicpu.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "aicpu/tensor_dump_aicpu.h"


// =============================================================================
// sync_start drain protocol
// =============================================================================

// Take ownership of slot_state and signal all threads to enter drain mode.
// Returns true if this thread won the CAS and owns the drain slot.
// Returns false if another thread already holds drain; caller must re-push slot_state.
//
// Two-phase protocol: CAS 0 -> -1 (sentinel) to claim ownership, store task and
// reset election flag, then release-store block_num.  Other threads acquire-load
// sync_start_pending; seeing block_num > 0 ensures all relaxed stores are visible.
bool SchedulerContext::enter_drain_mode(PTO2TaskSlotState *slot_state, int32_t block_num) {
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

// Count total available resources across all scheduler threads for a given shape.
int32_t SchedulerContext::count_global_available(PTO2ResourceShape shape) {
    int32_t total = 0;
    for (int32_t t = 0; t < active_sched_threads_; t++) {
        total += core_trackers_[t].get_idle_core_offset_states(shape).count();
    }
    return total;
}

// Drain worker: dispatch all blocks in one pass across all threads' trackers.
// Called only when global resources >= block_num, so one pass always suffices.
// All other threads are spinning -- the drain worker has exclusive tracker access.
void SchedulerContext::drain_worker_dispatch(int32_t block_num) {
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

// Called by each scheduler thread when drain_state_.sync_start_pending != 0.
//
// Protocol (single-stage ack barrier):
//   1. Ack barrier: all threads signal they've stopped dispatch, then spin
//      until all ack bits are set.
//      If this thread's bit gets cleared while waiting, a reset occurred -- return.
//   2. Election: one thread wins the CAS and becomes the drain worker.
//      If resources are insufficient, reset ack/election fields and return --
//      all threads resume completion polling to free running cores, then retry.
//   3. Dispatch: elected thread dispatches all blocks (one pass, resources guaranteed).
//      Non-elected threads spin-wait until sync_start_pending == 0.
//      During dispatch the elected thread has exclusive tracker access.
void SchedulerContext::handle_drain_mode(int32_t thread_idx) {
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
