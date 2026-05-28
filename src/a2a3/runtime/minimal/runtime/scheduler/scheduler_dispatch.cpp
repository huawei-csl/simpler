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

#include <algorithm>
#include <cinttypes>

#include "common.h"  // debug_assert

#include "common/unified_log.h"
#include "aicpu/device_time.h"
#include "aicpu/platform_regs.h"
#include "callable.h"
#include "common/l2_perf_profiling.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "pto_runtime2.h"
#include "runtime.h"
#include "spin_hint.h"

// Performance profiling headers
#include "aicpu/l2_perf_collector_aicpu.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "aicpu/tensor_dump_aicpu.h"

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

// =============================================================================
// Dispatch helpers
// =============================================================================

namespace {
inline constexpr int32_t PTO2_DEFERRED_RELEASE_CAP = 256;
}

void SchedulerContext::dispatch_block(
    int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state, PTO2ResourceShape shape, bool to_pending,
    int32_t block_idx
) {
    if (shape == PTO2ResourceShape::MIX) {
        dispatch_mix_block_to_cluster(thread_idx, core_offset, slot_state, to_pending, block_idx);
    } else if (shape == PTO2ResourceShape::AIC) {
        dispatch_subtask_to_core(thread_idx, core_offset, slot_state, PTO2SubtaskSlot::AIC, to_pending, block_idx);
    } else {
        dispatch_subtask_to_core(thread_idx, core_offset, slot_state, PTO2SubtaskSlot::AIV0, to_pending, block_idx);
    }
}

void SchedulerContext::dispatch_shape(
    int32_t thread_idx, PTO2ResourceShape shape, CoreTracker::DispatchPhase phase, PTO2LocalReadyBuffer &local_buf,
    CoreTracker &tracker, bool &entered_drain, bool &made_progress, bool &try_pushed
) {
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

void SchedulerContext::dispatch_ready_tasks(
    int32_t thread_idx, CoreTracker &tracker, PTO2LocalReadyBuffer (&local_bufs)[PTO2_NUM_RESOURCE_SHAPES],
    bool pmu_active, bool &made_progress, bool &try_pushed
) {
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

// =============================================================================
// Main scheduler dispatch loop
// =============================================================================

int32_t SchedulerContext::resolve_and_dispatch(Runtime *runtime, int32_t thread_idx) {
    always_assert(sched_ != nullptr);
    CoreTracker &tracker = core_trackers_[thread_idx];

    PTO2SharedMemoryHeader *header = sched_->sm_header;
    if (!header) {
        LOG_ERROR("PTO2 dispatch: header is null");
        return -1;
    }

    Handshake *hank = static_cast<Handshake *>(runtime->workers);

    // One-time init: assign perf buffers (one thread does it; others wait)
    if (!pto2_init_done_.exchange(true, std::memory_order_acq_rel)) {
        pto2_init_complete_.store(true, std::memory_order_release);
    } else {
        while (!pto2_init_complete_.load(std::memory_order_acquire)) {
            SPIN_WAIT_HINT();
        }
    }

    int32_t cur_thread_completed = 0;
    int32_t idle_iterations = 0;
    int32_t last_progress_count = 0;

    constexpr int LOCAL_READY_CAP_PER_TYPE = 64;
    PTO2TaskSlotState *local_ptrs[PTO2_NUM_RESOURCE_SHAPES][LOCAL_READY_CAP_PER_TYPE];
    PTO2LocalReadyBuffer local_bufs[PTO2_NUM_RESOURCE_SHAPES];
    for (int32_t i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        local_bufs[i].reset(local_ptrs[i], LOCAL_READY_CAP_PER_TYPE);
    }
    PTO2TaskSlotState *deferred_release_slot_states[PTO2_DEFERRED_RELEASE_CAP];
    int32_t deferred_release_count = 0;

    bool cores_released = false;

    // PMU runs require single-issue dispatch — overlapping in-flight tasks
    // pollute per-task PMU counters, so skip the PENDING pre-load phase.
    // Cached at function scope: is_pmu_enabled() is extern "C" and the
    // compiler cannot hoist it across the dispatch loop on its own.
    const bool pmu_active = is_pmu_enabled();

    while (true) {
        if (completed_.load(std::memory_order_acquire)) {
            break;
        }
        bool made_progress = false;
        int32_t task_count = 0;
        if (!tracker.has_any_running_cores()) {
            LoopAction action = handle_orchestrator_exit(thread_idx, header, runtime, task_count);
            if (action == LoopAction::BREAK_LOOP) break;
        }

        if (!cores_released && orch_to_sched_) {
            LoopAction action = handle_core_transition(cores_released);
            if (action == LoopAction::BREAK_LOOP) break;
        }

        // Phase 1: Check running cores for completion
        int32_t completed_this_turn = 0;

        bool try_completed = tracker.has_any_running_cores();
        if (try_completed) {
            check_running_cores_for_completion(
                thread_idx, hank, completed_this_turn, cur_thread_completed, made_progress,
                deferred_release_slot_states, deferred_release_count, local_bufs
            );
        }
        if (completed_this_turn > 0) {
            int32_t prev = completed_tasks_.fetch_add(completed_this_turn, std::memory_order_relaxed);
            int32_t new_total = prev + completed_this_turn;
            last_progress_count = new_total;
            if (thread_idx == 0 && task_count > 0) {
                if (new_total <= PROGRESS_VERBOSE_THRESHOLD ||
                    new_total / PROGRESS_LOG_INTERVAL != prev / PROGRESS_LOG_INTERVAL || new_total >= task_count) {
                    LOG_INFO_V9(
                        "PTO2 progress: completed=%d total=%d (%.1f%%)", new_total, task_count,
                        100.0 * new_total / task_count
                    );
                }
            }
        }

        if (rt_ != nullptr && rt_->aicore_mailbox != nullptr &&
            (sched_->async_wait_list.count > 0 || sched_->async_wait_list.pending_completion_count > 0)) {
            AsyncPollResult poll_result = sched_->async_wait_list.poll_and_complete<false>(
                rt_->aicore_mailbox, sched_, local_bufs, deferred_release_slot_states, deferred_release_count,
                PTO2_DEFERRED_RELEASE_CAP
            );
            if (poll_result.error_code != PTO2_ERROR_NONE) {
                int32_t expected = PTO2_ERROR_NONE;
                header->sched_error_code.compare_exchange_strong(
                    expected, poll_result.error_code, std::memory_order_acq_rel, std::memory_order_acquire
                );
                completed_.store(true, std::memory_order_release);
                break;
            }
            if (poll_result.completed > 0) {
                int32_t prev = completed_tasks_.fetch_add(poll_result.completed, std::memory_order_relaxed);
                int32_t new_total = prev + poll_result.completed;
                last_progress_count = new_total;
                made_progress = true;
            }
        }

        bool try_pushed = false;

        // Phase 2 drain check
        if (drain_state_.sync_start_pending.load(std::memory_order_acquire) != 0) {
            handle_drain_mode(thread_idx);
            continue;
        }

        // Phase 3: Drain wiring queue (thread 0 only)
        if (thread_idx == 0) {
            int wired = sched_->drain_wiring_queue(orchestrator_done_);
            if (wired > 0) {
                made_progress = true;
            }
        }

        // Phase 3b: Drain dummy ready queue (thread 0 only).
        //
        // Dependency-only tasks bypass AICore dispatch: they go through the
        // scheduler so fanin/fanout edges stay consistent, but completion is
        // signalled inline here. Pinned to thread 0 to avoid cross-thread
        // races and to keep cache hot near the wiring drain above.
        if (thread_idx == 0) {
            constexpr int DUMMY_DRAIN_BATCH = 16;
            PTO2TaskSlotState *dummy_batch[DUMMY_DRAIN_BATCH];
            int dummy_got = sched_->dummy_ready_queue.pop_batch(dummy_batch, DUMMY_DRAIN_BATCH);
            for (int di = 0; di < dummy_got; di++) {
                PTO2TaskSlotState &dummy_slot = *dummy_batch[di];
                sched_->on_mixed_task_complete(dummy_slot, local_bufs);
                // Dummy tasks have no subtasks to retire and no fanout pre-conditions
                // beyond their own producers; release self-reference so the slot can
                // reach CONSUMED once all consumers drain.
                deferred_release_slot_states[deferred_release_count++] = &dummy_slot;
                if (deferred_release_count >= PTO2_DEFERRED_RELEASE_CAP) {
                    while (deferred_release_count > 0) {
                        sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count]);
                    }
                }
                int32_t prev = completed_tasks_.fetch_add(1, std::memory_order_relaxed);
                last_progress_count = prev + 1;
                cur_thread_completed++;
            }
            if (dummy_got > 0) {
                made_progress = true;
            }
        }

        // Phase 4: MIX-strict-priority dispatch with phase-split and
        // cross-thread idle gating. See dispatch_ready_tasks for the policy.
        dispatch_ready_tasks(thread_idx, tracker, local_bufs, pmu_active, made_progress, try_pushed);
        (void)try_completed;
        (void)try_pushed;

        if (made_progress) {
            idle_iterations = 0;
        } else {
            while (deferred_release_count > 0) {
                sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count]);
            }
            idle_iterations++;

            if (idle_iterations % FATAL_ERROR_CHECK_INTERVAL == 0) {
                LoopAction action = check_idle_fatal_error(thread_idx, header, runtime);
                if (action == LoopAction::BREAK_LOOP) break;
            }

            if (idle_iterations % STALL_LOG_INTERVAL == 0) {
                log_stall_diagnostics(thread_idx, total_tasks_, idle_iterations, last_progress_count);
            }
            if (idle_iterations >= MAX_IDLE_ITERATIONS) {
                return handle_timeout_exit(
                    thread_idx, header, runtime, idle_iterations
                );
            } else {
                SPIN_WAIT_HINT();
            }
        }
    }

    // Drain any entries left in the deferred-release batch. The in-loop flush
    // only fires on idle iterations and on buffer-full; a loop exit while the
    // last iteration made progress can leave entries un-released. Drop them
    // here so every consumed producer slot completes its on_task_release
    // regardless of which loop-exit path fired.
    while (deferred_release_count > 0) {
        sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count]);
    }

    return cur_thread_completed;
}
