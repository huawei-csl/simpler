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

#ifdef __SIMULATED_DEVICE__
#include "aicpu/asimgq_core.h"
#endif

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdio>

#include <tracr/tracr.hpp>
#include <tracr_simpler_markers.hpp>

#include "assert_compat.h"
#include "common/unified_log.h"
#include "aicpu/device_time.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/platform_regs.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "aicpu/args_dump_aicpu.h"
#include "common/memory_barrier.h"
#include "common/chip_swimlane_profiling.h"
#include "common/platform_config.h"
#include "host_build_graph/runtime_status.h"
#include "host_build_graph/runtime_core.h"
#include "host_build_graph/shared_memory.h"
#include "runtime.h"
#include "spin_hint.h"

// =============================================================================
// Cold-path helpers for the main dispatch loop (noinline to reduce hot-loop icache)
// =============================================================================

static void latch_scheduler_error(SharedMemoryHeader *header, int32_t thread_idx, int32_t error_code) {
    if (header == nullptr || error_code == SIMPLER_ERROR_NONE) {
        return;
    }
    // The first error code/thread pair wins; the bitmap cumulatively records all reporting threads.
    int32_t expected = SIMPLER_ERROR_NONE;
    if (header->sched_error_code.compare_exchange_strong(expected, error_code, std::memory_order_acq_rel)) {
        header->sched_error_thread.store(thread_idx, std::memory_order_release);
    }
    if (thread_idx >= 0 && thread_idx < 32) {
        header->sched_error_bitmap.fetch_or(1U << static_cast<uint32_t>(thread_idx), std::memory_order_acq_rel);
    }
}

void SchedulerContext::fail_scheduler(Runtime *runtime, int32_t thread_idx, int32_t error_code) {
    latch_scheduler_error(sched_ == nullptr ? nullptr : sched_->sm_header, thread_idx, error_code);
    if (!completed_.exchange(true, std::memory_order_acq_rel)) {
        emergency_shutdown(runtime);
    }
}

LoopAction
SchedulerContext::check_latched_sched_error(int32_t thread_idx, SharedMemoryHeader *header, Runtime *runtime) {
    if (completed_.load(std::memory_order_acquire)) {
        return LoopAction::BREAK_LOOP;
    }
    int32_t sched_err = header->sched_error_code.load(std::memory_order_acquire);
    if (sched_err != SIMPLER_ERROR_NONE) {
        LOG_ERROR(
            "Thread %d: Scheduler fatal error detected (code=%d), sending EXIT_SIGNAL to all cores. "
            "completed_tasks=%d, total_tasks=%d",
            thread_idx, sched_err, completed_tasks_.load(std::memory_order_relaxed), total_tasks_
        );
        if (!completed_.exchange(true, std::memory_order_acq_rel)) {
            emergency_shutdown(runtime);
        }
        return LoopAction::BREAK_LOOP;
    }
    return LoopAction::NONE;
}

LoopAction SchedulerContext::check_exit_conditions(
    int32_t thread_idx, SharedMemoryHeader *header, Runtime *runtime, int32_t &task_count
) {
    if (check_latched_sched_error(thread_idx, header, runtime) == LoopAction::BREAK_LOOP) {
        return LoopAction::BREAK_LOOP;
    }

    task_count = total_tasks_;
    if (task_count > 0 && completed_tasks_.load(std::memory_order_relaxed) >= task_count) {
        completed_.store(true, std::memory_order_release);
        LOG_INFO(
            "Thread %d: completed tasks %d/%d", thread_idx, completed_tasks_.load(std::memory_order_relaxed), task_count
        );
        return LoopAction::BREAK_LOOP;
    }
    return LoopAction::NONE;
}

LoopAction SchedulerContext::check_idle_fatal_error(int32_t thread_idx, SharedMemoryHeader *header, Runtime *runtime) {
    return check_latched_sched_error(thread_idx, header, runtime);
}

// =============================================================================
// Stall diagnostic log format.
//
// Every line is self-contained — when scheduler threads emit concurrently and
// device_log interleaves their output, each line still carries enough context
// to identify which thread / iteration / object it belongs to.
//
// Prefix on every line:
//   [STALL thread=N idle_iterations=K] CATEGORY ...
//
// All scheduler threads spinning at the same idle rate hit STALL_LOG_INTERVAL
// together, so lines with the same idle_iterations belong to one diagnostic
// round; grep "idle_iterations=N" groups one round's output.
//
// Categories (and which thread emits them):
//   SUMMARY  — completed / total counts and scan totals               (thread 0 only)
//   TASK     — one per non-completed task in the shared table          (thread 0 only)
//              - state=RUNNING: includes running_on=[...] cross-ref
//              - state=READY:   fanin satisfied but no idle core yet
//              - state=WAIT:    includes missing_deps=N
//   CLUSTER  — one per cluster owned by this thread                   (every thread)
//              - busy slot shows kernel + task_id + cond_reg_state;
//                ANOMALY suffix when COND register is fin while software
//                still has the slot marked busy.
//
// Reader workflow:
//   1. grep SUMMARY                          -> overall completion status
//   2. grep "idle_iterations=N TASK"         -> stuck RUNNING task and which
//                                               core/thread it is on
//   3. grep "idle_iterations=N CLUSTER.*task=<id>" -> cross-check via the
//                                                     cluster line (or just
//                                                     read running_on in step 2)
// =============================================================================

namespace {

// Format a core's idle/busy state into a fixed buffer. Used inside CLUSTER lines.
// Layout (idle):    coreN(idle)
// Layout (busy):    coreN(busy kernel=K task=T cond_reg_state=ack)
// Layout (anomaly): coreN(busy kernel=K task=T cond_reg_state=fin ANOMALY)
//
// Healthy busy: COND register reports ack (AICore still executing). fin means
// AICore wrote completion but AICPU hasn't recycled the running slot yet —
// either a completion-poll bug or the diagnostic raced the recycle.
void format_core_status(
    char *buf, size_t buf_size, int32_t core_id, bool idle, const CoreExecState *core_state, uint64_t reg_addr_for_cond
) {
    if (idle) {
        snprintf(buf, buf_size, "core%d(idle)", core_id);
        return;
    }
    int32_t kernel = -1;
    int64_t task_id_raw = -1;
    if (core_state && core_state->running_slot_state) {
        int32_t subslot = static_cast<int32_t>(core_state->running_subslot);
        kernel = core_state->running_slot_state->to_descriptor().kernel_id[subslot];
        task_id_raw = static_cast<int64_t>(core_state->running_slot_state->to_descriptor().task_id.raw);
    }
    uint64_t cond_reg = read_reg(reg_addr_for_cond, RegId::COND);
    int32_t hw_state = EXTRACT_TASK_STATE(cond_reg);
    const char *cond_reg_state_str = (hw_state == TASK_ACK_STATE) ? "ack" : "fin";
    if (hw_state == TASK_ACK_STATE) {
        snprintf(
            buf, buf_size, "core%d(busy kernel=%d task=%" PRId64 " cond_reg_state=%s)", core_id, kernel, task_id_raw,
            cond_reg_state_str
        );
    } else {
        snprintf(
            buf, buf_size,
            "core%d(busy kernel=%d task=%" PRId64
            " cond_reg_state=%s ANOMALY cond_tok=%d running_tok=%d pending_tok=%d)",
            core_id, kernel, task_id_raw, cond_reg_state_str, EXTRACT_TASK_ID(cond_reg),
            core_state->running_reg_task_id, core_state->pending_reg_task_id
        );
    }
}

}  // namespace

int32_t SchedulerContext::find_core_owner_thread(int32_t core_id) const {
    for (int32_t t = 0; t < aicpu_thread_num_; t++) {
        const int32_t *ids = core_trackers_[t].core_ids();
        int32_t n = core_trackers_[t].core_num();
        for (int32_t i = 0; i < n; i++) {
            if (ids[i] == core_id) return t;
        }
    }
    return -1;
}

bool SchedulerContext::self_owns_running_task(int32_t thread_idx) const {
    const int32_t *cores = core_trackers_[thread_idx].core_ids();
    int32_t core_num = core_trackers_[thread_idx].core_num();
    for (int32_t i = 0; i < core_num; i++) {
        if (core_exec_states_[cores[i]].running_slot_state != nullptr) {
            return true;
        }
    }
    return false;
}

bool SchedulerContext::no_thread_owns_running_task() const {
    for (int32_t t = 0; t < aicpu_thread_num_; t++) {
        if (self_owns_running_task(t)) return false;
    }
    return true;
}

void SchedulerContext::log_stall_diagnostics(
    int32_t thread_idx, int32_t task_count, int32_t idle_iterations, int32_t last_progress_count
) {
    CoreTracker &tracker = core_trackers_[thread_idx];

    if (part_active_) {
        const PartFeed &pf = part_feed_[thread_idx];
        const int32_t cur = pf.cur;
        LOG_INFO(
            "[STALL thread=%d] PART queue_part=%d installed=%d cur=%d item=%u gate=%d held=%zu filled=%d "
            "route_cursor=%" PRIu64 " store_room=%u positions_room=%u wait_room=%" PRIu64 " wait_part=%" PRIu64
            " wait_gate=%" PRIu64 " chain_wait_ext=%" PRIu64,
            thread_idx, asimgq::queue_partition(static_cast<uint32_t>(thread_idx)), pf.installed, cur, pf.cur_item,
            cur < kMaxPlanPartitions ? part_installed_[cur].load(std::memory_order_relaxed) : -1, pf.held.size(),
            pf.filled, pf.route_cursor, asimgq::part_store_room(static_cast<uint32_t>(thread_idx)),
            gq_index_[thread_idx].room(), pf.wait_room, pf.wait_part, pf.wait_gate, pf.chain_wait_ext
        );
        const int32_t ck = part_cohort_cur_.load(std::memory_order_relaxed);
        if (ck < static_cast<int32_t>(part_cohorts_.size())) {
            const PartCohort &c = part_cohorts_[static_cast<size_t>(ck)];
            LOG_INFO(
                "[STALL thread=%d] PART cohort=%d/%zu task=%d entries=%d share=%d seen=%d done=%d staged=[%d %d %d %d] "
                "stamped=%d released=%d",
                thread_idx, ck, part_cohorts_.size(), c.task, c.entries, c.share[thread_idx], pf.cohort_seen,
                pf.cohort_done, part_cohort_staged_[0].load(std::memory_order_relaxed),
                part_cohort_staged_[1].load(std::memory_order_relaxed),
                part_cohort_staged_[2].load(std::memory_order_relaxed),
                part_cohort_staged_[3].load(std::memory_order_relaxed),
                part_cohort_at_.load(std::memory_order_relaxed) != 0 ? 1 : 0,
                part_cohort_released_.load(std::memory_order_relaxed)
            );
        }
        for (size_t i = 0; i < pf.held.size() && i < 8; ++i) {
            const part_plan::Item &item = *part_plan::at<part_plan::Item>(plan_, pf.held[i].item);
            LOG_INFO(
                "[STALL thread=%d] PART held part=%d task=%d kind=%u flags=%#x entries=%u", thread_idx, pf.held[i].part,
                item.task, item.kind, item.flags, pf.held[i].entries
            );
        }
    }
    if (silo_active_ && !part_active_ && sched_->task_view.tasks != nullptr) {
        const SiloFeed &sf = silo_feed_[thread_idx];
        SharedMemoryTaskHeader &stasks = *sched_->task_view.tasks;
        const int32_t head = sf.cursor;
        int32_t lo = 0, hi = 0, nb = 0, kind = -1;
        uint32_t stamp = 0, fanin = 0, feeds = 0, done = 0;
        if (head < task_count) {
            const ChipTaskSlotState &hs = stasks.get_slot_state_by_task_id(head);
            stamp = hs.to_descriptor().silo;
            nb = hs.logical_block_num;
            kind = static_cast<int32_t>(hs.task_kind);
            fanin = static_cast<uint32_t>(hs.to_payload().fanin_count);
            feeds = silo_feeds(hs) ? 1 : 0;
            done = stasks.is_completed(head) ? 1 : 0;
            silo_piece(hs.to_descriptor().silo, nb, thread_idx, active_sched_threads_, lo, hi);
        }
        uint64_t admitted = 0, promoted = 0, refused = 0, published = 0, resolved = 0, timed = 0, late = 0, late_ns = 0;
        uint32_t held_high = 0;
        asimgq::queue_group_stats(static_cast<uint32_t>(thread_idx), &admitted, &promoted, &refused, &held_high);
        asimgq::queue_peer_stats(static_cast<uint32_t>(thread_idx), &published, &resolved, &timed, &late, &late_ns);
        LOG_INFO(
            "[STALL thread=%d] SILO cursor=%d stalled_at=%d pieces=%" PRIu64 " wait_unfed=%" PRIu64
            " wait_ordinary=%" PRIu64 " wait_comparators=%" PRIu64 " wait_room=%" PRIu64 " room=%u push=%" PRIu64
            " wm=%" PRIu64 " head: stamp=%u blocks=%d kind=%d fanin=%u feeds=%u done=%u"
            " piece=[%d,%d) admitted=%" PRIu64 " promoted=%" PRIu64 " refused=%" PRIu64 " published=%" PRIu64
            " resolved=%" PRIu64 " timed=%" PRIu64,
            thread_idx, sf.cursor, sf.stalled_at, sf.pieces, sf.wait_unfed, sf.wait_ordinary, sf.wait_comparators,
            sf.wait_room, gq_index_[thread_idx].room(), gq_index_[thread_idx].push_index(),
            gq_index_[thread_idx].watermark(), stamp, nb, kind, fanin, feeds, done, lo, hi, admitted, promoted, refused,
            published, resolved, timed
        );
        char held_buf[1536];
        asimgq::describe_held(static_cast<uint32_t>(thread_idx), held_buf, sizeof(held_buf));
        LOG_INFO("[STALL thread=%d] HELD %s", thread_idx, held_buf);
    }

    // T0 owns the task-table scan; printing it from other threads would
    // produce identical TASK lines once per scheduler thread.
    if (thread_idx == 0) {
        int32_t cnt_ready = 0, cnt_waiting = 0, cnt_running = 0;
        SharedMemoryTaskHeader &tasks = *sched_->task_view.tasks;
        // `task_count` is the run's task total, which both callers pass from
        // total_tasks_. It bounds the scan as well as the SUMMARY line: every slot
        // below it was claimed by the host orchestrator, and no slot above it was.
        for (int32_t si = 0; si < task_count; si++) {
            ChipTaskSlotState &slot_state = tasks.get_slot_state_by_task_id(si);
            const bool completed = tasks.is_completed(si, std::memory_order_relaxed);
            // Polling: no fanin_refcount. Recompute met/total from the inline
            // fanin ids vs the task_states array (rc = satisfied producers,
            // fi = raw producer count) so the stall dump still shows readiness.
            int32_t fi = slot_state.to_payload().fanin_count;
            int32_t rc = 0;
            {
                const int32_t *fanin = slot_state.to_payload().fanin_data();
                for (int32_t k = 0; k < fi; k++) {
                    if (tasks.is_completed(fanin[k], std::memory_order_relaxed)) rc++;
                }
            }
            int32_t kid_aic = slot_state.to_descriptor().kernel_id[0];
            int32_t kid_aiv0 = slot_state.to_descriptor().kernel_id[1];
            int32_t kid_aiv1 = slot_state.to_descriptor().kernel_id[2];
            int64_t task_id = static_cast<int64_t>(slot_state.to_descriptor().task_id.raw);
            if (completed) continue;
            // The state byte has no intermediate ready/running value — a task
            // stays PENDING until it publishes PUBLISHED or COMPLETED, neither
            // of which distinguishes queued from running. Classify
            // by the ground truth instead: a slot is RUNNING iff some
            // core has it as running_slot_state. A task occupies at most
            // 3 cores (one cluster), all under the same owner thread by
            // construction of assign_cores_to_threads.
            char running_on[192] = {0};
            int32_t owner = -1;
            int32_t pos = 0;
            bool is_running = false;
            for (int32_t cid = 0; cid < cores_total_num_ && pos + 32 < (int32_t)sizeof(running_on); cid++) {
                if (core_exec_states_[cid].running_slot_state != &slot_state) continue;
                is_running = true;
                if (owner < 0) owner = find_core_owner_thread(cid);
                const char *sname = subslot_name(core_exec_states_[cid].running_subslot);
                int32_t written = snprintf(
                    running_on + pos, sizeof(running_on) - pos, "%score=%d(%s)", pos == 0 ? "" : " ", cid, sname
                );
                if (written > 0) pos += written;
            }

            if (is_running) {
                cnt_running++;
                if (cnt_running > STALL_DUMP_READY_MAX) continue;
                LOG_INFO(
                    "[STALL thread=%d idle_iterations=%d] TASK ring=%d task_id=%" PRId64
                    " state=RUNNING fanin_met=%d/%d kernels=[aic:%d aiv0:%d aiv1:%d] "
                    "running_on=[owner_thread=%d cores=[%s]]",
                    thread_idx, idle_iterations, 0, task_id, rc, fi, kid_aic, kid_aiv0, kid_aiv1, owner, running_on
                );
                continue;
            }
            if (rc >= fi) {
                cnt_ready++;
                if (cnt_ready > STALL_DUMP_READY_MAX) continue;
                LOG_INFO(
                    "[STALL thread=%d idle_iterations=%d] TASK ring=%d task_id=%" PRId64
                    " state=READY   fanin_met=%d/%d kernels=[aic:%d aiv0:%d aiv1:%d]",
                    thread_idx, idle_iterations, 0, task_id, rc, fi, kid_aic, kid_aiv0, kid_aiv1
                );
                if (slot_state.task_attrs.requires_sync_start()) {
                    // A cohort staged only into pending slots is on no core's running slot,
                    // so it reads as READY above; say where its blocks actually are.
                    int32_t pending_on = 0;
                    for (int32_t cid = 0; cid < cores_total_num_; cid++) {
                        if (core_exec_states_[cid].pending_slot_state == &slot_state) pending_on++;
                    }
                    const TaskPayload &pl = slot_state.to_payload();
                    int32_t staged_cores = 0;
                    for (int w = 0; w < EARLY_DISPATCH_CORE_MASK_WORDS; w++) {
                        staged_cores += __builtin_popcountll(pl.staged_core_mask[w].load(std::memory_order_relaxed));
                    }
                    LOG_INFO(
                        "[STALL thread=%d idle_iterations=%d] SYNC task_id=%" PRId64
                        " ed_flags=0x%x ed_state=%u esd=0x%x launch=%u next_blk=%d published=%d"
                        " running_slot_count=%d staged_cores=%d pending_on=%d",
                        thread_idx, idle_iterations, task_id, static_cast<unsigned>(slot_state.ed_flags),
                        static_cast<unsigned>(pl.early_dispatch_state.load(std::memory_order_relaxed)),
                        static_cast<unsigned>(pl.early_sync_drain_state.load(std::memory_order_relaxed)),
                        static_cast<unsigned>(pl.early_dispatch_launch_state.load(std::memory_order_relaxed)),
                        static_cast<int32_t>(slot_state.next_block_idx.load(std::memory_order_relaxed)),
                        static_cast<int32_t>(pl.published_block_count.load(std::memory_order_relaxed)),
                        static_cast<int32_t>(pl.running_slot_count.load(std::memory_order_relaxed)), staged_cores,
                        pending_on
                    );
                }
                continue;
            }
            cnt_waiting++;
            if (cnt_waiting > STALL_DUMP_WAIT_MAX) continue;
            LOG_INFO(
                "[STALL thread=%d idle_iterations=%d] TASK ring=%d task_id=%" PRId64
                " state=WAIT    fanin_met=%d/%d kernels=[aic:%d aiv0:%d aiv1:%d] missing_deps=%d",
                thread_idx, idle_iterations, 0, task_id, rc, fi, kid_aic, kid_aiv0, kid_aiv1, fi - rc
            );
        }
        int32_t c = completed_tasks_.load(std::memory_order_relaxed);
        LOG_INFO(
            "[STALL thread=%d idle_iterations=%d] SUMMARY completed=%d/%d last_progress_iteration=%d "
            "scan_ready=%d scan_waiting=%d scan_running=%d",
            thread_idx, idle_iterations, c, task_count, last_progress_count, cnt_ready, cnt_waiting, cnt_running
        );
    }

    // CLUSTER lines: one per cluster this thread owns.
    // cluster_id = local_cluster_idx * active_sched_threads_ + thread_idx, matching the
    // round-robin assignment in assign_cores_to_threads.
    int32_t ast = active_sched_threads_ > 0 ? active_sched_threads_ : aicpu_thread_num_;
    for (int32_t cli = 0; cli < tracker.get_cluster_count() && cli < STALL_DUMP_CORE_MAX; cli++) {
        int32_t offset = cli * 3;
        int32_t aic_id = tracker.get_aic_core_id(offset);
        int32_t aiv0_id = tracker.get_aiv0_core_id(offset);
        int32_t aiv1_id = tracker.get_aiv1_core_id(offset);
        bool aic_idle = tracker.is_aic_core_idle(offset);
        bool aiv0_idle = tracker.is_aiv0_core_idle(offset);
        bool aiv1_idle = tracker.is_aiv1_core_idle(offset);
        int32_t cluster_id = cli * ast + thread_idx;
        char aic_buf[192], aiv0_buf[192], aiv1_buf[192];
        format_core_status(
            aic_buf, sizeof(aic_buf), aic_id, aic_idle, &core_exec_states_[aic_id], core_exec_states_[aic_id].reg_addr
        );
        format_core_status(
            aiv0_buf, sizeof(aiv0_buf), aiv0_id, aiv0_idle, &core_exec_states_[aiv0_id],
            core_exec_states_[aiv0_id].reg_addr
        );
        format_core_status(
            aiv1_buf, sizeof(aiv1_buf), aiv1_id, aiv1_idle, &core_exec_states_[aiv1_id],
            core_exec_states_[aiv1_id].reg_addr
        );
        LOG_INFO(
            "[STALL thread=%d idle_iterations=%d] CLUSTER cluster_id=%d aic=%s aiv0=%s aiv1=%s", thread_idx,
            idle_iterations, cluster_id, aic_buf, aiv0_buf, aiv1_buf
        );
    }

    // Every offset holding a pending-occupied mark. An idle one ("!IDLE") breaks the
    // tracker's invariant and makes its cluster unplaceable for MIX work.
    {
        char occ[512] = {0};
        int32_t pos = 0;
        int32_t n_occ = 0;
        int32_t n_idle_occ = 0;
        CoreTracker::BitStates p = tracker.debug_pending_occupied();
        while (p.has_value()) {
            int32_t off = p.pop_first();
            int32_t cid = tracker.get_core_id_by_offset(off);
            bool idle = tracker.debug_is_idle(off);
            const CoreExecState &ce = core_exec_states_[cid];
            long long run = ce.running_slot_state ? (long long)ce.running_slot_state->to_descriptor().task_id.raw : -1;
            long long pend = ce.pending_slot_state ? (long long)ce.pending_slot_state->to_descriptor().task_id.raw : -1;
            n_occ++;
            if (idle) n_idle_occ++;
            if (pos + 64 < static_cast<int32_t>(sizeof(occ))) {
                int w = snprintf(
                    occ + pos, sizeof(occ) - pos, " c%d%s(r=%lld p=%lld preg=%d)", cid, idle ? "!IDLE" : "", run, pend,
                    ce.pending_reg_task_id
                );
                if (w > 0) pos += w;
            }
        }
        LOG_INFO(
            "[STALL thread=%d idle_iterations=%d] PENDING_OCC count=%d idle_with_mark=%d:%s", thread_idx,
            idle_iterations, n_occ, n_idle_occ, occ
        );
    }
    if (thread_idx == 0) {
        ChipTaskSlotState *pt = drain_state_.pending_task.load(std::memory_order_relaxed);
        LOG_INFO(
            "[STALL thread=%d idle_iterations=%d] DRAIN pending=%d attempt=%llu task=%lld stage_go=%d done_mask=0x%x"
            " acks=[%llx %llx %llx %llx] rsq_mix=%llu early_sync_q=%llu",
            thread_idx, idle_iterations, drain_state_.sync_start_pending.load(std::memory_order_relaxed),
            (unsigned long long)drain_state_.drain_attempt.load(std::memory_order_relaxed),
            pt ? (long long)pt->to_descriptor().task_id.raw : -1LL,
            drain_state_.drain_stage_go.load(std::memory_order_relaxed),
            drain_state_.drain_stage_done_mask.load(std::memory_order_relaxed),
            (unsigned long long)drain_ack_tokens_[0].load(std::memory_order_relaxed),
            (unsigned long long)drain_ack_tokens_[1].load(std::memory_order_relaxed),
            (unsigned long long)drain_ack_tokens_[2].load(std::memory_order_relaxed),
            (unsigned long long)drain_ack_tokens_[3].load(std::memory_order_relaxed),
            (unsigned long long)sched_->ready_sync_queues[static_cast<int32_t>(ResourceShape::MIX)].size(),
            (unsigned long long)sched_->early_sync_start_queue.size()
        );
    }
}

void SchedulerContext::log_shutdown_stall_snapshot(
    int32_t trigger_thread_idx, int32_t trigger_idle_iterations, int32_t trigger_last_progress_count
) {
    LOG_WARN(
        "[SHUTDOWN_SNAPSHOT trigger_thread=%d reason=scheduler_timeout idle_iterations=%d] "
        "dumping all scheduler threads before emergency shutdown",
        trigger_thread_idx, trigger_idle_iterations
    );
    int32_t thread_count = active_sched_threads_ > 0 ? active_sched_threads_ : aicpu_thread_num_;
    if (thread_count < 0 || thread_count > MAX_AICPU_THREADS) {
        LOG_ERROR(
            "[SHUTDOWN_SNAPSHOT trigger_thread=%d] invalid thread_count=%d, clamping to [0,%d]", trigger_thread_idx,
            thread_count, MAX_AICPU_THREADS
        );
        thread_count = thread_count < 0 ? 0 : MAX_AICPU_THREADS;
    }
    for (int32_t t = 0; t < thread_count; t++) {
        log_stall_diagnostics(t, total_tasks_, trigger_idle_iterations, trigger_last_progress_count);
    }
}

// Whether every producer that reaches `group` from outside it has retired.
//
// Edges run forward and a group is a contiguous id range, so a group's external
// producers all sit in lower-numbered groups: this can only turn true, and the
// groups turn true in order. That is what lets the ready group queue be a single
// position rather than a set, and what replaces a per-group counter the
// completion path would otherwise have to maintain -- under polling a producer
// does not enumerate its consumers, so there is no edge list to decrement along.
bool SchedulerContext::group_externals_met(int32_t group) const {
    if (!gq_group::active() || sched_->task_view.tasks == nullptr) return false;
    SharedMemoryTaskHeader &tasks = *sched_->task_view.tasks;
    const int32_t submitted = total_tasks_;
    const int32_t members = gq_group::group_member_count(group);
    const gq_group::GroupRun run{group, gq_group::group_member(group, 0), gq_group::group_member(group, 0) + members};
    for (int32_t mi = 0; mi < members; ++mi) {
        const int32_t id = run.first + mi;
        if (id < 0 || id >= submitted) continue;
        const TaskPayload &p = tasks.get_slot_state_by_task_id(id).to_payload();
        const int32_t *fanin = p.fanin_data();
        for (int32_t k = 0; k < p.fanin_count; ++k) {
            const int32_t prod = fanin[k];
            if (run.holds(prod)) continue;
            if (!tasks.is_completed(prod)) return false;
        }
    }
    return true;
}

// Hand the controller more of the group this thread owns, in ascending task
// order, taking a position and filling it in the same step.
//
// Ascending order is what lets the controller hold an internal edge at all: a
// consumer can only name producers that already hold positions. Taking and
// filling a position together is what keeps the manager retiring -- it harvests
// completions as a contiguous prefix, so a position reserved now and submitted
// later stops that prefix at itself and nothing behind it is ever retired.
bool SchedulerContext::feed_open_groups(int32_t thread_idx) {
    if (!gq_group::active() || sched_->task_view.tasks == nullptr) return false;
    SharedMemoryTaskHeader &tasks = *sched_->task_view.tasks;
    GqIndexSpace &space = gq_index_[thread_idx];
    GroupFeed &f = group_feed_[thread_idx];
    const int32_t submitted = total_tasks_;
    // The controller reports how many entries it can still hold for tasks whose
    // producers have not retired; offering past that is what a refusal would be.
    const asimgq::SimQueueStatus st = asimgq::read_queue_status(static_cast<uint32_t>(thread_idx));
    uint32_t hold_budget = st.held_room;
    bool progress = false;

    while (space.room() > 0 && hold_budget > 0) {
        if (f.group < 0) {
            int32_t from = gq_group::g_scan_from.load(std::memory_order_acquire);
            const uint64_t epoch = gq_group::ENABLED ? g_completion_epoch.load(std::memory_order_acquire) : 0;
            // Judging a group ready walks its fanin, so re-ask only when the scan
            // point moved or something retired -- otherwise an idle thread spends
            // the window it is meant to be measuring.
            if (from == f.probed_group && epoch == f.probed_epoch && ++f.probe_skips < GroupFeed::kProbeStaleLimit) {
                return progress;
            }
            f.probed_group = from;
            f.probed_epoch = epoch;
            f.probe_skips = 0;
            int32_t taken = -1;
            const int32_t last = from + gq_group::GROUP_SCAN_WINDOW;
            for (int32_t g = from; g < last && g < gq_group::g_group_count; ++g) {
                if (gq_group::group_owner(g) != gq_group::NO_OWNER) {
                    // Carry the mark past groups already taken, as they are passed.
                    // Folding it into this walk costs nothing; a separate pass over
                    // them runs on every probe, and with many short-lived groups that
                    // is more than the anchor it exists to prevent.
                    if (g == from) from = gq_group::bump_scan_mark(g);
                    continue;
                }
                if (!group_externals_met(g)) continue;
                if (gq_group::try_claim_group(g, thread_idx)) {
                    taken = g;
                    break;
                }
            }
            if (taken < 0) return progress;
            f.group = taken;
            gq_group::mark_group_opened(f.group);
            f.mi = 0;
            f.members = gq_group::group_member_count(taken);
            f.retry_from = -1;
            f.swept_clean = true;
            f.probe_skips = 0;
            f.block = 0;
            f.sub = 0;
            f.accounted = false;
        }
        if (f.mi >= f.members) {
            if (!f.swept_clean && f.retry_from >= 0) {
                // Something was passed over; go round again from the lowest of them.
                // Returning here rather than looping keeps one pass bounded.
                f.mi = f.retry_from;
                f.block = 0;
                f.sub = 0;
                f.accounted = false;
                f.retry_from = -1;
                f.swept_clean = true;
                return progress;
            }
            f.group = -1;
            continue;
        }
        f.id = gq_group::group_member(f.group, f.mi);
        if (f.id < 0 || f.id >= submitted) {
            ++f.mi;
            f.block = 0;
            f.sub = 0;
            f.accounted = false;
            continue;
        }
        ChipTaskSlotState &slot = tasks.get_slot_state_by_task_id(f.id);
        // A sweep that had to pass something over restarts at the lowest of them, so
        // it revisits tasks it already emitted. Emitting one twice would count its
        // blocks twice and the publication seal would never match, leaving a task
        // that nothing is waiting for permanently unpublished. A position is the
        // record that it went out.
        if (f.block == 0 && f.sub == 0 && !f.accounted && task_position(f.id) != kNoPosition) {
            ++f.mi;
            continue;
        }
        if (tasks.is_completed(f.id) || !group_queue_delivers(slot)) {
            // Not this queue's to deliver; classify left it on the ordinary path.
            ++f.mi;
            f.block = 0;
            f.sub = 0;
            f.accounted = false;
            continue;
        }
        if (!f.accounted) {
            // Asked once per task, before any of it is accounted or emitted: the
            // answer is a property of the task, and a task must be emitted whole.
            uint64_t probe[asimgq::SIM_HELD_MAX_DEPS];
            if (sched_->group_deps(&slot, f.id, probe, asimgq::SIM_HELD_MAX_DEPS) < 0) {
                if (f.retry_from < 0) f.retry_from = f.mi;
                f.swept_clean = false;
                ++f.mi;
                f.block = 0;
                f.sub = 0;
                continue;
            }
            sched_->account_published_blocks(slot, slot.logical_block_num);
            f.accounted = true;
        }
        // Walk to this task's next active subtask slot; one entry per (block,
        // slot) is exactly the total_required_subtasks the completion side counts.
        while (f.sub < SUBTASK_SLOT_COUNT && !slot.active_mask.subtask_active(static_cast<SubtaskSlot>(f.sub))) {
            ++f.sub;
        }
        if (f.sub >= SUBTASK_SLOT_COUNT) {
            f.sub = 0;
            ++f.block;
            if (f.block >= slot.logical_block_num) {
                ++f.mi;
                f.block = 0;
                f.accounted = false;
            }
            continue;
        }

        const SubtaskSlot subslot = static_cast<SubtaskSlot>(f.sub);
        const int32_t fid = slot.to_descriptor().kernel_id[f.sub];
        const uint64_t idx = space.reserve();
        if (idx == UINT64_MAX) return progress;
        const int32_t book_core = core_trackers_[thread_idx].get_core_id_by_offset(0);
        // capacity_token is -1: this entry took nothing out of the thread's core
        // budget. The controller holds it until its producers retire and places it
        // itself, so there is no token for completion to give back.
        space.set_owner(idx, GqIndexSpace::Owner{&slot, feed_seq_[thread_idx]++, subslot, book_core, -1, false, false});
        note_task_position(f.id, idx);

        uint64_t deps[asimgq::SIM_HELD_MAX_DEPS];
        const int d = sched_->group_deps(&slot, f.id, deps, asimgq::SIM_HELD_MAX_DEPS);
        if (d < 0) {
            // Cleared at the task boundary above, so this can only mean a producer
            // retired mid-task and the answer improved; nothing to express is safe.
            space.unreserve(idx);
            return progress;
        }
        const int dep_n = d;
        const asimgq::SimTaskType ring =
            (subslot == SubtaskSlot::AIC) ? asimgq::SimTaskType::Cube : asimgq::SimTaskType::Vector;
        if (!asimgq::submit_grouped(
                static_cast<uint32_t>(thread_idx), &idx, &fid, 1,
                static_cast<int32_t>(slot.to_descriptor().task_id.raw), ring, deps, static_cast<uint32_t>(dep_n)
            )) {
            // Ordinary back-pressure: the controller's ring is full, or it has no
            // hold entry left. Give the position back rather than leaving it
            // reserved and empty, and re-offer this same entry on a later pass.
            space.unreserve(idx);
            note_task_position(f.id, kNoPosition);
            return progress;
        }
        if (dep_n > 0 && hold_budget > 0) --hold_budget;
        progress = true;
        ++f.sub;
    }
    return progress;
}

// Decide which groups the ready group queue can hold. A group qualifies when no
// edge enters it from outside: everything its members wait on is then a member
// too, so a thread that takes the group can submit all of it and let the
// controller resolve what is internal. A group any external edge reaches is left
// out and arrives task by task, because its entry tasks wait on producers no
// controller here holds.
void SchedulerContext::seed_ready_groups() {
    // The compile-time test, not active(): whether the run is grouped is the
    // question this function answers, so it cannot be its own precondition.
    if (!gq_group::ENABLED || sched_->task_view.tasks == nullptr) return;
    // The grouping is already in the task records the orchestrator wrote; all that
    // is left is to point at them and index the declaration's runs. Whether a group
    // has edges entering it, and whether a task is one the queue delivers, are both
    // asked where the task slot is already in hand -- a pre-scan of the whole graph
    // would cost a pass over every task and every edge, once per run, inside the
    // window being measured.
    gq_group::bind_declaration(sched_->task_view.tasks);
    const int32_t groups = gq_group::index_declaration();
    LOG_INFO(
        "[GQ_GROUP] groups=%d declared (%d tasks, first group holds %d)", groups, total_tasks_,
        gq_group::group_member_count(0)
    );
}

// Hand this thread's silo its next pieces, in task order: for each stamped task,
// the block range this silo runs, named against the pieces of every producer that
// has not finished. The controller holds the piece until those positions finish,
// in its own queue or -- by what the other queues publish -- in theirs, so no
// AICPU stands between a producer finishing and its consumer starting.
//
// The feed stops at the first task it cannot hand over yet: a producer piece no
// silo has fed, a producer left on the ordinary path that has not finished, more
// producer pieces running than the comparators hold, or no room for the piece.
// Every producer precedes its consumer in task order, so the lowest task any silo
// is stopped at has every producer piece fed already -- the feeds cannot wait on
// each other in a cycle.
bool SchedulerContext::feed_silos(int32_t thread_idx) {
    if (!silo_active_ || sched_->task_view.tasks == nullptr) return false;
    SharedMemoryTaskHeader &tasks = *sched_->task_view.tasks;
    SiloFeed &f = silo_feed_[thread_idx];
    GqIndexSpace &space = gq_index_[thread_idx];
    const int32_t submitted = total_tasks_;
    const int32_t silos = active_sched_threads_;
    bool progress = false;
    while (f.cursor < submitted) {
        const int32_t id = f.cursor;
        ChipTaskSlotState &slot = tasks.get_slot_state_by_task_id(id);
        if (tasks.is_completed(id) || !silo_feeds(slot)) {
            ++f.cursor;
            continue;
        }
        const TaskDescriptor &desc = slot.to_descriptor();
        int32_t lo = 0, hi = 0;
        silo_piece(desc.silo, slot.logical_block_num, thread_idx, silos, lo, hi);
        if (lo >= hi) {
            ++f.cursor;
            continue;
        }
        // A stop is re-examined only once something it waits on can have moved.
        const uint64_t completions = g_completion_epoch.load(std::memory_order_acquire);
        const uint64_t pieces = g_piece_epoch.load(std::memory_order_acquire);
        const bool fresh_stop = f.stalled_at != id;
        // The bound on skips is a backstop against a completion that moves no epoch.
        if (!fresh_stop && f.stalled_completions == completions && f.stalled_pieces == pieces &&
            f.stalled_room >= space.room() && ++f.stalled_skips < SiloFeed::kStaleLimit) {
            return progress;
        }
        auto stop = [&](uint64_t &counter) {
            if (fresh_stop) {
                ++counter;
                if (f.stalled_reason >= 0) f.stall_ticks[f.stalled_reason] += asimgq::now_ticks() - f.stalled_since;
                f.stalled_since = asimgq::now_ticks();
                f.stalled_reason = &counter == &f.wait_unfed       ? 0 :
                                   &counter == &f.wait_ordinary    ? 1 :
                                   &counter == &f.wait_comparators ? 2 :
                                                                     3;
            }
            f.stalled_at = id;
            f.stalled_completions = completions;
            f.stalled_pieces = pieces;
            f.stalled_room = space.room();
            f.stalled_skips = 0;
            return progress;
        };
        const int32_t subs = active_subtask_count(slot);
        const uint32_t n = static_cast<uint32_t>((hi - lo) * subs);
        if (n > space.room()) return stop(f.wait_room);
        asimgq::SimDep deps[asimgq::SIM_HELD_MAX_DEPS];
        uint32_t dep_n = 0;
        uint32_t retired = 0;
        const TaskPayload &payload = slot.to_payload();
        const int32_t *fanin = payload.fanin_data();
        for (int32_t k = 0; k < payload.fanin_count; ++k) {
            const int32_t prod = fanin[k];
            if (tasks.is_completed(prod)) {
                ++retired;
                continue;
            }
            const ChipTaskSlotState &ps = tasks.get_slot_state_by_task_id(prod);
            if (!silo_feeds(ps)) return stop(f.wait_ordinary);
            const int32_t psubs = active_subtask_count(ps);
            for (int32_t s = 0; s < silos; ++s) {
                int32_t plo = 0, phi = 0;
                silo_piece(ps.to_descriptor().silo, ps.logical_block_num, s, silos, plo, phi);
                if (plo >= phi) continue;
                const uint64_t first = piece_position(s, prod);
                if (first == kNoPosition) return stop(f.wait_unfed);
                if (dep_n == asimgq::SIM_HELD_MAX_DEPS) return stop(f.wait_comparators);
                deps[dep_n++] =
                    asimgq::SimDep{first, static_cast<uint32_t>((phi - plo) * psubs), static_cast<uint32_t>(s)};
            }
        }
        // One entry per (block, subtask), on consecutive positions, so a consumer
        // names the whole piece with one range.
        uint64_t idx[asimgq::SIM_HELD_CAP];
        int32_t fid[asimgq::SIM_HELD_CAP];
        asimgq::SimTaskType type[asimgq::SIM_HELD_CAP];
        uint32_t m = 0;
        const int32_t book_core = core_trackers_[thread_idx].get_core_id_by_offset(0);
        for (int32_t b = lo; b < hi; ++b) {
            for (int32_t sub = 0; sub < SUBTASK_SLOT_COUNT; ++sub) {
                const SubtaskSlot subslot = static_cast<SubtaskSlot>(sub);
                if (!slot.active_mask.subtask_active(subslot)) continue;
                // capacity_token is -1: the controller places this entry itself, so
                // it takes nothing from the thread's core budget.
                idx[m] = space.assign(
                    GqIndexSpace::Owner{&slot, feed_seq_[thread_idx]++, subslot, book_core, -1, false, false}
                );
                fid[m] = desc.kernel_id[sub];
                type[m] = (subslot == SubtaskSlot::AIC) ? asimgq::SimTaskType::Cube : asimgq::SimTaskType::Vector;
                ++m;
            }
        }
        // Account before the entries reach the queue (see account_published_blocks).
        sched_->account_published_blocks(slot, hi - lo);
        if (!asimgq::submit_held(
                static_cast<uint32_t>(thread_idx), idx, fid, type, m, static_cast<int32_t>(desc.task_id.raw), deps,
                dep_n
            )) {
            // The piece was sized against the position window, which bounds the hold
            // slots too, so a refusal means that accounting is wrong.
            for (uint32_t j = m; j-- > 0;) {
                space.unreserve(idx[j]);
            }
            LOG_ERROR(
                "Thread %d: silo refused a piece of task %d (%u entries, %u producers)", thread_idx, id, m, dep_n
            );
            return stop(f.wait_room);
        }
        note_piece_position(thread_idx, id, idx[0]);
        f.producers_retired += retired;
        if (payload.fanin_count > 0 && retired == static_cast<uint32_t>(payload.fanin_count)) ++f.pieces_behind;
        f.named += dep_n;
        f.entries += m;
        ++f.pieces;
        if (f.stalled_reason >= 0 && f.stalled_at == id) {
            f.stall_ticks[f.stalled_reason] += asimgq::now_ticks() - f.stalled_since;
            f.stalled_reason = -1;
        }
        f.stalled_at = -1;
        ++f.cursor;
        progress = true;
    }
    return progress;
}

// An unfinished producer of `task`, a dependency-only producer seen through to its
// own producers -- the latest submitted, as the likeliest to finish last -- or -1
// once every one has finished. `walk` is scratch.
static int32_t unfinished_producer(SharedMemoryTaskHeader &tasks, int32_t task, std::vector<int32_t> &walk) {
    const TaskPayload &payload = tasks.get_slot_state_by_task_id(task).to_payload();
    walk.assign(payload.fanin_data(), payload.fanin_data() + payload.fanin_count);
    int32_t latest = -1;
    while (!walk.empty()) {
        const int32_t prod = walk.back();
        walk.pop_back();
        if (tasks.is_completed(prod)) continue;
        const ChipTaskSlotState &ps = tasks.get_slot_state_by_task_id(prod);
        if (ps.task_kind == TaskKind::DUMMY || ps.active_mask.to_shape() == ResourceShape::DUMMY) {
            const TaskPayload &pp = ps.to_payload();
            walk.insert(walk.end(), pp.fanin_data(), pp.fanin_data() + pp.fanin_count);
        } else {
            latest = std::max(latest, prod);
        }
    }
    return latest;
}

// Whether every producer of a task that the controllers do not run has finished,
// seen through dependency-only tasks.
static bool outside_producers_done(SharedMemoryTaskHeader &tasks, int32_t task, std::vector<int32_t> &walk) {
    const TaskPayload &payload = tasks.get_slot_state_by_task_id(task).to_payload();
    walk.assign(payload.fanin_data(), payload.fanin_data() + payload.fanin_count);
    while (!walk.empty()) {
        const int32_t prod = walk.back();
        walk.pop_back();
        if (tasks.is_completed(prod)) continue;
        const ChipTaskSlotState &ps = tasks.get_slot_state_by_task_id(prod);
        if (part_feeds(ps)) continue;
        if (ps.task_kind != TaskKind::DUMMY && ps.active_mask.to_shape() != ResourceShape::DUMMY) return false;
        const TaskPayload &pp = ps.to_payload();
        walk.insert(walk.end(), pp.fanin_data(), pp.fanin_data() + pp.fanin_count);
    }
    return true;
}

// Whether a task's dispatch predicate has its final value -- once the task writing
// it has finished, or with none named every producer -- and, if so, whether it runs.
static bool predicate_decided(SharedMemoryTaskHeader &tasks, int32_t task, std::vector<int32_t> &walk, bool &runs) {
    const ChipTaskSlotState &slot = tasks.get_slot_state_by_task_id(task);
    const int32_t writer = slot.to_descriptor().predicate_writer;
    const int32_t wait_on =
        writer >= 0 ? (tasks.is_completed(writer) ? -1 : writer) : unfinished_producer(tasks, task, walk);
    if (wait_on >= 0) return false;
    runs = slot.to_payload().predicate.pass();
    return true;
}

static_assert(sizeof(asimgq::SimMapEntry) == sizeof(part_plan::MapEntry), "a plan map entry is what the queue reads");
static_assert(offsetof(asimgq::SimMapEntry, local) == offsetof(part_plan::MapEntry, local));
static_assert(offsetof(asimgq::SimMapEntry, expected) == offsetof(part_plan::MapEntry, expected));

const part_plan::SubPartition &SchedulerContext::plan_sub(int32_t part, int32_t g) const {
    const part_plan::Header &h = *part_plan::at<part_plan::Header>(plan_, 0);
    return *part_plan::at<part_plan::SubPartition>(
        plan_, h.sub_offset + (static_cast<uint32_t>(part) * part_plan::kGroups + static_cast<uint32_t>(g)) *
                                  part_plan::words_of<part_plan::SubPartition>()
    );
}

uint32_t SchedulerContext::part_room(int32_t thread_idx, int32_t part) const {
    const PartFeed &pf = part_feed_[thread_idx];
    const uint32_t q = static_cast<uint32_t>(thread_idx);
    // Room an earlier partition's held items need is not this one's to take; only a
    // partition the controller has not finished can still be waiting.
    uint32_t reserved = 0;
    for (int32_t k = std::max(0, asimgq::queue_partition(q));
         k < part && static_cast<size_t>(k) < pf.held_entries.size(); ++k) {
        reserved += pf.held_entries[static_cast<size_t>(k)];
    }
    const uint32_t free = std::min(asimgq::part_store_room(q), gq_index_[thread_idx].room());
    return free > reserved ? free - reserved : 0;
}

static uint32_t item_entries(const part_plan::Item &item, const ChipTaskSlotState &slot) {
    if (item.kind == part_plan::kSlot) return 1;
    return (item.block_hi - item.block_lo) * static_cast<uint32_t>(active_subtask_count(slot));
}

void SchedulerContext::stage_item(PartFeed &pf, const part_plan::Item &item, bool empty) {
    SharedMemoryTaskHeader &tasks = *sched_->task_view.tasks;
    const ChipTaskSlotState &slot = tasks.get_slot_state_by_task_id(item.task);
    const TaskDescriptor &desc = slot.to_descriptor();
    const uint32_t set = static_cast<uint32_t>(pf.sets.size());
    pf.sets.push_back(
        asimgq::SimLocalSet{
            static_cast<uint32_t>(pf.ranges.size()), item.named, item.threshold, item.gid, empty,
            (item.flags & part_plan::kFlagAnnounce) != 0, (item.flags & part_plan::kFlagCohort) != 0
        }
    );
    for (uint32_t k = 0; k < item.named; ++k) {
        pf.ranges.push_back(asimgq::SimLocalRange{item.ranges[k].lo, item.ranges[k].hi});
    }
    pf.set_task.push_back(item.task);
    auto type_of = [](SubtaskSlot s) {
        return s == SubtaskSlot::AIC ? asimgq::SimTaskType::Cube : asimgq::SimTaskType::Vector;
    };
    if (item.kind == part_plan::kSlot) {
        // A home slot stands for the whole task: one entry, which completes every
        // subtask of it when it retires.
        SubtaskSlot first = SubtaskSlot::AIC;
        for (int32_t sub = 0; sub < SUBTASK_SLOT_COUNT; ++sub) {
            if (slot.active_mask.subtask_active(static_cast<SubtaskSlot>(sub))) {
                first = static_cast<SubtaskSlot>(sub);
                break;
            }
        }
        pf.entries.push_back(
            asimgq::SimLocalEntry{
                0, desc.kernel_id[static_cast<int32_t>(first)], item.task, type_of(first), set, item.local
            }
        );
        pf.subslots.push_back(first);
        pf.set_blocks.push_back(slot.logical_block_num);
        pf.set_precount.push_back(static_cast<int16_t>(slot.total_required_subtasks - 1));
        return;
    }
    const int32_t subs = active_subtask_count(slot);
    for (uint32_t blk = item.block_lo; blk < item.block_hi; ++blk) {
        int32_t ordinal = 0;
        for (int32_t sub = 0; sub < SUBTASK_SLOT_COUNT; ++sub) {
            const SubtaskSlot subslot = static_cast<SubtaskSlot>(sub);
            if (!slot.active_mask.subtask_active(subslot)) continue;
            pf.entries.push_back(
                asimgq::SimLocalEntry{
                    0, desc.kernel_id[sub], item.task, type_of(subslot), set,
                    item.local + (blk - item.block_lo) * static_cast<uint32_t>(subs) + static_cast<uint32_t>(ordinal)
                }
            );
            pf.subslots.push_back(subslot);
            ++ordinal;
        }
    }
    pf.set_blocks.push_back(static_cast<int32_t>(item.block_hi - item.block_lo));
    pf.set_precount.push_back(0);
}

bool SchedulerContext::submit_staged(int32_t thread_idx, int32_t part) {
    SharedMemoryTaskHeader &tasks = *sched_->task_view.tasks;
    PartFeed &pf = part_feed_[thread_idx];
    GqIndexSpace &space = gq_index_[thread_idx];
    const uint32_t q = static_cast<uint32_t>(thread_idx);
    const uint64_t from = asimgq::now_ticks();
    const int32_t book_core = core_trackers_[thread_idx].get_core_id_by_offset(0);
    for (uint32_t i = 0; i < pf.entries.size(); ++i) {
        ChipTaskSlotState &slot = tasks.get_slot_state_by_task_id(pf.entries[i].task_id);
        pf.entries[i].index = space.assign(
            GqIndexSpace::Owner{&slot, feed_seq_[thread_idx]++, pf.subslots[i], book_core, -1, false, false}
        );
    }
    // Account before the entries reach the queue (see account_published_blocks).
    for (size_t j = 0; j < pf.sets.size(); ++j) {
        sched_->account_published_blocks(tasks.get_slot_state_by_task_id(pf.set_task[j]), pf.set_blocks[j]);
    }
    const uint64_t submit_from = asimgq::now_ticks();
    const uint64_t ledger_from = asimgq::queue_self_overrun_ticks(q);
    const bool ok = asimgq::submit_local(
        q, part, pf.entries.data(), static_cast<uint32_t>(pf.entries.size()), pf.sets.data(),
        static_cast<uint32_t>(pf.sets.size()), pf.ranges.data(), static_cast<uint32_t>(pf.ranges.size())
    );
    pf.submit_ticks += asimgq::now_ticks() - submit_from;
    pf.submit_ledger_ticks += asimgq::queue_self_overrun_ticks(q) - ledger_from;
    if (!ok) {
        for (size_t j = pf.entries.size(); j-- > 0;) {
            space.unreserve(pf.entries[j].index);
        }
        LOG_ERROR(
            "Thread %d: partition %d store refused %zu entries (room %u)", thread_idx, part, pf.entries.size(),
            asimgq::part_store_room(q)
        );
        ++pf.wait_room;
    } else {
        // Its one entry retires a home slot's every subtask, and no other entry of the
        // task exists, so the rest are counted now.
        for (size_t j = 0; j < pf.sets.size(); ++j) {
            if (pf.set_precount[j] > 0) {
                tasks.get_slot_state_by_task_id(pf.set_task[j])
                    .completed_subtasks.fetch_add(pf.set_precount[j], std::memory_order_relaxed);
            }
        }
        pf.pieces += pf.sets.size();
        pf.entries_posted += pf.entries.size();
    }
    pf.entries.clear();
    pf.sets.clear();
    pf.ranges.clear();
    pf.subslots.clear();
    pf.set_task.clear();
    pf.set_blocks.clear();
    pf.set_precount.clear();
    pf.post_ticks += asimgq::now_ticks() - from;
    return ok;
}

void SchedulerContext::decide_chain(int32_t thread_idx, uint32_t chain, bool runs) {
    PartFeed &pf = part_feed_[thread_idx];
    if (chain >= pf.chain_seen.size() || pf.chain_seen[chain] != 0) return;
    pf.chain_seen[chain] = 1;
    const part_plan::Header &h = *part_plan::at<part_plan::Header>(plan_, 0);
    const part_plan::Chain &c =
        *part_plan::at<part_plan::Chain>(plan_, h.chain_offset + chain * part_plan::words_of<part_plan::Chain>());
    if (runs) {
        // The k-th chain of a partition to run goes to group k, counted from the
        // partition's own number, so no group takes more than its share of any one.
        const int32_t k = chains_dealt_seq_[c.part].fetch_add(1, std::memory_order_relaxed);
        const uint32_t g = (static_cast<uint32_t>(k) + c.part) % static_cast<uint32_t>(active_sched_threads_);
        chains_dealt_[c.part][g].fetch_add(1, std::memory_order_relaxed);
        const uint64_t t = route_tail_[g].fetch_add(1, std::memory_order_relaxed);
        std::atomic<uint64_t> &seq = route_seq_[g][t % kRouteRing];
        // The slot's ticket of a lap ago is posted before this one is written.
        while (seq.load(std::memory_order_acquire) != t) {
#if defined(__aarch64__)
            __asm__ volatile("yield");
#endif
        }
        route_ring_[g][t % kRouteRing] = chain;
        seq.store(t + 1, std::memory_order_release);
        ++pf.chains_dealt;
    }
    // Every chain dealt is counted to its group before the partition counts it
    // decided, so a manager that sees all decided sees every one dealt to it.
    chains_decided_[c.part].fetch_add(1, std::memory_order_release);
    ++pf.chains_decided;
}

// Post the current partition's next items, in order, as far as there is room. A home
// slot or predicated piece not yet decided is held back and the items behind it go
// on; a piece naming a producer the controllers do not run stops the posting until
// that producer has finished.
bool SchedulerContext::post_in_order(int32_t thread_idx) {
    SharedMemoryTaskHeader &tasks = *sched_->task_view.tasks;
    PartFeed &pf = part_feed_[thread_idx];
    const part_plan::SubPartition &sp = plan_sub(pf.cur, thread_idx);
    const uint32_t room = part_room(thread_idx, pf.cur);
    const uint32_t first_item = pf.cur_item;
    const size_t held_before = pf.held.size();
    uint32_t staged = 0;
    bool progress = false;
    bool outside = false;
    while (pf.cur_item < sp.items) {
        const uint32_t off = sp.item_offset + pf.cur_item * part_plan::words_of<part_plan::Item>();
        const part_plan::Item &item = *part_plan::at<part_plan::Item>(plan_, off);
        const ChipTaskSlotState &slot = tasks.get_slot_state_by_task_id(item.task);
        const uint32_t need = item_entries(item, slot);
        bool empty = false;
        if (item.kind == part_plan::kSlot || (item.flags & part_plan::kFlagPredicated) != 0) {
            bool runs = false;
            if (!predicate_decided(tasks, item.task, pf.walk, runs)) {
                pf.held.push_back(PartHeld{pf.cur, off, need});
                pf.held_entries[static_cast<size_t>(pf.cur)] += need;
                ++pf.held_count;
                ++pf.cur_item;
                progress = true;
                continue;
            }
            if ((item.flags & part_plan::kFlagDecides) != 0) decide_chain(thread_idx, item.chain, runs);
            if (item.kind == part_plan::kSlot && runs) {
                // Its chain runs where it is dealt; the map completes this slot.
                ++pf.slots_routed;
                ++pf.cur_item;
                progress = true;
                continue;
            }
            empty = !runs;
        } else if ((item.flags & part_plan::kFlagHold) != 0 && !outside_producers_done(tasks, item.task, pf.walk)) {
            // A producer the controllers do not run, which the managers dispatch on
            // the ordinary path, takes its positions only once its own producers have
            // finished. Nothing behind its consumer may take positions before it, or
            // the prefix that reports the producer's finish would wait on work that
            // waits on it.
            ++pf.wait_outside;
            outside = true;
            if (pf.hold_since == 0) {
                const uint32_t q = static_cast<uint32_t>(thread_idx);
                pf.hold_since = asimgq::now_ticks() - asimgq::queue_self_overrun_ticks(q);
            }
            break;
        } else if ((item.flags & part_plan::kFlagHold) != 0 && pf.hold_since != 0) {
            const uint32_t q = static_cast<uint32_t>(thread_idx);
            pf.hold_wait_ticks += asimgq::now_ticks() - asimgq::queue_self_overrun_ticks(q) - pf.hold_since;
            ++pf.hold_waits;
            pf.hold_since = 0;
        }
        if (staged + need > room) break;
        if (item.kind == part_plan::kSlot) ++pf.slots_skipped;
        stage_item(pf, item, empty);
        staged += need;
        ++pf.cur_item;
    }
    if (!pf.sets.empty()) {
        if (submit_staged(thread_idx, pf.cur)) {
            progress = true;
        } else {
            // Whatever this pass staged or held is taken again from where it started.
            for (size_t i = held_before; i < pf.held.size(); ++i) {
                pf.held_entries[static_cast<size_t>(pf.held[i].part)] -= pf.held[i].entries;
            }
            pf.held_count -= pf.held.size() - held_before;
            pf.held.resize(held_before);
            pf.cur_item = first_item;
            return false;
        }
    } else if (pf.cur_item < sp.items && !progress && !outside) {
        ++pf.wait_room;
    }
    return progress;
}

// Post held items that can go now, the lowest partition's first, once a completion
// may have decided one.
bool SchedulerContext::feed_held(int32_t thread_idx) {
    PartFeed &pf = part_feed_[thread_idx];
    if (pf.held.empty()) return false;
    const uint64_t epoch = g_completion_epoch.load(std::memory_order_acquire);
    if (epoch == pf.held_epoch) return false;
    pf.held_epoch = epoch;
    SharedMemoryTaskHeader &tasks = *sched_->task_view.tasks;
    bool progress = false;
    const int32_t part = pf.held.front().part;
    uint32_t room = part_room(thread_idx, part);
    std::vector<PartHeld> keep;
    std::vector<PartHeld> staged_items;
    for (const PartHeld &h : pf.held) {
        const part_plan::Item &item = *part_plan::at<part_plan::Item>(plan_, h.item);
        bool empty = false;
        if (h.part != part) {
            keep.push_back(h);
            continue;
        }
        if (item.kind == part_plan::kSlot || (item.flags & part_plan::kFlagPredicated) != 0) {
            bool runs = false;
            if (!predicate_decided(tasks, item.task, pf.walk, runs)) {
                keep.push_back(h);
                continue;
            }
            if ((item.flags & part_plan::kFlagDecides) != 0) decide_chain(thread_idx, item.chain, runs);
            if (item.kind == part_plan::kSlot && runs) {
                ++pf.slots_routed;
                pf.held_entries[static_cast<size_t>(h.part)] -= h.entries;
                progress = true;
                continue;
            }
            empty = !runs;
        }
        if (h.entries > room) {
            keep.push_back(h);
            continue;
        }
        if (item.kind == part_plan::kSlot) ++pf.slots_skipped;
        stage_item(pf, item, empty);
        staged_items.push_back(h);
        room -= h.entries;
    }
    if (!pf.sets.empty()) {
        if (submit_staged(thread_idx, part)) {
            for (const PartHeld &h : staged_items) {
                pf.held_entries[static_cast<size_t>(h.part)] -= h.entries;
            }
            pf.held_fed += staged_items.size();
            progress = true;
        } else {
            keep.insert(keep.end(), staged_items.begin(), staged_items.end());
            std::sort(keep.begin(), keep.end(), [](const PartHeld &x, const PartHeld &y) {
                return x.part != y.part ? x.part < y.part : x.item < y.item;
            });
            pf.held_epoch = ~0ULL;
        }
    }
    pf.held.swap(keep);
    return progress;
}

// Post the chains dealt to this group, in the order dealt, each once every
// controller maps its partition and its producers outside it are complete.
bool SchedulerContext::feed_chains(int32_t thread_idx) {
    SharedMemoryTaskHeader &tasks = *sched_->task_view.tasks;
    PartFeed &pf = part_feed_[thread_idx];
    const uint32_t g = static_cast<uint32_t>(thread_idx);
    const part_plan::Header &h = *part_plan::at<part_plan::Header>(plan_, 0);
    bool progress = false;
    for (;;) {
        const uint64_t t = pf.route_cursor;
        std::atomic<uint64_t> &seq = route_seq_[g][t % kRouteRing];
        if (seq.load(std::memory_order_acquire) != t + 1) break;
        const uint32_t chain = route_ring_[g][t % kRouteRing];
        const part_plan::Chain &c =
            *part_plan::at<part_plan::Chain>(plan_, h.chain_offset + chain * part_plan::words_of<part_plan::Chain>());
        const int32_t part = static_cast<int32_t>(c.part);
        if (part >= pf.installed || part_installed_[part].load(std::memory_order_acquire) < active_sched_threads_) {
            ++pf.wait_gate;
            break;
        }
        const int32_t *ext = reinterpret_cast<const int32_t *>(plan_ + c.ext_offset);
        bool ready = true;
        for (uint32_t i = 0; i < c.external && ready; ++i) {
            ready = tasks.is_completed(ext[i]);
        }
        if (!ready) {
            ++pf.chain_wait_ext;
            break;
        }
        if (c.ids > part_room(thread_idx, part)) {
            ++pf.wait_room;
            break;
        }
        const uint32_t base = pf.dyn_next[static_cast<size_t>(part)];
        for (uint32_t k = 0; k < c.tasks; ++k) {
            const part_plan::ChainTask &ct = *part_plan::at<part_plan::ChainTask>(
                plan_, c.task_offset + k * part_plan::words_of<part_plan::ChainTask>()
            );
            const ChipTaskSlotState &slot = tasks.get_slot_state_by_task_id(ct.task);
            part_plan::Item item{};
            item.task = ct.task;
            item.local = base + ct.offset;
            item.block_lo = 0;
            item.block_hi = static_cast<uint32_t>(slot.logical_block_num);
            item.kind = part_plan::kPiece;
            item.flags = part_plan::kFlagAnnounce;
            item.gid = ct.gid;
            item.named = ct.named;
            for (uint32_t r = 0; r < ct.named; ++r) {
                item.ranges[r] = part_plan::Range{base + ct.ranges[r].lo, base + ct.ranges[r].hi};
            }
            stage_item(pf, item, false);
        }
        if (!submit_staged(thread_idx, part)) break;
        pf.dyn_next[static_cast<size_t>(part)] = base + c.ids;
        ++pf.chains_posted[static_cast<size_t>(part)];
        seq.store(t + kRouteRing, std::memory_order_release);
        ++pf.route_cursor;
        ++pf.chains_in;
        progress = true;
    }
    return progress;
}

// Mark the unused ids of each partition's dynamic region finished, in partition
// order, once every chain of the partition is decided and every one dealt here is
// posted.
bool SchedulerContext::feed_fills(int32_t thread_idx) {
    PartFeed &pf = part_feed_[thread_idx];
    bool progress = false;
    while (pf.filled < pf.installed) {
        const int32_t p = pf.filled;
        const part_plan::SubPartition &sp = plan_sub(p, thread_idx);
        if (sp.chain_count > 0) {
            if (chains_decided_[p].load(std::memory_order_acquire) < static_cast<int32_t>(sp.chain_count)) break;
            if (pf.chains_posted[static_cast<size_t>(p)] <
                chains_dealt_[p][thread_idx].load(std::memory_order_relaxed)) {
                break;
            }
        }
        const uint32_t end = sp.dyn_lo + sp.dyn_count;
        if (pf.dyn_next[static_cast<size_t>(p)] < end) {
            asimgq::fill_local(static_cast<uint32_t>(thread_idx), pf.dyn_next[static_cast<size_t>(p)], end);
            ++pf.fills;
        }
        ++pf.filled;
        progress = true;
    }
    return progress;
}

// Feed this thread's controller from the partition plan. A partition's map goes in
// once the controller is within one partition of it, so a controller maps at most
// two; its items go out once every controller maps it, and no further ahead than
// the partition after the one the controller is in.
bool SchedulerContext::feed_partitions(int32_t thread_idx) {
    if (!part_active_ || plan_ == nullptr || sched_->task_view.tasks == nullptr) return false;
    PartFeed &pf = part_feed_[thread_idx];
    const uint32_t q = static_cast<uint32_t>(thread_idx);
    const int32_t parts = static_cast<int32_t>(part_plan::at<part_plan::Header>(plan_, 0)->partitions);
    bool progress = false;
    const int32_t at = asimgq::queue_partition(q);
    while (pf.installed < parts && pf.installed <= at + 1) {
        const part_plan::SubPartition &sp = plan_sub(pf.installed, thread_idx);
        const auto *map = reinterpret_cast<const asimgq::SimMapEntry *>(plan_ + sp.map_offset);
        if (!asimgq::install_subpartition(q, pf.installed, sp.base, sp.span, map, sp.map)) {
            ++pf.wait_install;
            break;
        }
        part_installed_[pf.installed].fetch_add(1, std::memory_order_release);
        pf.map_entries += sp.map;
        ++pf.installed;
        progress = true;
    }
    if (feed_chains(thread_idx)) progress = true;
    if (feed_held(thread_idx)) progress = true;
    while (pf.cur < parts) {
        const part_plan::SubPartition &sp = plan_sub(pf.cur, thread_idx);
        if (pf.cur_item >= sp.items) {
            ++pf.cur;
            pf.cur_item = 0;
            ++pf.partitions;
            continue;
        }
        if (pf.cur >= pf.installed) break;
        if (part_installed_[pf.cur].load(std::memory_order_acquire) < active_sched_threads_) {
            ++pf.wait_gate;
            break;
        }
        if (pf.cur > at + 1) {
            ++pf.wait_part;
            break;
        }
        if (!post_in_order(thread_idx)) break;
        progress = true;
    }
    if (feed_fills(thread_idx)) progress = true;
    return progress;
}

// Everything the GroupQueue carries that is scoped to one run. Positions and
// group claims both name state a run resets -- every queue's watermark goes back
// to zero -- so one carried over from an earlier round would wait on a prefix
// that no longer exists. Runs on the leader once the arena's slot arrays are
// wired and seeded, and before any thread is released into its dispatch loop.
void SchedulerContext::gq_prepare_run() {
#ifdef __SIMULATED_DEVICE__
    silo_active_ = sched_->task_view.tasks != nullptr && sched_->task_view.tasks->silo_tasks > 0;
    part_active_ = silo_active_ && sched_->task_view.tasks->partitions > 0 && sched_->task_view.tasks->plan_words > 0;
    plan_ = nullptr;
    if (part_active_) {
        // The plan is numbered for a fixed set of groups; a run with another number of
        // managers feeds the silos without it.
        const SharedMemoryTaskHeader &tasks = *sched_->task_view.tasks;
        plan_ = reinterpret_cast<const uint32_t *>(reinterpret_cast<const char *>(&tasks) + tasks.plan_offset);
        const part_plan::Header &h = *part_plan::at<part_plan::Header>(plan_, 0);
        if (h.words != tasks.plan_words || h.groups != part_plan::kGroups ||
            h.partitions != static_cast<uint32_t>(tasks.partitions) ||
            active_sched_threads_ != static_cast<int32_t>(part_plan::kGroups) ||
            tasks.partitions > kMaxPlanPartitions) {
            LOG_ERROR(
                "Partition plan for %u groups and %u partitions does not fit %d managers; feeding silos without it",
                h.groups, h.partitions, active_sched_threads_
            );
            part_active_ = false;
            plan_ = nullptr;
        }
    }
    // The plan's sync-start tasks, in the order the controllers are given their
    // pieces. Every block of a piece holds one of its group's clusters until the
    // release, so a piece wider than its group could never be released.
    part_cohorts_.clear();
    if (part_active_) {
        const part_plan::Header &h = *part_plan::at<part_plan::Header>(plan_, 0);
        bool fits = true;
        for (uint32_t k = 0; k < h.cohorts && fits; ++k) {
            const part_plan::Cohort &pc = *part_plan::at<part_plan::Cohort>(
                plan_, h.cohort_offset + k * part_plan::words_of<part_plan::Cohort>()
            );
            PartCohort c{};
            c.task = pc.task;
            for (uint32_t g = 0; g < part_plan::kGroups; ++g) {
                c.share[g] = static_cast<int32_t>(pc.blocks[g] * pc.subs);
                c.entries += c.share[g];
                if (static_cast<int32_t>(pc.blocks[g]) > core_trackers_[g].get_cluster_count()) fits = false;
            }
            part_cohorts_.push_back(c);
        }
        if (!fits) {
            LOG_ERROR("A sync-start piece of the partition plan is wider than its group; feeding silos without it");
            part_cohorts_.clear();
            part_active_ = false;
            plan_ = nullptr;
        }
    }
    part_cohort_cur_.store(0, std::memory_order_relaxed);
    part_cohort_at_.store(0, std::memory_order_relaxed);
    part_cohort_released_.store(0, std::memory_order_relaxed);
    for (int32_t i = 0; i < MAX_AICPU_THREADS; i++) {
        part_cohort_staged_[i].store(0, std::memory_order_relaxed);
    }
    // A silo's share of a partition is resident whole, with the head of the next, so
    // its positions span both: the look-ahead horizon bounds the window rather than
    // the 32 hold slots.
    for (int32_t i = 0; i < active_sched_threads_; i++) {
        gq_index_[i].init(part_active_ ? asimgq::SIM_AHEAD_HORIZON : asimgq::SIM_HELD_CAP);
        for (int32_t off = 0; off < kGqOffsets; off++) {
            belled_held_[i][off] = 0;
            belled_deferred_[i][off] = 0;
        }
    }
    ++g_position_epoch;  // every position recorded by an earlier run is now stale
    gq_group::reset_group_owners();
    if (silo_active_ && !part_active_) size_piece_positions(total_tasks_);
    asimgq::set_peer_forwarding(silo_active_ && !part_active_);
    asimgq::set_partition_mode(part_active_);
    for (int32_t i = 0; i < MAX_AICPU_THREADS; i++) {
        silo_feed_[i] = SiloFeed{};
        part_feed_[i] = PartFeed{};
    }
    if (part_active_) {
        const part_plan::Header &h = *part_plan::at<part_plan::Header>(plan_, 0);
        for (int32_t i = 0; i < active_sched_threads_; i++) {
            PartFeed &pf = part_feed_[i];
            pf.held_entries.assign(h.partitions, 0);
            pf.chains_posted.assign(h.partitions, 0);
            pf.chain_seen.assign(h.chains, 0);
            pf.dyn_next.resize(h.partitions);
            for (uint32_t p = 0; p < h.partitions; ++p) {
                pf.dyn_next[p] = plan_sub(static_cast<int32_t>(p), i).dyn_lo;
            }
        }
        for (uint32_t p = 0; p < h.partitions; ++p) {
            part_installed_[p].store(0, std::memory_order_relaxed);
            chains_decided_[p].store(0, std::memory_order_relaxed);
            chains_dealt_seq_[p].store(0, std::memory_order_relaxed);
            for (uint32_t g = 0; g < part_plan::kGroups; ++g) {
                chains_dealt_[p][g].store(0, std::memory_order_relaxed);
            }
        }
        // Each route ring restarts at ticket 0: a slot is free for ticket t when its
        // sequence reads t. Only the slots the last run used need setting again.
        for (uint32_t g = 0; g < part_plan::kGroups; ++g) {
            const uint64_t used = route_slots_set_ ?
                                      std::min<uint64_t>(route_tail_[g].load(std::memory_order_relaxed), kRouteRing) :
                                      kRouteRing;
            for (uint64_t i = 0; i < used; ++i) {
                route_seq_[g][i].store(i, std::memory_order_relaxed);
            }
            route_tail_[g].store(0, std::memory_order_relaxed);
        }
        route_slots_set_ = true;
    }
    cohort_.task_id.store(0, std::memory_order_relaxed);
    cohort_.state.store(SyncStartCohort::kIdle, std::memory_order_relaxed);
    cohort_.target.store(0, std::memory_order_relaxed);
    cohort_.release_at.store(0, std::memory_order_relaxed);
    cohort_.claimed_at.store(0, std::memory_order_relaxed);
    for (int32_t i = 0; i < MAX_AICPU_THREADS; i++) {
        cohort_.staged_by[i].store(0, std::memory_order_relaxed);
        cohort_claim_tid_[i] = 0;
        cohort_claim_n_[i] = 0;
        cohort_claim_entries_[i] = 0;
        cohort_held_seen_[i] = 0;
        cohort_stamped_[i] = 0;
        cohort_assembly_ticks_[i] = 0;
        cohort_blocks_[i] = 0;
    }
    // Seeding first: the partition below applies only to a run that declared
    // groups, and that is what seeding determines.
    seed_ready_groups();
    partition_ready_queues_by_owner();
#endif
}

// Give each Scheduler thread its own span of every shape's ready-queue slot
// array. The spans are disjoint and equal, so a thread's pushes and pops touch
// no other thread's slots and the whole partition still fits the one region the
// arena already reserved. Each span is re-seeded to its own empty ramp because
// the ramp a queue's push tests against is a function of its capacity.
void SchedulerContext::partition_ready_queues_by_owner() {
    if (!gq_group::active()) return;
    const int32_t threads = active_sched_threads_ > 0 ? active_sched_threads_ : 1;
    for (int32_t shape = 0; shape < NUM_RESOURCE_SHAPES; ++shape) {
        ChipReadyQueue &whole = sched_->ready_queues[shape];
        if (whole.slots == nullptr || whole.capacity == 0) continue;
        uint64_t span = whole.capacity / static_cast<uint64_t>(threads);
        // push_tagged's claim test indexes with a mask, so a span must be a power
        // of two; round down to the largest one that fits.
        uint64_t pow2 = 1;
        while (pow2 * 2 <= span)
            pow2 *= 2;
        span = pow2;
        for (int32_t t = 0; t < MAX_AICPU_THREADS; ++t) {
            ChipReadyQueue &q = sched_->ready_queues_gq[t][shape];
            q.slots = (t < threads) ? whole.slots + static_cast<uint64_t>(t) * span : nullptr;
            q.capacity = (t < threads) ? span : 0;
            q.mask = (t < threads) ? span - 1 : 0;
            q.enqueue_pos.store(0, std::memory_order_relaxed);
            q.dequeue_pos.store(0, std::memory_order_relaxed);
            q.max_occupancy.store(0, std::memory_order_relaxed);
            if (t < threads) q.seed_slots();
        }
    }
}

int32_t SchedulerContext::handle_timeout_exit(
    int32_t thread_idx, SharedMemoryHeader *header, Runtime *runtime, int32_t idle_iterations,
    int32_t last_progress_count
#if SIMPLER_DFX
    ,
    uint64_t sched_start_ts
#endif
) {
    LOG_ERROR(
        "[STALL thread=%d idle_iterations=%d] TIMEOUT_EXIT after_idle_iterations=%d", thread_idx, idle_iterations,
        idle_iterations
    );
    latch_scheduler_error(header, thread_idx, SIMPLER_ERROR_SCHEDULER_TIMEOUT);
#ifdef __SIMULATED_DEVICE__
    {
        uint64_t admitted = 0, promoted = 0, refused = 0;
        uint32_t high = 0;
        asimgq::queue_group_stats(static_cast<uint32_t>(thread_idx), &admitted, &promoted, &refused, &high);
        LOG_ERROR(
            "[GQ_GROUP thread=%d] admitted=%" PRIu64 " promoted=%" PRIu64 " refused=%" PRIu64 " held_high=%u",
            thread_idx, admitted, promoted, refused, high
        );
        LOG_ERROR(
            "[GQ_STUCK thread=%d] group=%d id=%d block=%d sub=%d head=%d room=%u heldroom=%u push=%" PRIu64
            " wm=%" PRIu64 " named=%" PRIu64 " toomany=%" PRIu64 " bailed=%" PRIu64
            " bail_task=%d bail_prod=%d bail_kind=%d",
            thread_idx, group_feed_[thread_idx].group, group_feed_[thread_idx].id, group_feed_[thread_idx].block,
            group_feed_[thread_idx].sub, gq_group::g_scan_from.load(std::memory_order_relaxed),
            gq_index_[thread_idx].room(), asimgq::read_queue_status(static_cast<uint32_t>(thread_idx)).held_room,
            gq_index_[thread_idx].push_index(), gq_index_[thread_idx].watermark(),
            g_gq_deps_named.load(std::memory_order_relaxed), g_gq_deps_toomany.load(std::memory_order_relaxed),
            g_gq_deps_bailed.load(std::memory_order_relaxed), g_gq_bail_task.load(std::memory_order_relaxed),
            g_gq_bail_prod.load(std::memory_order_relaxed), g_gq_bail_kind.load(std::memory_order_relaxed)
        );
        if (sched_->task_view.tasks != nullptr) {
            // Why the scan point cannot advance: name the first external producer of
            // the group sitting there that has not retired, and say what it is.
            SharedMemoryTaskHeader &tk = *sched_->task_view.tasks;
            const int32_t g = gq_group::g_scan_from.load(std::memory_order_relaxed);
            const int32_t lo = gq_group::group_member(g, 0);
            const int32_t hi = lo + gq_group::group_member_count(g);
            bool named = false;
            for (int32_t id = lo; id < hi && id < total_tasks_ && !named; ++id) {
                const TaskPayload &pp = tk.get_slot_state_by_task_id(id).to_payload();
                const int32_t *fi = pp.fanin_data();
                for (int32_t k = 0; k < pp.fanin_count; ++k) {
                    const int32_t pr = fi[k];
                    if (gq_group::group_of(pr) == g || tk.is_completed(pr)) continue;
                    ChipTaskSlotState &ps = tk.get_slot_state_by_task_id(pr);
                    LOG_ERROR(
                        "[GQ_BLOCK] group=%d task=%d waits on ext prod=%d (group %d) blocks=%d shape=%d "
                        "delivered=%d owner=%d opened=%d published=%d",
                        g, id, pr, gq_group::group_of(pr), static_cast<int>(ps.logical_block_num),
                        static_cast<int>(ps.active_mask.to_shape()), group_queue_delivers(ps) ? 1 : 0,
                        gq_group::group_owner(gq_group::group_of(pr)),
                        gq_group::group_is_opened(gq_group::group_of(pr)) ? 1 : 0, tk.is_published(pr) ? 1 : 0
                    );
                    named = true;
                    break;
                }
            }
            if (!named) LOG_ERROR("[GQ_BLOCK] group=%d has every external producer retired", g);
        }
        {
            const int32_t bp = g_gq_bail_prod.load(std::memory_order_relaxed);
            if (bp >= 0 && sched_->task_view.tasks != nullptr && bp < total_tasks_) {
                SharedMemoryTaskHeader &tk = *sched_->task_view.tasks;
                ChipTaskSlotState &bs = tk.get_slot_state_by_task_id(bp);
                LOG_ERROR(
                    "[GQ_BAILPROD] prod=%d blocks=%d kind=%d shape=%d completed=%d published=%d pos=%llu", bp,
                    static_cast<int>(bs.logical_block_num), static_cast<int>(bs.task_kind),
                    static_cast<int>(bs.active_mask.to_shape()), tk.is_completed(bp) ? 1 : 0,
                    tk.is_published(bp) ? 1 : 0, (unsigned long long)task_position(bp)
                );
            }
        }
    }
#endif
    if (!completed_.exchange(true, std::memory_order_acq_rel)) {
        log_shutdown_stall_snapshot(thread_idx, idle_iterations, last_progress_count);
#if SIMPLER_DFX
        // Capture the in-flight kernels' partial output before signalling the
        // cores to exit, so the dump reflects the live stuck state.
        if (is_dump_args_enabled()) {
            dump_running_task_outputs(
                cores_total_num_,
                [this](int32_t cid) {
                    return core_exec_states_[cid].running_slot_state;
                },
                [this, thread_idx](const ChipTaskSlotState &slot_state) {
                    dump_args_for_task<SUBTASK_SLOT_COUNT>(
                        thread_idx, slot_state.to_descriptor(), slot_state.to_payload(), slot_state.active_mask,
                        ArgsDumpStage::AFTER_COMPLETION,
                        [](ActiveMask active_mask, int raw_subtask_id) {
                            return active_mask.subtask_active(static_cast<SubtaskSlot>(raw_subtask_id));
                        },
                        [this](int32_t func_id) {
                            return get_function_bin_addr(func_id);
                        },
                        &slot_state.to_payload().dump_metadata
                    );
                }
            );
        }
#endif
        emergency_shutdown(runtime);
    }
#if SIMPLER_DFX
    uint64_t sched_timeout_ts = get_sys_cnt_aicpu();
    LOG_INFO(
        "Thread %d: sched_start=%" PRIu64 " sched_end(timeout)=%" PRIu64 " sched_cost=%.3fus", thread_idx,
        static_cast<uint64_t>(sched_start_ts), static_cast<uint64_t>(sched_timeout_ts),
        cycles_to_us(sched_timeout_ts - sched_start_ts)
    );
#endif
    return -SIMPLER_ERROR_SCHEDULER_TIMEOUT;
}

#if SIMPLER_DFX
void SchedulerContext::log_chip_swimlane_summary(int32_t thread_idx, int32_t cur_thread_completed) {
    auto &chip_swimlane = sched_chip_swimlane_[thread_idx];
    uint64_t sched_end_ts = get_sys_cnt_aicpu();
    LOG_INFO(
        "Thread %d: sched_start=%" PRIu64 " sched_end=%" PRIu64 " sched_cost=%.3fus", thread_idx,
        static_cast<uint64_t>(chip_swimlane.sched_start_ts), static_cast<uint64_t>(sched_end_ts),
        cycles_to_us(sched_end_ts - chip_swimlane.sched_start_ts)
    );

    uint64_t sched_total =
        chip_swimlane.sched_complete_cycle + chip_swimlane.sched_dispatch_cycle + chip_swimlane.sched_idle_cycle;
    if (sched_total == 0) sched_total = 1;

#if SIMPLER_SCHED_PROFILING
    {
        SchedProfilingData sp = scheduler_get_profiling(thread_idx);
        uint64_t otc_total = sp.lock_cycle + sp.fanout_cycle + sp.fanin_cycle + sp.self_consumed_cycle;
        uint64_t complete_poll =
            (chip_swimlane.sched_complete_cycle > otc_total + chip_swimlane.sched_complete_perf_cycle) ?
                (chip_swimlane.sched_complete_cycle - otc_total - chip_swimlane.sched_complete_perf_cycle) :
                0;
        uint64_t dispatch_poll = (chip_swimlane.sched_dispatch_cycle >
                                  chip_swimlane.sched_dispatch_pop_cycle + chip_swimlane.sched_dispatch_setup_cycle) ?
                                     (chip_swimlane.sched_dispatch_cycle - chip_swimlane.sched_dispatch_pop_cycle -
                                      chip_swimlane.sched_dispatch_setup_cycle) :
                                     0;

        LOG_INFO(
            "Thread %d: === Scheduler Phase Breakdown: total=%.3fus, %d tasks ===", thread_idx,
            cycles_to_us(sched_total), cur_thread_completed
        );

        // fanout / fanin per-thread aggregates live in
        // sched_overhead_analysis.compute_dag_stats_from_deps (deps.json edges
        // × core_to_thread).
        LOG_INFO(
            "Thread %d:   complete       : %.3fus (%.1f%%)", thread_idx,
            cycles_to_us(chip_swimlane.sched_complete_cycle), chip_swimlane.sched_complete_cycle * 100.0 / sched_total
        );

        uint64_t c_parent = chip_swimlane.sched_complete_cycle > 0 ? chip_swimlane.sched_complete_cycle : 1;
        uint64_t complete_miss_count = (chip_swimlane.complete_probe_count > chip_swimlane.complete_hit_count) ?
                                           (chip_swimlane.complete_probe_count - chip_swimlane.complete_hit_count) :
                                           0;
        double complete_hit_rate = chip_swimlane.complete_probe_count > 0 ?
                                       chip_swimlane.complete_hit_count * 100.0 / chip_swimlane.complete_probe_count :
                                       0.0;
        LOG_INFO(
            "Thread %d:     poll         : %.3fus (%.1f%%)  hit=%" PRIu64 ", miss=%" PRIu64 ", hit_rate=%.1f%%",
            thread_idx, cycles_to_us(complete_poll), complete_poll * 100.0 / c_parent,
            static_cast<uint64_t>(chip_swimlane.complete_hit_count), static_cast<uint64_t>(complete_miss_count),
            complete_hit_rate
        );
        LOG_INFO(
            "Thread %d:     otc_lock     : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "", thread_idx,
            cycles_to_us(sp.lock_cycle), sp.lock_cycle * 100.0 / c_parent,
            cycles_to_us(sp.lock_cycle - sp.lock_wait_cycle), cycles_to_us(sp.lock_wait_cycle),
            static_cast<uint64_t>(sp.lock_atomic_count)
        );
        LOG_INFO(
            "Thread %d:     otc_fanout   : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "", thread_idx,
            cycles_to_us(sp.fanout_cycle), sp.fanout_cycle * 100.0 / c_parent,
            cycles_to_us(sp.fanout_cycle - sp.push_wait_cycle), cycles_to_us(sp.push_wait_cycle),
            static_cast<uint64_t>(sp.fanout_atomic_count)
        );
        LOG_INFO(
            "Thread %d:     otc_fanin    : %.3fus (%.1f%%)  atomics=%" PRIu64 "", thread_idx,
            cycles_to_us(sp.fanin_cycle), sp.fanin_cycle * 100.0 / c_parent,
            static_cast<uint64_t>(sp.fanin_atomic_count)
        );
        LOG_INFO(
            "Thread %d:     otc_self     : %.3fus (%.1f%%)  atomics=%" PRIu64 "", thread_idx,
            cycles_to_us(sp.self_consumed_cycle), sp.self_consumed_cycle * 100.0 / c_parent,
            static_cast<uint64_t>(sp.self_atomic_count)
        );
        LOG_INFO(
            "Thread %d:     perf         : %.3fus (%.1f%%)", thread_idx,
            cycles_to_us(chip_swimlane.sched_complete_perf_cycle),
            chip_swimlane.sched_complete_perf_cycle * 100.0 / c_parent
        );

        LOG_INFO(
            "Thread %d:   dispatch       : %.3fus (%.1f%%)", thread_idx,
            cycles_to_us(chip_swimlane.sched_dispatch_cycle), chip_swimlane.sched_dispatch_cycle * 100.0 / sched_total
        );

        uint64_t d_parent = chip_swimlane.sched_dispatch_cycle > 0 ? chip_swimlane.sched_dispatch_cycle : 1;
        LOG_INFO(
            "Thread %d:     poll         : %.3fus (%.1f%%)", thread_idx, cycles_to_us(dispatch_poll),
            dispatch_poll * 100.0 / d_parent
        );
        LOG_INFO(
            "Thread %d:     pop          : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "", thread_idx,
            cycles_to_us(chip_swimlane.sched_dispatch_pop_cycle),
            chip_swimlane.sched_dispatch_pop_cycle * 100.0 / d_parent,
            cycles_to_us(chip_swimlane.sched_dispatch_pop_cycle - sp.pop_wait_cycle), cycles_to_us(sp.pop_wait_cycle),
            static_cast<uint64_t>(sp.pop_atomic_count)
        );
        LOG_INFO(
            "Thread %d:     setup        : %.3fus (%.1f%%)", thread_idx,
            cycles_to_us(chip_swimlane.sched_dispatch_setup_cycle),
            chip_swimlane.sched_dispatch_setup_cycle * 100.0 / d_parent
        );

        LOG_INFO(
            "Thread %d:   idle           : %.3fus (%.1f%%)", thread_idx, cycles_to_us(chip_swimlane.sched_idle_cycle),
            chip_swimlane.sched_idle_cycle * 100.0 / sched_total
        );

        if (cur_thread_completed > 0) {
            LOG_INFO(
                "Thread %d:   avg/complete   : %.3fus", thread_idx,
                cycles_to_us(chip_swimlane.sched_complete_cycle) / cur_thread_completed
            );
        }
    }
#endif
    LOG_INFO(
        "Thread %d: Scheduler summary: total_time=%.3fus, loops=%" PRIu64 ", tasks_scheduled=%d", thread_idx,
        cycles_to_us(sched_total), static_cast<uint64_t>(chip_swimlane.sched_loop_count), cur_thread_completed
    );
}
#endif

// =============================================================================
// Shutdown: each thread retires the cores it owns, on its own way out.
// Core ownership is a partition — assign_cores_to_threads hands every cluster
// to exactly one scheduler thread — so concurrent retirements never name the
// same core. Emergency shutdown sweeps the whole table and claims per core, so
// a core is retired exactly once no matter which path reaches it first.
// =============================================================================
int32_t SchedulerContext::shutdown(int32_t thread_idx, Runtime *runtime) {
    const int32_t *cores = core_trackers_[thread_idx].core_ids();
    const int32_t core_num = core_trackers_[thread_idx].core_num();
    if (core_num == 0) return 0;  // threads that own no core

#if SIMPLER_DFX
    if (is_pmu_enabled()) {
        pmu_aicpu_finalize(cores, core_num);
    }
#endif
#ifdef __SIMULATED_DEVICE__
    {
        // One line per thread and two per run, the same as the M0 arm prints: these
        // lines are written inside the device_wall the arm reports, so an arm that
        // printed more would carry the difference into its measurement.
        //
        // The compute issued, compared across an A/B, is what says both legs ran
        // the same graph, which a skip-golden run cannot. poll_overrun_us plus
        // push_overrun_us is the model's excess this run: exactly what the phase
        // recorder removes from this thread's window.
        const uint32_t q = static_cast<uint32_t>(thread_idx);
        uint64_t aic = 0, aiv = 0;
        asimgq::queue_busy_ticks(q, &aic, &aiv);
        uint32_t ahead_hw = 0;
        uint64_t ahead_dropped = 0;
        asimgq::queue_ahead_stats(q, &ahead_hw, &ahead_dropped);
        uint64_t poll_over_us = 0, poll_over_calls = 0;
        asimgq::queue_poll_overrun(q, &poll_over_us, &poll_over_calls);
        uint64_t steals_tried = 0, steals_won = 0;
        int64_t steal_gain_us = 0;
        asimgq::queue_steal_stats(q, &steals_tried, &steals_won, &steal_gain_us);
        uint64_t work_total = 0, work_max = 0, work_calls = 0;
        asimgq::queue_poll_work(q, &work_total, &work_max, &work_calls);
        uint64_t push_over_us = 0, push_over_calls = 0, push_mean_ns = 0;
        asimgq::queue_push_overrun(q, &push_over_us, &push_over_calls, &push_mean_ns);
        uint64_t admitted = 0, promoted = 0, refused = 0;
        uint32_t held_high = 0;
        asimgq::queue_group_stats(q, &admitted, &promoted, &refused, &held_high);
        uint64_t bells = 0, bells_waiting = 0;
        uint64_t hold_cube = 0, hold_vector = 0;
        asimgq::queue_cohort_hold(q, &hold_cube, &hold_vector);
        asimgq::doorbell_counts(q, &bells, &bells_waiting);
        LOG_INFO(
            "[GQ_WORK thread=%d] aic_busy_us=%.1f aiv_busy_us=%.1f positions=%" PRIu64 " poll_overrun_us=%" PRIu64
            " push_overrun_us=%" PRIu64 " polls=%" PRIu64 " poll_mean_ns=%" PRIu64 " poll_max_ns=%" PRIu64
            " ahead_high=%u ahead_dropped=%" PRIu64 " held_admitted=%" PRIu64 " held_promoted=%" PRIu64
            " held_refused=%" PRIu64 " held_high=%u steals_tried=%" PRIu64 " steals_won=%" PRIu64
            " steal_gain_us=%" PRId64 " cohorts=%" PRIu64 " cohort_blocks=%" PRIu64 " cohort_assembly_us=%.1f"
            " cohort_piped=%" PRIu64 " cohort_hold_us=%.1f/%.1f bells=%" PRIu64 " bells_waiting=%" PRIu64,
            thread_idx, cycles_to_us(aic), cycles_to_us(aiv), gq_index_[thread_idx].push_index(), poll_over_us,
            push_over_us, work_calls, work_calls ? (uint64_t)(cycles_to_us(work_total) * 1000.0 / work_calls) : 0,
            (uint64_t)(cycles_to_us(work_max) * 1000.0), ahead_hw, ahead_dropped, admitted, promoted, refused,
            held_high, steals_tried, steals_won, steal_gain_us, cohort_stamped_[thread_idx], cohort_blocks_[thread_idx],
            cycles_to_us(cohort_assembly_ticks_[thread_idx]), asimgq::queue_cohort_piped(q), cycles_to_us(hold_cube),
            cycles_to_us(hold_vector), bells, bells_waiting
        );
        if (part_active_) {
            const PartFeed &pf = part_feed_[thread_idx];
            asimgq::SimPartStats ps{};
            asimgq::queue_part_stats(q, &ps);
            LOG_INFO(
                "[GQ_PART thread=%d] partitions=%" PRIu64 " pieces=%" PRIu64 " entries=%" PRIu64 " map_entries=%" PRIu64
                " slots_skipped=%" PRIu64 " slots_routed=%" PRIu64 " held=%" PRIu64 " held_fed=%" PRIu64
                " wait_room=%" PRIu64 " wait_part=%" PRIu64 " wait_gate=%" PRIu64 " wait_install=%" PRIu64
                " wait_outside=%" PRIu64 " chains_decided=%" PRIu64 " chains_dealt=%" PRIu64 " chains_in=%" PRIu64
                " chain_wait_ext=%" PRIu64 " fills=%" PRIu64
                " post_us=%.1f submit_us=%.1f submit_ledger_us=%.1f store_peak=%" PRIu64 " map_peak=%" PRIu64
                " sent=%" PRIu64 " events_in=%" PRIu64 " matched=%" PRIu64 " early=%" PRIu64 " bumped=%" PRIu64
                " timed=%" PRIu64 " late=%" PRIu64 " late_us=%.1f late_arrive=%" PRIu64 " late_own=%" PRIu64
                " late_peer=%" PRIu64 " late_threshold=%" PRIu64 " late_map=%" PRIu64 " thr_waits=%" PRIu64
                " thr_wait_us=%.1f la_waits=%" PRIu64 " la_wait_us=%.1f"
                " skips=%" PRIu64 " sim_fills=%" PRIu64 " switches=%" PRIu64 " stream=%" PRIu64
                " range_conflicts=%" PRIu64 " hold_waits=%" PRIu64 " hold_wait_us=%.1f",
                thread_idx, pf.partitions, pf.pieces, pf.entries_posted, pf.map_entries, pf.slots_skipped,
                pf.slots_routed, pf.held_count, pf.held_fed, pf.wait_room, pf.wait_part, pf.wait_gate, pf.wait_install,
                pf.wait_outside, pf.chains_decided, pf.chains_dealt, pf.chains_in, pf.chain_wait_ext, pf.fills,
                cycles_to_us(pf.post_ticks), cycles_to_us(pf.submit_ticks), cycles_to_us(pf.submit_ledger_ticks),
                ps.resident_peak, ps.map_peak, ps.sent, ps.events_in, ps.matched, ps.early, ps.bumped, ps.timed,
                ps.late, static_cast<double>(ps.late_ns) / 1000.0, ps.late_by[0], ps.late_by[1], ps.late_by[2],
                ps.late_by[3], ps.late_by[4], ps.waits[0], static_cast<double>(ps.wait_ns[0]) / 1000.0, ps.waits[1],
                static_cast<double>(ps.wait_ns[1]) / 1000.0, ps.skips, ps.fills, ps.switches, ps.stream_len,
                ps.range_conflicts, pf.hold_waits, cycles_to_us(pf.hold_wait_ticks)
            );
        }
        if (silo_active_ && !part_active_) {
            const SiloFeed &sf = silo_feed_[thread_idx];
            uint64_t published = 0, resolved = 0, timed = 0, late = 0, late_ns = 0, adm_late = 0, adm_late_ns = 0;
            asimgq::queue_peer_stats(q, &published, &resolved, &timed, &late, &late_ns);
            asimgq::queue_admit_lateness(q, &adm_late, &adm_late_ns);
            LOG_INFO(
                "[GQ_SILO thread=%d] pieces=%" PRIu64 " entries=%" PRIu64 " named=%" PRIu64 " wait_unfed=%" PRIu64
                " wait_ordinary=%" PRIu64 " wait_comparators=%" PRIu64 " wait_room=%" PRIu64 " peer_resolved=%" PRIu64
                " peer_timed=%" PRIu64 " peer_late=%" PRIu64 " peer_late_us=%.1f cursor=%d producers_retired=%" PRIu64
                " pieces_behind=%" PRIu64 " admit_late=%" PRIu64 " admit_late_us=%.1f stall_unfed_us=%.1f"
                " stall_ordinary_us=%.1f stall_comparators_us=%.1f stall_room_us=%.1f",
                thread_idx, sf.pieces, sf.entries, sf.named, sf.wait_unfed, sf.wait_ordinary, sf.wait_comparators,
                sf.wait_room, resolved, timed, late, static_cast<double>(late_ns) / 1000.0, sf.cursor,
                sf.producers_retired, sf.pieces_behind, adm_late, static_cast<double>(adm_late_ns) / 1000.0,
                cycles_to_us(sf.stall_ticks[0]), cycles_to_us(sf.stall_ticks[1]), cycles_to_us(sf.stall_ticks[2]),
                cycles_to_us(sf.stall_ticks[3])
            );
        }
        if (bells_waiting > 0) {
            char bell_buf[768];
            asimgq::describe_waiting_bells(q, bell_buf, sizeof(bell_buf));
            LOG_INFO("[GQ_BELLS thread=%d] %s", thread_idx, bell_buf);
        }
#if ASIMGQ_SELF_PROFILE
        uint64_t retire_ns = 0, status_ns = 0, profiled = 0;
        asimgq::queue_poll_profile(q, &retire_ns, &status_ns, &profiled);
        LOG_INFO(
            "[ASIM_PROF] thread=%d calls=%" PRIu64 " retire_ns=%" PRIu64 " status_ns=%" PRIu64, thread_idx, profiled,
            retire_ns, status_ns
        );
#endif
        if (thread_idx == 0) {
            uint64_t h[8] = {};
            asimgq::func_hist(h, 8);
            LOG_INFO(
                "[M2_HIST] f0=%" PRIu64 " f1=%" PRIu64 " f2=%" PRIu64 " f3=%" PRIu64 " f4=%" PRIu64 " f5=%" PRIu64
                " OTHER=%" PRIu64 " maxid=%" PRIu64 " sum=%" PRIu64,
                h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[0] + h[1] + h[2] + h[3] + h[4] + h[5] + h[6]
            );
            // Completion, the share of durations the calibration supplied, and why
            // the grouping path passed tasks over, in the run's second line.
            uint64_t hits = 0, misses = 0;
            asimgq::calib_coverage(&hits, &misses);
            LOG_INFO(
                "[M2_DONE] completed_tasks=%d total=%d calibrated=%" PRIu64 " default=%" PRIu64 " skip_fired=%" PRIu64
                " skip_unclaimed=%" PRIu64 " skip_nopos=%" PRIu64 " deps_named=%" PRIu64 " deps_allcomplete=%" PRIu64
                " deps_bailed=%" PRIu64 " deps_toomany=%" PRIu64,
                completed_tasks_.load(std::memory_order_relaxed), total_tasks_, hits, misses,
                g_gq_skip_fired.load(std::memory_order_relaxed), g_gq_skip_unclaimed.load(std::memory_order_relaxed),
                g_gq_skip_nopos.load(std::memory_order_relaxed), g_gq_deps_named.load(std::memory_order_relaxed),
                g_gq_deps_allcomplete.load(std::memory_order_relaxed), g_gq_deps_bailed.load(std::memory_order_relaxed),
                g_gq_deps_toomany.load(std::memory_order_relaxed)
            );
        }
    }
#endif
    LOG_INFO("Thread %d: retiring %d cores", thread_idx, core_num);
    return retire_cores(runtime, cores, core_num);
}

int32_t SchedulerContext::retire_cores(Runtime *runtime, const int32_t *core_ids, int32_t core_num) {
    AicoreExitTarget targets[PLATFORM_MAX_CORES];
    int32_t claimed_ids[PLATFORM_MAX_CORES];
    size_t count = 0;
    for (int32_t i = 0; i < core_num; ++i) {
        const int32_t core_id = core_ids[i];
        if (core_id < 0 || core_id >= cores_total_num_) continue;
        if (core_exec_states_[core_id].reg_addr == 0) continue;
        // Claiming decides ownership of this core's register window and return
        // gate. The loser must not touch either again: writing to a window
        // whose worker was already released is the very ordering violation the
        // return gate exists to prevent.
        if (core_retired_[core_id].exchange(true, std::memory_order_acq_rel)) continue;
        claimed_ids[count] = core_id;
        targets[count] = {core_exec_states_[core_id].reg_addr, &runtime->get_teardown_gates()[core_id]};
        ++count;
    }
    if (count == 0) return 0;

    // platform_retire_aicore_group fills every entry on every path it returns
    // from, so this needs no initializer.
    bool released[PLATFORM_MAX_CORES];
    const int32_t rc = platform_retire_aicore_group(targets, count, platform_aicore_exit_deadline(), released);
    if (rc != 0) {
        // Naming the cores is the only signal an unreleased worker leaves: it
        // spins on a gate it cannot log about, and the host sees just a stream
        // timeout. See docs/troubleshooting/a2a3-worker-retirement.md.
        for (size_t i = 0; i < count; ++i) {
            if (!released[i]) {
                LOG_ERROR(
                    "AICore retirement: core %d not released (COND=0x%llx); worker stays blocked", claimed_ids[i],
                    static_cast<unsigned long long>(read_reg(targets[i].reg_addr, RegId::COND))
                );
            }
        }
    }
    return rc;
}

int32_t SchedulerContext::retire_all_cores(Runtime *runtime) {
    int32_t all[PLATFORM_MAX_CORES];
    int32_t n = 0;
    for (int32_t i = 0; i < cores_total_num_; ++i)
        all[n++] = i;
    return retire_cores(runtime, all, n);
}

// =============================================================================
// Handshake a contiguous slice of AICore workers. Runs on every AICPU thread in
// parallel (partitioned by tidx/nthreads); the leader's pre_handshake_init has
// already zeroed state, set cores_total_num_, and reset the counts/flag. The
// per-core work here — releasing the core, then opening its register window over
// serial MMIO — is what dominates preamble, so splitting the slice across
// threads is the whole point. Within a slice we still sweep (poll every
// outstanding core per pass, service whichever reported) so one slow core's
// wakeup overlaps its neighbours' instead of blocking them. Worker-id lists are
// built serially in post_handshake_init (core-index order) once every slice has
// landed, so the shared aic_count_/aiv_count_ are written by one thread only.
// =============================================================================
void SchedulerContext::handshake_partition(Runtime *runtime, int32_t tidx, int32_t nthreads) {
    Handshake *all_handshakes = reinterpret_cast<Handshake *>(runtime->workers);
    const int32_t total = cores_total_num_;
    const int32_t lo = static_cast<int32_t>((static_cast<int64_t>(tidx) * total) / nthreads);
    const int32_t hi = static_cast<int32_t>((static_cast<int64_t>(tidx + 1) * total) / nthreads);

    // The AICore publishes {physical_core_id, core_type, aicore_done} on launch,
    // gated by nothing. task is not published here: the AICore's aicore_done
    // report flushes its whole handshake cache line, so a task stored before the
    // report would be clobbered. task is written per core in the sweep below,
    // after that core's aicore_done is observed and before its window opens (the
    // point the AICore reads task).

    // Get platform physical cores count for validation
    uint32_t max_physical_cores_count = platform_get_physical_cores_count();

    // Step 2: collect responses from this slice. Each core reports
    // {physical_core_id, core_type, aicore_done} in one write, then waits — by
    // polling its own DATA_MAIN_BASE SPR — for us to open its register window.
    // We sweep the slice: poll every outstanding core per pass and service
    // whichever have reported, rather than blocking on core i before looking at
    // core i+1, so per-core wakeups overlap (≈ max, not Σ). aicore_done is a GM
    // read (not the nGnRE MMIO reg window), so sweeping is not forced serial the
    // way RegId::COND polling is.
    //
    // Servicing a core = validate its physical_core_id, then open its register
    // window (platform_init_aicore_regs: FAST_PATH + DATA_MAIN_BASE=IDLE). That
    // IDLE write is *also* the signal the core polls for to leave its
    // post-report wait — so opening the window IS the acknowledgement. There is
    // no separate aicpu_regs_ready ack and no second round-trip. AIC/AIV
    // classification is deferred to post_handshake_init (serial) so aic_count_/
    // aiv_count_ are never incremented from more than one thread.
    uint64_t *regs = reinterpret_cast<uint64_t *>(regs_);
    bool core_serviced[RUNTIME_MAX_WORKER] = {false};

    // Every core publishes aicore_done on launch, so the whole slice is already
    // reported when the AICPU sweeps it. The reported cores are collected first,
    // then serviced in batched phases (publish tasks, open windows, store
    // CoreExecStates); each phase issues its stores without interleaving another
    // phase's, so posted MMIO STRs and write-through GM stores do not serialize.
    struct ReadyCore {
        int32_t i;
        uint32_t pcid;
        uint64_t reg_addr;
        CoreType core_type;
    };
    ReadyCore ready[RUNTIME_MAX_WORKER];
    int32_t n_ready = 0;

    // Phase 1: collect every reported core in this slice and prefetch its
    // CoreExecState line for write, so the Phase 4 struct store hits a warm line.
    for (int32_t remaining = hi - lo; remaining > 0;) {
        for (int32_t i = lo; i < hi; i++) {
            if (core_serviced[i]) continue;
            Handshake *hank = &all_handshakes[i];
            if (hank->aicore_done == 0) {
                SPIN_WAIT_HINT();
                continue;
            }
            uint32_t physical_core_id = hank->physical_core_id;
            if (physical_core_id >= max_physical_cores_count) {
                LOG_ERROR(
                    "Core %d reported invalid physical_core_id=%u (platform max=%u)", i, physical_core_id,
                    max_physical_cores_count
                );
                handshake_failed_.store(true, std::memory_order_release);
                core_serviced[i] = true;
                remaining--;
                continue;
            }
            __builtin_prefetch(&core_exec_states_[i], 1, 3);
            ready[n_ready++] = {i, physical_core_id, regs[physical_core_id], hank->core_type};
            core_serviced[i] = true;
            remaining--;
        }
    }

    // Phase 2: publish every task pointer, then ONE barrier. The core reads task
    // only after its window opens (Phase 3); a single barrier orders all task
    // stores before any window STR. Writing task now (after the report) also
    // keeps the core's CACHELINE_OUT report flush from clobbering it.
    for (int32_t r = 0; r < n_ready; r++) {
        all_handshakes[ready[r].i].task = reinterpret_cast<uint64_t>(&payload_per_core_[ready[r].i][0]);
    }
    OUT_OF_ORDER_STORE_BARRIER();

    // Phase 3: open every window. platform_init_aicore_regs' STRs are posted
    // Device-nGnRE writes, issued back-to-back with no interleaved GM stores.
    for (int32_t r = 0; r < n_ready; r++) {
        platform_init_aicore_regs(ready[r].reg_addr);
    }

    // Phase 4: publish each CoreExecState with a single (prefetched) struct store.
    // core_exec_states_ is AICPU-private (the scheduler reads it, never the core),
    // so it may be written after the windows open.
    for (int32_t r = 0; r < n_ready; r++) {
        int32_t i = ready[r].i;
        CoreExecState st{};
        st.reg_addr = ready[r].reg_addr;
        st.cond_ptr = get_reg_ptr(ready[r].reg_addr, RegId::COND);
        st.running_reg_task_id = AICPU_TASK_INVALID;
        st.pending_reg_task_id = AICPU_TASK_INVALID;
#if !SIMPLER_DFX
        st.worker_id = i;
        st.physical_core_id = ready[r].pcid;
        st.core_type = ready[r].core_type;
#endif
        core_exec_states_[i] = st;
        core_type_compact_[i] = static_cast<uint8_t>(ready[r].core_type);
#if SIMPLER_DFX
        physical_core_ids_[i] = ready[r].pcid;
#endif
    }
    OUT_OF_ORDER_STORE_BARRIER();
}

// =============================================================================
// Assign discovered cores to scheduler threads (cluster-aligned round-robin).
// =============================================================================
bool SchedulerContext::assign_cores_to_threads() {
    // Cluster-aligned round-robin assignment: cluster ci -> sched thread ci % active_sched_threads_.
    // Each cluster = 1 AIC + 2 adjacent AIV; the triple is always kept together.
    //
    // Symmetric threads: every AICPU thread owns clusters, polls their COND
    // registers and resolves the completions it observes. Cores therefore
    // partition across all of them, and one thread is a valid configuration —
    // there is no core-less resolution thread to reserve.
    //
    // Polling a core's COND is a Device-nGnRE load that cannot pipeline against
    // another (~95 ns each, serialised by the nR attribute), so a thread's
    // FIN-detection round costs clusters_owned x lanes x ~95 ns. Spreading the
    // same clusters over more threads is the only way to shorten that round.
    if (aicpu_thread_num_ < 1) {
        LOG_ERROR("host_build_graph requires aicpu_thread_num >= 1; got %d", aicpu_thread_num_);
        return false;
    }
    active_sched_threads_ = aicpu_thread_num_;
    int32_t cluster_count = aic_count_;

    // Max clusters any single sched thread can hold: ceil(cluster_count / active_sched_threads_).
    int32_t max_clusters_per_thread = (cluster_count + active_sched_threads_ - 1) / active_sched_threads_;
    int32_t thread_cores_num = max_clusters_per_thread * 3;

    if (thread_cores_num > CoreTracker::MAX_CORE_PER_THREAD) {
        LOG_ERROR("Can't assign more then 64 cores in per scheduler");
        return false;
    }

    LOG_INFO(
        "Assigning cores (round-robin): %d clusters across %d sched threads (%d AIC, %d AIV)", cluster_count,
        active_sched_threads_, aic_count_, aiv_count_
    );

    // running_reg_task_id / pending_reg_task_id for every serviced core are reset
    // in handshake_partition's sweep.

    // Count clusters per thread first (round-robin may distribute unevenly)
    int32_t clusters_per_thread[MAX_AICPU_THREADS] = {};
    for (int32_t ci = 0; ci < cluster_count; ci++) {
        clusters_per_thread[ci % active_sched_threads_]++;
    }
    for (int32_t i = 0; i < active_sched_threads_; i++) {
        core_trackers_[i].init(clusters_per_thread[i]);
    }

    int32_t cluster_idx_per_thread[MAX_AICPU_THREADS] = {};

    for (int32_t ci = 0; ci < cluster_count; ci++) {
        int32_t t = ci % active_sched_threads_;

        int32_t aic_wid = aic_worker_ids_[ci];
        int32_t aiv0_wid = aiv_worker_ids_[2 * ci];
        int32_t aiv1_wid = aiv_worker_ids_[2 * ci + 1];

        core_trackers_[t].set_cluster(cluster_idx_per_thread[t]++, aic_wid, aiv0_wid, aiv1_wid);
#ifdef __SIMULATED_DEVICE__
        // The controller places work itself, so it needs the same grouping: which
        // three cores share a package, and which manager's queue serves them. Only
        // the handshake knows the real worker ids, so this is registered here
        // rather than guessed at bring-up.
        asimgq::set_package(
            static_cast<uint32_t>(ci), static_cast<uint32_t>(t), static_cast<uint32_t>(aic_wid),
            static_cast<uint32_t>(aiv0_wid), static_cast<uint32_t>(aiv1_wid)
        );
#endif

        LOG_DEBUG("Thread %d: cluster %d (AIC=%d, AIV0=%d, AIV1=%d)", t, ci, aic_wid, aiv0_wid, aiv1_wid);
    }

    for (int32_t t = 0; t < aicpu_thread_num_; t++) {
        LOG_DEBUG(
            "Thread %d: total %d cores (%d clusters)", t, core_trackers_[t].core_num(),
            core_trackers_[t].get_cluster_count()
        );
    }

    LOG_INFO(
        "Config: threads=%d, cores=%d, cores_per_thread=%d", aicpu_thread_num_, cores_total_num_, thread_cores_num
    );
    return true;
}

// =============================================================================
// Emergency shutdown: broadcast exit signal to every handshake'd core and
// deinit their AICore register blocks. Idempotent.
// =============================================================================
void SchedulerContext::emergency_shutdown(Runtime *runtime) {
    // Sweeps every core rather than one thread's slice: a fatal run must not
    // depend on the owning threads reaching their own shutdown. Per-core
    // claiming keeps whatever they already retired untouched. Cores whose
    // register windows never opened remain the host recovery path's
    // responsibility.
    LOG_WARN("Emergency shutdown: retiring all initialized AICores");
    (void)retire_all_cores(runtime);
}

// =============================================================================
// Lifecycle: init / deinit
// =============================================================================
int32_t SchedulerContext::pre_handshake_init(Runtime *runtime, int32_t aicpu_thread_num, uint64_t regs_base) {
    always_assert(runtime != nullptr);

    // Zero all per-core execution state before handshake
    memset(core_exec_states_, 0, sizeof(core_exec_states_));
    for (int32_t i = 0; i < PLATFORM_MAX_CORES; ++i) {
        core_retired_[i].store(false, std::memory_order_relaxed);
    }

    // Wire thread configuration that handshake/assign need to read.
    aicpu_thread_num_ = aicpu_thread_num;
    regs_ = regs_base;

#if SIMPLER_DFX
    // chip_swimlane_aicpu_init promotes g_chip_swimlane_level from the shared-memory
    // header — must be called BEFORE caching the level, otherwise the cached
    // value would still be 0 (only the binary enable bit has been seeded by
    // kernel.cpp at this point). Reset the cached level on disabled runs so a
    // prior enabled launch's level can't leak into the phase-record gates in
    // scheduler_dispatch. This runs on the leader before it publishes
    // hs_setup_done_, so it happens-before every thread's handshake_partition
    // and therefore before any register window is opened.
    if (is_chip_swimlane_enabled()) {
        chip_swimlane_aicpu_init(runtime->worker_count);
        chip_swimlane_level_ = get_chip_swimlane_level();
        if (chip_swimlane_level_ >= ChipSwimlaneLevel::SCHED_PHASES) {
            // Sched-phase pool count must match the dump_args_init thread count
            // below. This block runs before assign_cores_to_threads, so the
            // active_sched_threads_ member isn't set yet; every AICPU thread is a
            // scheduler, so the count is aicpu_thread_num_. Without it, init_phase
            // would prime zero sched pools and all sched_phase emits would silently
            // drop.
            const int sched_phase_threads = aicpu_thread_num_;
            // HBG orchestration already completed on the host. Its level-4
            // records use the host callback path, so the device initializes no
            // dead AICPU orchestrator pool.
            const int orch_phase_threads = 0;
            chip_swimlane_aicpu_init_phase(runtime->worker_count, sched_phase_threads, orch_phase_threads);
        }
    } else {
        chip_swimlane_level_ = ChipSwimlaneLevel::DISABLED;
    }
#endif

    // Core count is needed by every thread to compute its handshake slice.
    cores_total_num_ = runtime->worker_count;
    if (cores_total_num_ == 0 || cores_total_num_ > RUNTIME_MAX_WORKER) {
        LOG_ERROR("Invalid cores_total_num %d (expected 1-%d)", cores_total_num_, RUNTIME_MAX_WORKER);
        return -1;
    }
    // The prior launch may have left RELEASE=1. The wmb() is what orders these
    // resets before hs_setup_done_ and before any register window opens: a
    // window is a plain Device-nGnRE store, carrying no release semantics of
    // its own.
    memset(runtime->get_teardown_gates(), 0, sizeof(AicoreTeardownControl) * cores_total_num_);
    wmb();
    aic_count_ = 0;
    aiv_count_ = 0;
    handshake_failed_.store(false, std::memory_order_release);

    LOG_INFO("Handshaking with %d cores", cores_total_num_);
    return 0;
}

int32_t SchedulerContext::post_handshake_init(Runtime *runtime) {
    if (handshake_failed_.load(std::memory_order_acquire)) {
        emergency_shutdown(runtime);
        return -1;
    }

    // Build the AIC/AIV worker-id lists in core-index order, which
    // assign_cores_to_threads pairs into clusters. core_type is read from the
    // contiguously packed core_type_compact_ the sweep filled, not the 64B-aligned
    // per-core Handshake struct. aic_worker_ids_/aiv_worker_ids_ store through to
    // HBM, so the lists are built in local (cached) buffers and published with two
    // wide memcpys rather than element by element.
    int32_t local_aic[RUNTIME_MAX_WORKER];
    int32_t local_aiv[RUNTIME_MAX_WORKER];
    int32_t la = 0, lv = 0;
    for (int32_t i = 0; i < cores_total_num_; i++) {
        if (static_cast<CoreType>(core_type_compact_[i]) == CoreType::AIC) {
            local_aic[la++] = i;
        } else {
            local_aiv[lv++] = i;
        }
    }
    memcpy(aic_worker_ids_, local_aic, static_cast<size_t>(la) * sizeof(int32_t));
    memcpy(aiv_worker_ids_, local_aiv, static_cast<size_t>(lv) * sizeof(int32_t));
    aic_count_ = la;
    aiv_count_ = lv;
    LOG_INFO("Core discovery complete: %d AIC, %d AIV", aic_count_, aiv_count_);

    if (!assign_cores_to_threads()) {
        return -1;
    }

    // Profiling-subsystem buffer/state init: single-threaded cold path (leader
    // only), so the "do it once" guarantee is structural (no CAS needed). Runs
    // after the handshake / assign_cores_to_threads because pmu_aicpu_init needs
    // physical_core_ids_ / cores_total_num_. Mirrors the chip_swimlane_aicpu_init
    // convention above.
#if SIMPLER_DFX
    if (is_dump_args_enabled()) {
        dump_args_init(active_sched_threads_);
    }
    if (is_pmu_enabled()) {
        pmu_aicpu_init(physical_core_ids_, cores_total_num_);
    }
#endif

    // Initialize task counters. Task count comes from shared memory.
    // 0 is the correct count at boot: the graph is not attached yet, and
    // on_graph_attached latches the host-built total before releasing any
    // scheduler thread.
    total_tasks_ = 0;
    completed_tasks_.store(0, std::memory_order_release);

    // prepare_subtask_to_core fully writes a per-core payload / deferred-slab slot
    // before the AICore is told to read it: build_payload sets
    // function_bin_addr/args/local_context/not_ready, and deferred_slab->count/
    // error_code are reset inline on every dispatch. An AICore reads a slot only
    // after a dispatch targets it (DATA_MAIN_BASE), so a prior round's bytes in an
    // untouched slot are never observed.

    // Initialize per-core GlobalContext (sub_block_id) based on cluster position.
    // This is done once at startup and never modified afterwards.
    for (int32_t t = 0; t < active_sched_threads_; t++) {
        CoreTracker &tracker = core_trackers_[t];
        for (int32_t c = 0; c < tracker.get_cluster_count(); c++) {
            int32_t cluster_offset = c * 3;  // Each cluster = 1 AIC + 2 AIV
            auto aiv0_id = tracker.get_core_id_by_offset(tracker.get_aiv0_core_offset(cluster_offset));
            auto aiv1_id = tracker.get_core_id_by_offset(tracker.get_aiv1_core_offset(cluster_offset));
            payload_per_core_[aiv0_id][0].global_context.sub_block_id = 0;
            payload_per_core_[aiv0_id][1].global_context.sub_block_id = 0;
            payload_per_core_[aiv1_id][0].global_context.sub_block_id = 1;
            payload_per_core_[aiv1_id][1].global_context.sub_block_id = 1;
        }
    }

    // Prefill the per-dispatch AsyncCtx constant fields once. Of AsyncCtx's five
    // fields, four are constant for a given (core, buf_idx): the three pointers
    // target the fixed deferred_slab_per_core_[core][buf] members, and capacity is
    // MAX_COMPLETIONS_PER_TASK. Only task_token varies per dispatch, so build_payload
    // writes just that; these constants survive across dispatches because the
    // payload buffer is never zeroed between them.
    // The two context-pointer args are also per-(core, buf_idx) constants — they
    // target this buffer's own local_context / global_context — so prefill them
    // here too and drop them from the per-dispatch build_payload writes. This keeps
    // the per-dispatch write footprint on the CL0 control block only (args[48]/[49]
    // live on a later line).
    for (int32_t core_id = 0; core_id < RUNTIME_MAX_WORKER; core_id++) {
        for (int32_t buf = 0; buf < 2; buf++) {
            DispatchPayload &dp = payload_per_core_[core_id][buf];
            AsyncCtx &ac = dp.local_context.async_ctx;
            volatile DeferredCompletionSlab *slab = &deferred_slab_per_core_[core_id][buf];
            ac.completion_count = &slab->count;
            ac.completion_error_code = &slab->error_code;
            ac.completion_entries = &slab->entries[0];
            ac.completion_capacity = MAX_COMPLETIONS_PER_TASK;
            // Clear the slab once here; thereafter only the completion path re-clears
            // count (and only when a deferred task dirtied it), never per dispatch.
            slab->count = 0;
            slab->error_code = SIMPLER_ERROR_NONE;
            dp.args[PAYLOAD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&dp.local_context);
            dp.args[PAYLOAD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&dp.global_context);
        }
    }

    func_id_to_addr_ = runtime->func_id_to_addr_;

    return 0;
}

void SchedulerContext::deinit() {
    // Close only the core lanes this run actually used. channel_names is sized to
    // worker_count (== cores_total_num_ here), so a reset on a higher lane emits a
    // trace event on a channelId the metadata never names.
    for (int32_t i = 0; i < cores_total_num_; i++) {
#if !(defined(__SIMULATED_DEVICE__) && ASIMGQ_TRACR_CORE_LANES)
        // Suppressed when the GroupQueue model writes these lanes itself: a reset
        // here closes whichever span the model has open on that lane, clipping it
        // to the instant the manager observed the completion rather than the
        // modelled end of the kernel body.
        INSTRUMENTATION_MARK_RESET(aicpu_thread_num_ + i);
#endif
    }

    // Reset all per-core execution state
    for (int32_t i = 0; i < RUNTIME_MAX_WORKER; i++) {
        core_exec_states_[i] = {};
        core_exec_states_[i].running_reg_task_id = AICPU_TASK_INVALID;
        core_exec_states_[i].pending_reg_task_id = AICPU_TASK_INVALID;
    }

    // No per-core memset of payload_per_core_ / deferred_slab_per_core_ here
    // (~300 KB across all cores). They are re-initialized before they can be read:
    // build_payload() overwrites the per-dispatch payload fields (function addr,
    // args[0..num_args) or src_payload, block_idx/block_num, async_ctx.task_token)
    // on the exact [core][buf_idx] about to run; the async_ctx slab pointers +
    // capacity, the two context-pointer args, and the deferred slab (count = 0 /
    // error_code = NONE) are all cleared once per run in init() — the slab is
    // thereafter re-cleared only by the completion path after a deferred task.
    // The consumer side cannot reach a stale slot either: the
    // drain only services a core's running_reg_task_id, and the loop above
    // already reset every core_exec_states_[].running/pending_reg_task_id to
    // AICPU_TASK_INVALID — so no FIN for an undispatched slot is processed, and
    // the count-gated consumer never reads entries[] past the fresh count.

    // A run that aborted mid-classify leaves a share outstanding, which would hold
    // the next run's threads in their dispatch loops.
    for (int32_t i = 0; i < MAX_AICPU_THREADS; i++) {
        classify_cursor_[i] = 0;
        classify_end_[i] = 0;
    }
    classify_pending_.store(0, std::memory_order_release);

    // Reset sync-start drain coordination — a previous run that aborted mid-drain
    // would otherwise leave dirty pending/ack state for the next reuse.
    drain_state_.sync_start_pending.store(0, std::memory_order_release);
    drain_state_.drain_attempt.store(0, std::memory_order_release);
    for (int32_t t = 0; t < MAX_AICPU_THREADS; t++) {
        drain_ack_tokens_[t].store(0, std::memory_order_release);
    }
    drain_state_.drain_stage_go.store(0, std::memory_order_release);
    drain_state_.drain_stage_done_mask.store(0, std::memory_order_release);
    drain_state_.drain_running_staged.store(0, std::memory_order_release);
    drain_state_.pending_task.store(nullptr, std::memory_order_release);

    // Reset task counters and orchestrator state
    completed_tasks_.store(0, std::memory_order_release);
    total_tasks_ = 0;
    completed_.store(false, std::memory_order_release);

    // Reset core discovery and assignment state
    aic_count_ = 0;
    aiv_count_ = 0;
    cores_total_num_ = 0;
    aicpu_thread_num_ = 0;
    active_sched_threads_ = 0;
    for (int32_t t = 0; t < MAX_AICPU_THREADS; t++) {
        core_trackers_[t] = CoreTracker{};
    }

    regs_ = 0;
    sched_ = nullptr;
    rt_ = nullptr;
    func_id_to_addr_ = nullptr;
}

void SchedulerContext::bind_runtime(RuntimeContext *rt) {
    rt_ = rt;
    sched_ = rt->scheduler;
}

// =============================================================================
// Post-attach bookkeeping. Runs once on the boot leader after the host-built
// image is attached; latches total_tasks_, sizes the per-S queues to it, and
// folds inline-completed tasks. classify_ready_ is released after this call and
// is what publishes total_tasks_ to the peer threads, which acquire it before
// classify_partition reads the count.
// =============================================================================
void SchedulerContext::on_graph_attached(RuntimeContext *rt, [[maybe_unused]] int32_t thread_idx, int32_t total_tasks) {
    total_tasks_ = total_tasks;

    // No S->P completion transport: the thread that observes a FIN resolves it,
    // so there is nothing to hand off and nothing to size here.

    // Fold tasks completed inline during orchestration
    int32_t inline_completed = static_cast<int32_t>(rt->inline_completed_tasks);
    if (inline_completed > 0) {
        completed_tasks_.fetch_add(inline_completed, std::memory_order_relaxed);
#if SIMPLER_SCHED_PROFILING
        rt->scheduler->tasks_completed.fetch_add(inline_completed, std::memory_order_relaxed);
#endif
    }

    // The polling initial classify (seed the ready queues + wake lists for the
    // whole graph) runs AFTER this: classify_begin() splits the submitted range
    // across all AICPU threads and each seeds its slice from its dispatch loop, so
    // the O(total_tasks) scan is neither serial on this thread nor ahead of the
    // cores.

#if SIMPLER_DFX
    // Write the core-to-thread mapping so the profiling data reflects the
    // scheduler threads' final core distribution.
    if (chip_swimlane_level_ >= ChipSwimlaneLevel::SCHED_PHASES) {
        chip_swimlane_aicpu_init_core_assignments(cores_total_num_);
        for (int32_t t = 0; t < active_sched_threads_; t++) {
            chip_swimlane_aicpu_write_core_assignments_for_thread(
                t, core_trackers_[t].core_ids(), core_trackers_[t].core_num()
            );
        }
    }
#endif
}

// Disjoint contiguous slices covering [0, submitted): thread t owns
// [submitted*t/nthreads, submitted*(t+1)/nthreads). int64 math avoids overflow.
void SchedulerContext::classify_begin(int32_t thread_idx, int32_t nthreads) {
    const int32_t submitted = total_tasks_;
    const int32_t lo = static_cast<int32_t>((static_cast<int64_t>(submitted) * thread_idx) / nthreads);
    const int32_t hi = static_cast<int32_t>((static_cast<int64_t>(submitted) * (thread_idx + 1)) / nthreads);
    classify_cursor_[thread_idx] = lo;
    classify_end_[thread_idx] = hi;
    if (hi > lo) classify_pending_.fetch_add(1, std::memory_order_acq_rel);
}

// Polling initial classify (device boot), partitioned across all AICPU threads
// and run a chunk at a time from each thread's dispatch loop. Each thread
// classifies its contiguous slice of the submitted-task range exactly once.
// Graph tasks additionally enter the bounded preparation queue; their external
// fanin follows the same ready/wake classification as any other outer task.
bool SchedulerContext::classify_chunk(int32_t thread_idx) {
    int32_t id = classify_cursor_[thread_idx];
    const int32_t first = id;
    const int32_t hi = classify_end_[thread_idx];
    if (id >= hi) return false;
    if (completed_.load(std::memory_order_acquire) || sched_->task_view.tasks == nullptr) {
        classify_cursor_[thread_idx] = hi;
        classify_pending_.fetch_sub(1, std::memory_order_acq_rel);
        return false;
    }
    SharedMemoryTaskHeader &tasks = *sched_->task_view.tasks;
    const int32_t chunk_end = (hi - id > kClassifyChunk) ? id + kClassifyChunk : hi;
    for (; id < chunk_end; id++) {
        if (tasks.is_completed(id)) {
            continue;  // completed on the host (hidden alloc); nothing to dispatch
        }
        ChipTaskSlotState &slot = tasks.get_slot_state_by_task_id(id);
        if (slot.task_kind == TaskKind::GRAPH) {
            if (graph_execution_localize(slot) == nullptr) slot.graph_context = nullptr;
            // A full preparation queue is a wait, not an end: the cursor stays on
            // this task so the next pass offers it again once the queue drains.
            if (!sched_->push_graph_prepare(&slot, slot.to_descriptor().task_id.raw, thread_idx)) break;
        }
        // The ready group queue delivers a group's dispatchable tasks, whole group
        // at a time, and only a task some group actually holds may rely on it. A
        // task the graph left undeclared is its own group: nothing will ever queue
        // it wholesale, so skipping the per-task path here would strand it and
        // every consumer that names it. The same is true of a task with no logical
        // block, which never reaches a core.
        if (gq_group::active() && gq_group::group_of(id) != gq_group::NO_GROUP && group_queue_delivers(slot)) {
            continue;
        }
        // A silo feeds this task whatever its producers' state -- in task order, or
        // with its partition -- so it must not also reach a ready queue.
        if (part_active_ ? part_feeds(slot) : (silo_active_ && silo_feeds(slot))) {
            continue;
        }
        int32_t state = sched_->classify_fanin_state(&slot);
        if (state < 0) {
            sched_->push_ready_routed(&slot);
        } else {
            int32_t prod_local = slot.to_payload().fanin_data()[state];
            sched_->register_wake(&tasks.get_slot_state_by_task_id(prod_local), &slot);
            // A not-yet-ready candidate also enters the publish list, hung on
            // its latest-submitted unpublished producer — or, when every
            // producer is already published at intake (pre-completed fanin),
            // goes straight to the ED queue. An already-ready task never does:
            // there is nothing left to pre-stage ahead of.
            if ((slot.ed_flags & ED_FLAG_CANDIDATE) != 0) {
                if (sched_->register_on_ed_publish_list(slot)) {
                    sched_->enqueue_early_dispatch_candidate(slot);
                }
            }
        }
    }
    classify_cursor_[thread_idx] = id;
    if (id >= hi) classify_pending_.fetch_sub(1, std::memory_order_acq_rel);
    return id > first;
}
