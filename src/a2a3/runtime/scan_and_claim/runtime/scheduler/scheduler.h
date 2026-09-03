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

/**
 * PTO Runtime2 - Scheduler Interface
 *
 * The Scheduler is responsible for:
 * 1. Maintaining per-resource-shape ready queues
 * 2. Polling-completion dependency resolution: a task is ready when every
 *    producer named in its inline fanin has set its completion_flags byte;
 *    a producer publishes completion + drains its wake list on finish
 * 3. Publishing the host-visible task_state mirror (PENDING -> COMPLETED) and
 *    advancing the per-ring completed_watermark (consumer-retirement signal)
 * 4. Two-stage mixed-task completion (subtask done bits -> mixed-task complete)
 *
 * The Scheduler runs on Device AI_CPU. host_build_graph is scheduler-only (the
 * orchestrator runs to completion on the host) and whole-graph-resident, so no
 * task slot or heap byte is reclaimed on device; the scheduler owns completion
 * state only.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include <atomic>

#include "common/core_type.h"
#include "common/platform_config.h"  // PLATFORM_MAX_AICPU_THREADS for the deferred-ready FIFOs
#include "common/memory_barrier.h"
#include "utils/device_arena.h"
#include "aicpu/platform_regs.h"  // get_reg_ptr / RegId for the early-dispatch doorbell
#include "async_wait.h"
#include "graph_execution.h"
#include "ring_buffer.h"
#include "runtime_types.h"
#include "shared_memory.h"

#include "aicpu/device_time.h"  // get_sys_cnt_aicpu (weak; used by early-dispatch doorbell timing too)
#if SIMPLER_SCHED_PROFILING
#define PTO2_SCHED_CYCLE_START() uint64_t _st0 = get_sys_cnt_aicpu(), _st1
#define PTO2_SCHED_CYCLE_LAP(acc)   \
    do {                            \
        _st1 = get_sys_cnt_aicpu(); \
        acc += (_st1 - _st0);       \
        _st0 = _st1;                \
    } while (0)
#endif

// =============================================================================
// Ready Queue (Lock-free bounded MPMC — Vyukov design)
// =============================================================================

/**
 * Per-slot entry: sequence counter for ABA safety + task payload
 */
struct PTO2ReadyQueueSlot {
    std::atomic<int64_t> sequence;
    ChipTaskSlotState *slot_state;
    uint64_t task_id_snapshot;
};

/**
 * Lock-free bounded MPMC queue (Dmitry Vyukov design)
 *
 * Key properties:
 * - enqueue_pos and dequeue_pos on separate cache lines (no false sharing)
 * - Per-slot sequence counter prevents ABA problem
 * - Empty queue pop returns immediately (single atomic load, no lock)
 * - CAS contention is split: producers only touch enqueue_pos,
 *   consumers only touch dequeue_pos
 */
struct alignas(64) PTO2ReadyQueue {
    PTO2ReadyQueueSlot *slots;
    uint64_t capacity;
    uint64_t mask;        // capacity - 1
    char _pad0[64 - 24];  // Pad to own cache line

    std::atomic<uint64_t> enqueue_pos;
    char _pad1[64 - sizeof(std::atomic<uint64_t>)];  // Own cache line

    std::atomic<uint64_t> dequeue_pos;
    // Occupancy high-water, for teardown reporting only. Atomic because pushes
    // are concurrent; relaxed throughout, so it is ordered against nothing.
    std::atomic<uint64_t> max_occupancy;
    char _pad2[64 - 2 * sizeof(std::atomic<uint64_t>)];  // Own cache line

    // Bring the slots[] array to its empty state: slot i's sequence must equal i
    // for push_tagged's `diff == 0` claim test to accept the first pass, so an
    // empty queue is a 0..capacity-1 ramp rather than zeroed memory. Runs on the
    // device after wire_arena_pointers, before any push — the region is reserved
    // past the uploaded range, so nothing seeds it on the host.
    void seed_slots() {
        for (uint64_t i = 0; i < capacity; i++) {
            slots[i].sequence.store(static_cast<int64_t>(i), std::memory_order_relaxed);
            slots[i].slot_state = nullptr;
        }
    }

    uint64_t size() {
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        return (e >= d) ? (e - d) : 0;
    }

    // Raise the high-water to the occupancy left by a push that published up to
    // `published_pos`. Off the fast path it is a load and a compare; the CAS runs
    // only on a new maximum, so a contended push never pays for it.
    void note_occupancy(uint64_t published_pos) {
        const uint64_t occ = published_pos - dequeue_pos.load(std::memory_order_relaxed);
        uint64_t observed = max_occupancy.load(std::memory_order_relaxed);
        while (occ > observed && !max_occupancy.compare_exchange_weak(
                                     observed, occ, std::memory_order_relaxed, std::memory_order_relaxed
                                 )) {}
    }

    bool push(ChipTaskSlotState *slot_state) { return push_tagged(slot_state, 0); }

    bool push_tagged(ChipTaskSlotState *slot_state, uint64_t task_id_snapshot) {
        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
        while (true) {
            pos = enqueue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - static_cast<int64_t>(pos);
            if (diff == 0) {
                if (enqueue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed
                    )) {
                    break;
                }
            } else if (diff < 0) {
                return false;  // Queue full
            }
        }

        slot->slot_state = slot_state;
        slot->task_id_snapshot = task_id_snapshot;
        slot->sequence.store(static_cast<int64_t>(pos + 1), std::memory_order_release);
        note_occupancy(pos + 1);
        return true;
    }

    // Batch push: reserve count slots with a single CAS after confirming
    // every target slot is available under the usual Vyukov sequence check.
    // Returns false without publishing anything when the queue cannot take all
    // `count` items. A target slot holding an older generation means full and
    // ends the call; a slot a peer has reserved but not yet published is
    // transient and retries, so this only spins while a peer is mid-publish.
    bool push_batch(ChipTaskSlotState **items, int count) { return push_batch_tagged(items, nullptr, count); }

    bool push_batch_tagged(ChipTaskSlotState **items, const uint64_t *task_id_snapshots, int count) {
        if (count == 0) return true;
        if (static_cast<uint64_t>(count) > capacity) return false;

        uint64_t pos;
        while (true) {
            pos = enqueue_pos.load(std::memory_order_relaxed);
            bool ready = true;
            for (int i = 0; i < count; i++) {
                PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
                int64_t seq = slot->sequence.load(std::memory_order_acquire);
                int64_t diff = seq - static_cast<int64_t>(pos + i);
                if (diff < 0) {
                    return false;  // Queue full
                }
                if (diff > 0) {
                    ready = false;
                    break;
                }
            }
            if (!ready) {
                continue;
            }
            if (enqueue_pos.compare_exchange_weak(
                    pos, pos + count, std::memory_order_relaxed, std::memory_order_relaxed
                )) {
                break;
            }
        }

        for (int i = 0; i < count; i++) {
            PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
            slot->slot_state = items[i];
            slot->task_id_snapshot = task_id_snapshots == nullptr ? 0 : task_id_snapshots[i];
            slot->sequence.store(static_cast<int64_t>(pos + i + 1), std::memory_order_release);
        }
        note_occupancy(pos + static_cast<uint64_t>(count));
        return true;
    }

#if SIMPLER_ORCH_PROFILING || SIMPLER_SCHED_PROFILING
    bool push(ChipTaskSlotState *slot_state, uint64_t &atomic_count, uint64_t &wait_cycle) {
        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
        uint64_t t0 = get_sys_cnt_aicpu();
        bool contended = false;
        uint32_t atomic_ops = 0;
        while (true) {
            pos = enqueue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - static_cast<int64_t>(pos);
            atomic_ops += 2;  // enqueue_pos.load + sequence.load
            if (diff == 0) {
                if (enqueue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed
                    )) {
                    atomic_ops++;  // successful CAS
                    break;
                }
                contended = true;
                atomic_ops++;  // failed CAS
            } else if (diff < 0) {
                return false;  // Queue full
            } else {
                contended = true;  // diff > 0: slot not yet released, spin
            }
        }
        atomic_ops++;  // final sequence.store
        atomic_count += atomic_ops;
        if (contended) {
            wait_cycle += (get_sys_cnt_aicpu() - t0);
        }

        slot->slot_state = slot_state;
        slot->sequence.store(static_cast<int64_t>(pos + 1), std::memory_order_release);
        return true;
    }
#endif

    ChipTaskSlotState *pop() { return pop_tagged(nullptr); }

    ChipTaskSlotState *pop_tagged(uint64_t *task_id_snapshot) {
        // Fast-path: skip slot load when queue is clearly empty
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        if (d >= e) {
            return nullptr;
        }

        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - static_cast<int64_t>(pos + 1);
            if (diff == 0) {
                if (dequeue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed
                    ))
                    break;
            } else if (diff < 0) {
                return nullptr;  // Queue empty
            }
        }

        ChipTaskSlotState *result = slot->slot_state;
        if (task_id_snapshot != nullptr) *task_id_snapshot = slot->task_id_snapshot;
        slot->sequence.store(static_cast<int64_t>(pos + mask + 1), std::memory_order_release);
        return result;
    }

#if SIMPLER_SCHED_PROFILING
    ChipTaskSlotState *pop(uint64_t &atomic_count, uint64_t &wait_cycle) {
        // Fast-path: skip slot load when queue is clearly empty
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        atomic_count += 2;  // dequeue_pos.load + enqueue_pos.load
        if (d >= e) {
            return nullptr;
        }

        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
        uint64_t t0 = get_sys_cnt_aicpu();
        bool contended = false;
        uint32_t atomic_ops = 0;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - static_cast<int64_t>(pos + 1);
            atomic_ops += 2;  // dequeue_pos.load + sequence.load
            if (diff == 0) {
                if (dequeue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed
                    )) {
                    atomic_ops++;  // successful CAS
                    break;
                }
                contended = true;
                atomic_ops++;  // failed CAS
            } else if (diff < 0) {
                atomic_count += atomic_ops;
                return nullptr;  // Queue empty
            } else {
                contended = true;
            }
        }
        atomic_ops++;  // final sequence.store
        atomic_count += atomic_ops;
        if (contended) {
            wait_cycle += (get_sys_cnt_aicpu() - t0);
        }

        ChipTaskSlotState *result = slot->slot_state;
        slot->sequence.store(static_cast<int64_t>(pos + mask + 1), std::memory_order_release);
        return result;
    }
#endif

    // Batch pop: reserve a contiguous run of ready slots with a single CAS.
    // Returns actual number of items popped (may be less than max_count).
    int pop_batch(ChipTaskSlotState **out, int max_count) { return pop_batch_tagged(out, nullptr, max_count); }

    int pop_batch_tagged(ChipTaskSlotState **out, uint64_t *task_id_snapshots, int max_count) {
        uint64_t pos;
        int count;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            count = 0;
            while (count < max_count) {
                PTO2ReadyQueueSlot *slot = &slots[(pos + count) & mask];
                int64_t seq = slot->sequence.load(std::memory_order_acquire);
                int64_t diff = seq - static_cast<int64_t>(pos + count + 1);
                if (diff == 0) {
                    count++;
                    continue;
                }
                if (diff < 0) {
                    break;
                }
                count = -1;
                break;
            }
            if (count == 0) return 0;
            if (count < 0) continue;
            if (dequeue_pos.compare_exchange_weak(
                    pos, pos + count, std::memory_order_relaxed, std::memory_order_relaxed
                )) {
                break;
            }
        }

        for (int i = 0; i < count; i++) {
            PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
            out[i] = slot->slot_state;
            if (task_id_snapshots != nullptr) task_id_snapshots[i] = slot->task_id_snapshot;
            slot->sequence.store(static_cast<int64_t>(pos + i + mask + 1), std::memory_order_release);
        }
        return count;
    }

#if SIMPLER_SCHED_PROFILING
    int pop_batch(ChipTaskSlotState **out, int max_count, uint64_t &atomic_count, uint64_t &wait_cycle) {
        uint64_t pos;
        int count;
        uint64_t t0 = get_sys_cnt_aicpu();
        bool contended = false;
        uint32_t atomic_ops = 0;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            atomic_ops++;  // dequeue_pos.load
            count = 0;
            while (count < max_count) {
                PTO2ReadyQueueSlot *slot = &slots[(pos + count) & mask];
                int64_t seq = slot->sequence.load(std::memory_order_acquire);
                int64_t diff = seq - static_cast<int64_t>(pos + count + 1);
                atomic_ops++;  // sequence.load
                if (diff == 0) {
                    count++;
                    continue;
                }
                if (diff < 0) {
                    break;
                }
                contended = true;
                count = -1;
                break;
            }
            if (count == 0) {
                atomic_count += atomic_ops;
                return 0;
            }
            if (count < 0) {
                continue;
            }
            if (dequeue_pos.compare_exchange_weak(
                    pos, pos + count, std::memory_order_relaxed, std::memory_order_relaxed
                )) {
                atomic_ops++;  // successful CAS
                break;
            }
            contended = true;
            atomic_ops++;  // failed CAS
        }

        for (int i = 0; i < count; i++) {
            PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
            out[i] = slot->slot_state;
            slot->sequence.store(static_cast<int64_t>(pos + i + mask + 1), std::memory_order_release);
            atomic_ops++;  // sequence.store
        }
        atomic_count += atomic_ops;
        if (contended) {
            wait_cycle += (get_sys_cnt_aicpu() - t0);
        }
        return count;
    }
#endif
};

// Cold-path ready queue operations (defined in scheduler.cpp). Declared
// as non-member so PTO2ReadyQueue stays a POD-like struct with cache-line
// alignment. Storage is owned by the caller-supplied arena.
//   reserve_layout: declare the slots[] region on the arena (must precede commit)
//   init_from_layout: initialize the queue header (capacity, mask, positions)
//   seed_slots: establish the slot array's empty ramp, on the device only
//   destroy: forget the slots pointer (arena owns the buffer)
size_t ready_queue_reserve_layout(DeviceArena &arena, uint64_t capacity);
// Writes the header fields only: capacity, mask, positions, occupancy counter.
// The slot array is neither addressed nor written — it lives past the uploaded
// range and PTO2ReadyQueue::seed_slots() fills it on the device. Call
// `ready_queue_wire_arena_pointers` to set the `slots` pointer itself.
void ready_queue_init_data_from_layout(PTO2ReadyQueue *queue, uint64_t capacity);
// Stores queue->slots = arena.region_ptr(slots_off). Idempotent.
void ready_queue_wire_arena_pointers(PTO2ReadyQueue *queue, DeviceArena &arena, size_t slots_off);
void ready_queue_destroy(PTO2ReadyQueue *queue);

/**
 * Statistics returned by mixed-task completion processing
 */
struct CompletionStats {
    int32_t fanout_edges;       // Number of fanout edges traversed (notify consumers)
    int32_t tasks_enqueued;     // Number of consumers that became READY
    int32_t fanin_edges;        // Number of fanin edges traversed (release producers)
    bool mixed_task_completed;  // True only when this callback completed a mixed task
};

/**
 * Layout descriptor produced by PTO2SchedulerState::reserve_layout(). Holds
 * the arena offsets of every sub-region the scheduler needs plus the
 * capacities used at layout time (init_from_layout reuses them).
 */
struct PTO2SchedulerLayout {
    size_t off_ready_queue_slots[PTO2_NUM_RESOURCE_SHAPES];
    size_t off_ready_sync_queue_slots[PTO2_NUM_RESOURCE_SHAPES];
    size_t off_dummy_ready_queue_slots;
    size_t off_graph_ready_queue_slots;
    size_t off_graph_prepare_queue_slots;
    size_t off_early_dispatch_queue_slots[PTO2_NUM_RESOURCE_SHAPES];
    size_t off_early_sync_start_queue_slots;
    uint64_t ready_queue_capacity;
};

/**
 * Scheduler state structure
 *
 * Contains dynamic state updated during task execution.
 * Separated from shared memory for cache efficiency.
 * Hot-path methods are defined inline (implicitly inline as member functions).
 */
struct PTO2SchedulerState {
    // Shared memory access
    PTO2SharedMemoryHeader *sm_header;

    // Per-ring state
    struct alignas(64) RingSchedState {
        // --- Cache Line 0: ring pointer (read-only) + hot path (read-write) ---
        PTO2SharedMemoryRingHeader *ring;
        std::atomic<int32_t> advance_lock;  // multi-thread CAS

        // Polling: no per-ring dep_pool. Readiness is derived from the SM ring's
        // completion_flags; there is no arena-side wiring pool to reserve or wire.
        // The `ring` field stores the device address of the SM ring header —
        // computed via offset arithmetic, no SM dereference.
        bool init_data_from_layout(void *sm_dev_base);
        void destroy();
    } ring_sched_state;

    // Ready queues remain global (scheduling is ring-agnostic)
    PTO2ReadyQueue ready_queues[PTO2_NUM_RESOURCE_SHAPES];

    // Ready sync_start queues, one per shape. A ready sync_start cohort parks here
    // instead of ready_queues[] so the dispatch loop can drain it as a strict Tier-0
    // (sync_start > MIX > C/V) before any regular ready task takes a core, while
    // reusing the same per-shape dispatch_shape machinery (fits-local inline vs
    // stop-the-world drain, per-core MIX placement, head-start spacing).
    PTO2ReadyQueue ready_sync_queues[PTO2_NUM_RESOURCE_SHAPES];

    // Dependency-only tasks (active_mask is empty, shape == DUMMY). Drained by
    // the dispatch loop and completed inline -- never goes to AICore.
    PTO2ReadyQueue dummy_ready_queue;

    // An outer Graph is control work, never an AICore task. External dependency
    // readiness and bounded materialization progress independently and meet at
    // the submission's single atomic activation gate.
    PTO2ReadyQueue graph_ready_queue;
    PTO2ReadyQueue graph_prepare_queue;

    alignas(64) AsyncWaitList async_wait_list;

    // Statistics (cold path, isolated from hot-path fields)
#if SIMPLER_SCHED_PROFILING
    alignas(64) std::atomic<int64_t> tasks_completed;
    std::atomic<int64_t> tasks_consumed;
#endif
    // =========================================================================
    // Inline hot-path methods
    // =========================================================================

    // Route a ready slot to the right global queue. Dep-only tasks — DUMMY-shaped
    // (empty active_mask) or a task whose dispatch predicate fails — live in
    // dummy_ready_queue and are retired inline; a ready sync_start cohort goes to
    // the per-shape ready_sync_queues[] (drained as Tier-0); everything else to
    // ready_queues[].
    void latch_ready_queue_overflow(int32_t thread_idx = -1) {
        int32_t expected = SIMPLER_ERROR_NONE;
        const bool latched = sm_header->sched_error_code.compare_exchange_strong(
            expected, SIMPLER_ERROR_READY_QUEUE_OVERFLOW, std::memory_order_acq_rel, std::memory_order_acquire
        );
        if (latched && thread_idx >= 0) {
            sm_header->sched_error_thread.store(thread_idx, std::memory_order_release);
        }
        if (thread_idx >= 0 && thread_idx < 32) {
            sm_header->sched_error_bitmap.fetch_or(1U << static_cast<uint32_t>(thread_idx), std::memory_order_acq_rel);
        }
    }

    bool push_graph_prepare(ChipTaskSlotState *slot_state, uint64_t task_id, int32_t thread_idx) {
        if (graph_prepare_queue.push_tagged(slot_state, task_id)) return true;
        latch_ready_queue_overflow(thread_idx);
        return false;
    }

    void push_ready_routed(ChipTaskSlotState *slot_state) {
        bool pushed;
        if (slot_state->task_kind == TaskKind::GRAPH) {
            pushed = graph_ready_queue.push(slot_state);
        } else {
            PTO2ResourceShape shape = slot_state->active_mask.to_shape();
            if (shape == PTO2ResourceShape::DUMMY ||
                (slot_state->task_attrs.has_predicate() && !slot_state->payload->predicate.pass())) {
                pushed = dummy_ready_queue.push(slot_state);
            } else if (slot_state->task_attrs.requires_sync_start()) {
                pushed = ready_sync_queues[static_cast<int32_t>(shape)].push(slot_state);
            } else {
                pushed = ready_queues[static_cast<int32_t>(shape)].push(slot_state);
            }
        }
        // Every ready / sync / dummy / graph task routes to exactly one queue. A
        // false push means that queue's peak concurrent occupancy exceeded
        // PTO2_READY_QUEUE_SIZE — a capacity mis-sizing, not a normal condition.
        // Silently dropping the task would stall the run, so latch a named error
        // (surfaces as READY_QUEUE_OVERFLOW rather than an anonymous
        // forward-progress timeout). The graph_ready push is checked identically
        // so a graph task cannot be dropped either.
        if (!pushed) {
            latch_ready_queue_overflow();
        }
    }

    // ---- Polling completion primitives (single-ring hbg) ----------------------
    // Readiness: a task is ready iff every producer named in its inline fanin has
    // set its completion_flags byte. Single-ring: all producers are ring 0, so
    // there is no per-edge ring indirection.

    // Unmet-fanin classification. Returns -1 (all fanins met -> route to ready)
    // or the index of an unmet fanin (register on that producer's wake list).
    // Scan direction is load-bearing: the builder fills the fanin region in
    // submission order, so the last unmet entry is the latest-submitted
    // producer -- the one likeliest to complete last. Targeting it minimises
    // wake-list transfers (a consumer re-registered onto a second producer once
    // its first one completes) and the CAS traffic those transfers put on the
    // lists. The decision is terminal: tasks are never re-polled; a producer's
    // completion re-scans its waiters via on_mixed_task_complete's wake drain.
    int classify_fanin_state(const ChipTaskSlotState *s) const {
        const PTO2TaskPayload &p = *s->payload;
        const PTO2SharedMemoryRingHeader &ring = *ring_sched_state.ring;
        const int32_t *fanin = p.fanin_data();
        for (int32_t i = p.fanin_count - 1; i >= 0; i--) {
            if (!ring.is_completion_flag_set(fanin[i])) return i;
        }
        return -1;
    }

    // =========================================================================
    // scan_and_claim: task discovery by scanning, not by being pushed to.
    // =========================================================================
    //
    // Find up to `max_count` dispatchable tasks of `want_shape`, scanning task
    // ids upward from the cursor. Replaces hbg's ready-queue pop.
    //
    // The cursor is `completed_watermark`: the highest id whose whole prefix
    // [0, watermark] has retired. It is initialised to -1, so the first scan
    // starts at 0. It only ever advances across a *contiguous* run of retired
    // tasks and stops dead at the first non-retired one — it is a floor, never a
    // hint, and it never passes a live task. The scan itself walks freely above
    // that floor; only the floor is contiguity-bound.
    //
    // Why scanning upward from the cursor terminates and cannot starve: task ids
    // are assigned in submission order, which is topological, so every producer
    // of task i has an id < i. Every id below the cursor has retired by
    // definition. Therefore the task *at* the cursor always has its dependencies
    // met — it may be already claimed, or need a core shape this thread has none
    // free of, but it is never unready.
    //
    // v1 scans the whole remaining window rather than a bounded prefix. A bounded
    // scan is the design's intent and its best property (cost per pass
    // independent of graph size), but a fixed cap turns the head-of-line case —
    // a long straggler pinning the cursor while the next CAP entries all depend
    // on it — from "slow" into "the forward-progress watchdog latches a hang that
    // isn't one". Correctness first; the cap arrives with its growth rule.
    //
    // No `state[]` array and no CAS claim here, because the tree already has
    // both: `completion_flags` is the RETIRED bit, and `next_block_idx` reaching
    // `logical_block_num` is the BUSY bit — advanced only through
    // claim_block_range's compare_exchange, which IS the atomic claim. A
    // partially-claimed SPMD task stays claimable, which is what lets peers take
    // its remaining blocks.
    // `out_retired` returns how many dependency-only tasks this call retired
    // inline. The caller MUST add it to the run's completed-task count: those
    // tasks are finished and will never be reported by any core-completion path,
    // so dropping the number leaves the total permanently short and the
    // "all tasks done" test can never fire — the run then dies of a
    // forward-progress timeout with a correctly-executed graph behind it.
    // Entries examined per call. Bounds the cost of one pass; `resume` carries
    // the position forward so bounding it costs coverage-per-pass, not coverage.
    static constexpr int SCAN_CAP = 128;

    // Per-thread deferred-ready FIFO: consumer ids whose fanin count hit zero
    // (observed by resolve_fanouts at some retire on this thread) and which are
    // not yet fully claimed. pop_ready_tasks_batch services this list BEFORE
    // the window scan, so a consumer readied by a FIN on this thread's core
    // dispatches on the very next pass instead of waiting for a scan cursor to
    // crawl to its slot — the discovery-latency fix for sparse ready sets
    // (the serialized drain tail). Single-owner: only thread t reads or writes
    // deferred_ready_[t], so no atomics; alignas keeps neighbors off each
    // other's cache lines.
    //
    // The list is a latency optimization, not a correctness structure: it is
    // bounded (an append to a full list is dropped), entries go stale (a peer's
    // scan claims the task first), and anything it loses the window scan
    // rediscovers from the flags/counters. An entry is removed once its task is
    // retired, fully claimed, or stale; a partially claimed SPMD task stays
    // listed so follow-up pops keep offering its remaining blocks.
    struct alignas(64) DeferredReadyFifo {
        static constexpr int32_t CAP = 256;
        int32_t ids[CAP];
        int32_t count{0};
    };
    DeferredReadyFifo deferred_ready_[PLATFORM_MAX_AICPU_THREADS]{};

    int scan_ready_tasks_batch(
        PTO2ResourceShape want_shape, ChipTaskSlotState **out, int max_count, int thread_idx,
        int32_t &out_retired, int32_t &resume
    ) {
        PTO2SharedMemoryRingHeader &ring = *ring_sched_state.ring;
        const int32_t submitted = ring.fc.current_task_index.load(std::memory_order_acquire);
        if (submitted <= 0) return 0;

        // The cursor is the floor: everything at or below it has retired, so
        // resuming below it can only waste loads.
        int32_t floor = ring.completed_watermark.load(std::memory_order_acquire) + 1;
        if (floor < 0) floor = 0;
        if (floor >= submitted) return 0;
        if (resume < floor || resume >= submitted) resume = floor;

        int found = 0;
        int examined = 0;
        bool wrapped = false;
        int32_t i = resume;
        for (; found < max_count && examined < SCAN_CAP; i++, examined++) {
            if (i >= submitted) {
                if (wrapped) break;  // one wrap per call: nothing to find right now
                wrapped = true;
                i = floor;
                if (i >= submitted) break;
            }
            // RETIRED. Acquire, so observing the flag also observes the outputs.
            if (ring.is_completion_flag_set(i)) continue;

            ChipTaskSlotState &s = ring.get_slot_state_by_task_id(i);
            if (s.task == nullptr || s.payload == nullptr) continue;  // slot not populated

            // v1 gap, skipped rather than mis-dispatched: recorded Graphs need the
            // activate / materialize path this scan does not implement.
            //
            // TaskKind::DUMMY is NOT skipped. It is a first-class kind (KERNEL=0,
            // DUMMY=1, GRAPH=2, GRAPH_NODE=3), and hbg does not special-case it
            // either — push_ready_routed branches only on GRAPH, letting a DUMMY
            // task fall through and be classified by its (empty) active_mask. A
            // `task_kind != KERNEL` filter here silently strands every dummy in
            // the graph and the run dies of a forward-progress timeout.
            if (s.task_kind == TaskKind::GRAPH || s.task_kind == TaskKind::GRAPH_NODE) continue;

            // Dependencies via the counter: one acquire load instead of walking
            // the fanin list. fanin_remaining[i] counts producers not yet
            // retired; the host wrote the initial value into the image and every
            // producer's retire decrements it (release RMW), so reading 0 with
            // acquire guarantees ALL producers' outputs are visible (RMWs form a
            // release sequence). The flags stay authoritative for the cursor and
            // the host; classify_fanin_state remains for the paths that still
            // key on flags -- the two views describe the same monotonic facts.
            if (ring.fanin_pending(i)) continue;

            const PTO2ResourceShape shape = s.active_mask.to_shape();
            const bool predicate_failed =
                s.task_attrs.has_predicate() && !s.payload->predicate.pass();

            // A dependency-only task — no active subtask slots, or a predicate
            // that evaluated false — occupies no core, so waiting for one to
            // finish would wait forever. Retire it inline, right here.
            //
            // Doing it during the scan also makes chains resolve transitively
            // within a single pass: retiring id k sets its flag before the loop
            // reaches k+1, so a consumer of k later in this same scan already
            // sees its dependency met. hbg needed a bounded drain loop
            // (DUMMY_DRAIN_BATCH) to get the same effect.
            if (shape == PTO2ResourceShape::DUMMY || predicate_failed) {
                // Claim before retiring: with N > 1, two threads can find the
                // same dep-only task ready in the same instant. The flag CAS
                // makes exactly one of them the retirer; the loser skips.
                if (!ring.try_claim_completion_flag(i)) continue;
                TaskCompletionOutcome outcome = complete_task(s, thread_idx);
                if (outcome.error_code == SIMPLER_ERROR_NONE) {
                    out_retired += outcome.stream_tasks_completed;
                }
                continue;
            }

            // From here on the task needs cores, so the resource guards apply.
            // They are deliberately AFTER the dependency-only retirement above: a
            // task that occupies no core must never be gated on core state, and
            // in particular must never be judged BUSY by a block counter it does
            // not use.
            // sync_start cohorts ARE dispatchable. hbg gave them a priority tier
            // (Tier 0 over ready_sync_queues, ahead of regular work) which M2
            // deleted along with the queues -- but that tier was a *policy*, not
            // the mechanism. The mechanism survives intact and is queue-free:
            // dispatch_shape's sync_start branch either places the whole cohort
            // outright, or calls enter_drain_mode, which parks the task in
            // drain_state_.pending_task and raises sync_start_pending; every
            // thread then converges in handle_drain_mode and stages the cohort
            // from drain_state_. No ready_sync_queue is consulted anywhere on
            // that path, so surfacing these tasks from the scan is sufficient.
            //
            // Consequence to watch: without the priority tier a sync_start task
            // competes with ordinary work for cores rather than pre-empting it.
            // enter_drain_mode stops the world to gather cores, so this should
            // cost latency rather than correctness -- but it is the reason to
            // keep an eye on wide cohorts (qwen3's per-layer paged-attention MIX
            // is block_num = attention_core_num).

            // BUSY: every logical block already claimed by some thread. Advanced
            // only via claim_block_range's compare_exchange, so this doubles as
            // the atomic claim; a partially-claimed SPMD task stays claimable.
            if (s.next_block_idx.load(std::memory_order_acquire) >= s.logical_block_num) continue;

            if (shape != want_shape) continue;
            out[found++] = &s;
        }
        // Remember where to carry on. Wrapping i back into range keeps the stored
        // value meaningful for the next call.
        resume = (i >= submitted) ? floor : i;
        return found;
    }

    // Serves pop_ready_tasks_batch from this thread's deferred-ready FIFO,
    // ahead of any window scan. Applies the scan's guards in the scan's order;
    // the fanin test alone is unnecessary — an id enters the list only at its
    // zero-crossing, and a counter at 0 stays 0. Dep-only tasks retire inline
    // exactly as in the scan, and their own zero-crossings append to this same
    // FIFO mid-iteration (`fifo.count` is re-read every step; appends land past
    // the read cursor and compaction writes only behind it), so dummy chains
    // cascade within a single call. Entries whose shape was not asked for, and
    // tasks with unclaimed blocks remaining, stay listed for later calls.
    int service_deferred_ready(
        PTO2ResourceShape want_shape, ChipTaskSlotState **out, int max_count, int thread_idx, int32_t &out_retired
    ) {
        DeferredReadyFifo &fifo = deferred_ready_[thread_idx];
        if (fifo.count == 0) return 0;
        PTO2SharedMemoryRingHeader &ring = *ring_sched_state.ring;
        int found = 0;
        int32_t kept = 0;
        for (int32_t idx = 0; idx < fifo.count; ++idx) {
            const int32_t i = fifo.ids[idx];
            bool keep = false;
            do {
                if (ring.is_completion_flag_set(i)) break;  // retired by a peer's claim
                ChipTaskSlotState &s = ring.get_slot_state_by_task_id(i);
                if (s.task == nullptr || s.payload == nullptr) break;
                if (s.task_kind == TaskKind::GRAPH || s.task_kind == TaskKind::GRAPH_NODE) break;

                // Cross-round staleness test. Within one round a listed id's
                // counter is 0 forever — counters only count down and each
                // producer decrements exactly once — so a nonzero counter
                // proves this id survived from a previous round's window,
                // where the re-uploaded image gave the slot a fresh fanin
                // count. Dispatching it would run a task whose producers have
                // not finished; drop it instead.
                if (ring.fanin_pending(i)) break;

                const PTO2ResourceShape shape = s.active_mask.to_shape();
                const bool predicate_failed = s.task_attrs.has_predicate() && !s.payload->predicate.pass();
                if (shape == PTO2ResourceShape::DUMMY || predicate_failed) {
                    // Same claim rule as the scan: the flag CAS elects exactly one
                    // retirer among this FIFO, peer FIFOs, and peer scans.
                    if (!ring.try_claim_completion_flag(i)) break;
                    TaskCompletionOutcome outcome = complete_task(s, thread_idx);
                    if (outcome.error_code == SIMPLER_ERROR_NONE) {
                        out_retired += outcome.stream_tasks_completed;
                    }
                    break;
                }
                if (s.next_block_idx.load(std::memory_order_acquire) >= s.logical_block_num) break;  // fully claimed
                keep = true;  // unclaimed blocks remain — offer again until fully claimed or stale
                if (shape != want_shape || found >= max_count) break;
                out[found++] = &s;
            } while (false);
            if (keep) fifo.ids[kept++] = fifo.ids[idx];
        }
        fifo.count = kept;
        return found;
    }

    // Producer completion under polling: publish the host-visible task_state
    // mirror + the device-visible completion_flags byte, drain the wake list
    // (route/re-register each waiter), then CAS-advance the monotonic
    // completed_watermark (load-bearing: the host wait_for_consumers gates on
    // watermark >= producer.last_consumer_local_id). Whole-graph-resident hbg
    // has no device slot reclaim, so no advance_ring_pointers here.
    void on_mixed_task_complete(ChipTaskSlotState &slot_state, int thread_idx) {
        const int32_t task_id = static_cast<int32_t>(slot_state.task->task_id.local());
        PTO2SharedMemoryRingHeader &ring = *ring_sched_state.ring;

        slot_state.mark_completed();  // host-visible mirror (task_state = COMPLETED)
        ring.set_completion_flag(task_id);
        // Counter-based dependency resolution: tell every consumer one of its
        // producers retired, and remember — in this thread's deferred-ready
        // FIFO — every consumer this retire made ready, so the next dispatch
        // pass places it without waiting for a scan cursor. Runs exactly once
        // per task, because on_mixed_task_complete itself does (the
        // last-subtask winner, or the flag-CAS winner for dep-only tasks) -- a
        // duplicated call here would over-decrement and make a consumer ready
        // EARLY, which is a wrong-answer bug, not a hang. Ordered after the
        // output publish like the flag.
        DeferredReadyFifo &fifo = deferred_ready_[thread_idx];
        fifo.count += ring.resolve_fanouts(task_id, &fifo.ids[fifo.count], DeferredReadyFifo::CAP - fifo.count);

        // scan_and_claim: no wake-list drain, and no cross-thread hand-off of
        // newly-ready consumers. Publishing the completion_flags byte and the
        // counter decrements IS the authoritative notification — every
        // scheduler thread re-derives readiness from them on its next scan, so
        // a FIFO entry that overflows, goes stale, or is claimed by a peer
        // first costs nothing. The FIFO is a single-owner latency hint in
        // front of scan rediscovery, not a hand-off protocol: there is still
        // no wakeup to lose.
        //
        // set_completion_flag is a release store and the payload writes precede
        // it, so a consumer that observes the flag (acquire, via
        // is_completion_flag_set) also observes this task's outputs.

        // completed_watermark = highest id such that every task in [0, watermark]
        // has its completion_flags byte set. The host wait_for_consumers gates on
        // watermark >= producer.last_consumer_local_id, so the walk must extend to
        // the full contiguous completed prefix — NOT cap at task_id. Capping at task_id
        // makes the final value order-dependent: a low-id task completing after a
        // higher one would leave the watermark stuck below the true prefix, hanging
        // any wait_for_consumers whose last_consumer sits in the gap.
        ring.update_completed_watermark();
    }

    // Polling: there is no ready-claim CAS (a producer routes each waiter exactly
    // once via the wake-list drain) and no per-producer consumer/scope refcount.
    // Consumer retirement is observed by the host through completed_watermark >=
    // producer.last_consumer_local_id, not by bumping a producer refcount.

    // Early-dispatch release. If the now-ready task was pre-staged
    // (gated on a core), ring its DATA_MAIN_BASE high-32 doorbell RIGHT HERE in
    // the completion path — the moment its last producer's FIN satisfies fanin —
    // instead of routing it through the ready queue and waiting for the dispatch
    // pass to pop it. Returns true if the task is fully handled (caller must NOT
    // push to the ready queue). Returns false when the caller must route C
    // normally: either it was never pre-staged, OR it is a SPMD consumer only
    // PARTIALLY pre-staged — the gated blocks are released by the doorbells rung
    // here, and the remaining (next_block_idx .. logical_block_num) blocks
    // dispatch normally off the ready queue. Lock-free claim shared with Hook 1
    // (the stager): CAS NONE->DISPATCHED wins => not pre-staged; otherwise flip
    // STAGING->DISPATCHED and destructively claim the published doorbell bits.

    // Per-core early-dispatch doorbell table. Hook 1 records each gated core's
    // (reg_addr, dispatch token) here at stage time; the completion-path release
    // reads it back for the cores set in the consumer's staged_core_mask. One
    // global table indexed by core_id (not per-task): gated cores in flight are
    // bounded by the chip's core count (no two-level pre-dispatch), so this is the
    // natural capacity and removes the old per-task 3-doorbell cap.
    struct EarlyDispatchDoorbell {
        uint64_t addr{0};
        uint32_t token{0};
    };
    EarlyDispatchDoorbell early_dispatch_doorbell_table[PTO2_EARLY_DISPATCH_CORE_MASK_WORDS * 64]{};

    // Cross-thread early-dispatch work queues, one PTO2ReadyQueue MPMC instance per
    // resource shape (AIC/AIV/MIX) — arena-backed, reserved/wired in pto_runtime2_init
    // alongside the per-shape ready queues, and indexed the same way. A candidate is
    // pushed to the queue for its own shape (active_mask.to_shape()) so the drain can
    // pop per shape and size the pop to that shape's free cores, exactly as normal
    // dispatch pops ready_queues[shape].
    //
    // A consumer's SPMD blocks span cores owned by several AICPU threads, but only a
    // thread RUNNING the consumer's producer discovers it (via the producer's
    // fanout). When that producer is thread-local (e.g. a 16-block AIV op filling one
    // thread's cores), the other threads never see the consumer and its blocks on
    // their cores can't pre-stage. The first claimer pushes the partially-staged
    // consumer here; every idle thread's early_dispatch pass pops one, stages a range
    // onto ITS OWN cores (range-claim via next_block_idx), and re-pushes if blocks
    // remain — exactly mirroring how a partially-dispatched SPMD task is re-pushed to
    // the ready queue (scheduler_dispatch: pop -> claim -> re-push). A stale/released
    // entry fails the STAGING check on pop and is dropped; a push that overflows is
    // logged and the consumer's blocks fall back to normal dispatch.
    PTO2ReadyQueue early_dispatch_queues[PTO2_NUM_RESOURCE_SHAPES];

    // sync_start early-dispatch candidates park here instead of early_dispatch_queues[]:
    // they need an atomic all-or-nothing stage, not early_dispatch_shape's
    // per-thread partial range-claim. Shape-agnostic (the rendezvous counts cores,
    // not blocks), so a single queue serves all shapes and one owner selects the
    // local stage or global-drain fallback.
    //
    // Deliberately single, vs the normal source's per-shape ready_sync_queues[]: a READY
    // sync cohort (producer done) can dispatch inline when it fits, so it reuses the
    // per-shape dispatch_shape. An EARLY sync cohort carries a non-zero src_payload
    // gate; its owner stages locally when one tracker fits and uses the global drain
    // otherwise. HBG producer propagation currently leaves this queue dormant.
    PTO2ReadyQueue early_sync_start_queue;

    static inline void ring_one_doorbell(uint64_t reg_addr, uint32_t token) {
        volatile uint64_t *dmb = reinterpret_cast<volatile uint64_t *>(get_reg_ptr(reg_addr, RegId::DATA_MAIN_BASE));
        uint64_t tk = static_cast<uint64_t>(token);
        *dmb = (tk << 32) | tk;  // 64-bit STR: high=low=token releases the gated AICore
    }

    inline void ring_staged_doorbell_bits(int word, uint64_t bits) {
        while (bits != 0) {
            int core_id = word * 64 + __builtin_ctzll(bits);
            bits &= bits - 1;
            ring_one_doorbell(
                early_dispatch_doorbell_table[core_id].addr, early_dispatch_doorbell_table[core_id].token
            );
        }
    }

    static inline uint64_t claim_all_staged_doorbell_bits(std::atomic<uint64_t> &mask) {
        return mask.exchange(0, std::memory_order_seq_cst);
    }

    static inline uint64_t claim_late_staged_doorbell_bits(std::atomic<uint64_t> &mask, uint64_t candidates) {
        return mask.fetch_and(~candidates, std::memory_order_seq_cst) & candidates;
    }

    static inline bool should_gate_early_dispatch(bool force_gate, uint8_t early_dispatch_state) {
        return force_gate || early_dispatch_state == PTO2_EARLY_DISPATCH_STAGING;
    }

    static inline bool
    ring_claimed_local_doorbell(uint64_t claimed_word, int core_id, uint64_t reg_addr, uint32_t token) {
        if ((claimed_word & (1ULL << (core_id & 63))) == 0) return false;
        ring_one_doorbell(reg_addr, token);
        return true;
    }

    static inline bool try_claim_early_dispatch_launch(PTO2TaskPayload &payload) {
        uint8_t expected = PTO2_EARLY_DISPATCH_LAUNCH_NONE;
        return payload.early_dispatch_launch_state.compare_exchange_strong(
            expected, PTO2_EARLY_DISPATCH_LAUNCH_RINGING, std::memory_order_seq_cst, std::memory_order_seq_cst
        );
    }

    inline void record_published_blocks(ChipTaskSlotState &slot_state, int32_t count) {
        if (count <= 0 || !slot_state.task_attrs.allow_early_resolve()) return;
        slot_state.payload->published_block_count.fetch_add(static_cast<int16_t>(count), std::memory_order_seq_cst);
    }

    // Ring one sync_start cohort from its stable staged_core_mask. The caller owns
    // the NONE->RINGING launch latch and invokes this exactly once after local or
    // global staging completes, while the corresponding per-core table entries are live.
    inline void ring_all_staged_doorbells(ChipTaskSlotState &slot_state) {
        for (int w = 0; w < PTO2_EARLY_DISPATCH_CORE_MASK_WORDS; w++) {
            uint64_t bits = slot_state.payload->staged_core_mask[w].load(std::memory_order_seq_cst);
            while (bits != 0) {
                int core_id = w * 64 + __builtin_ctzll(bits);
                bits &= bits - 1;
                ring_one_doorbell(
                    early_dispatch_doorbell_table[core_id].addr, early_dispatch_doorbell_table[core_id].token
                );
            }
        }
    }

    static inline bool try_claim_early_sync_drain(PTO2TaskPayload &payload) {
        uint8_t expected = PTO2_EARLY_SYNC_DRAIN_NONE;
        return payload.early_sync_drain_state.compare_exchange_strong(
            expected, PTO2_EARLY_SYNC_DRAIN_OWNER, std::memory_order_seq_cst, std::memory_order_seq_cst
        );
    }

    static inline bool owns_early_sync_drain(const PTO2TaskPayload &payload) {
        return (payload.early_sync_drain_state.load(std::memory_order_acquire) & PTO2_EARLY_SYNC_DRAIN_OWNER) != 0;
    }

    static inline void mark_early_sync_drain_armed(PTO2TaskPayload &payload) {
        payload.early_sync_drain_state.fetch_or(PTO2_EARLY_SYNC_DRAIN_ARMED, std::memory_order_seq_cst);
    }

    static inline bool publish_ready_to_early_sync_drain(PTO2TaskPayload &payload) {
        uint8_t previous =
            payload.early_sync_drain_state.fetch_or(PTO2_EARLY_SYNC_DRAIN_READY, std::memory_order_seq_cst);
        return (previous & PTO2_EARLY_SYNC_DRAIN_OWNER) != 0;
    }

    inline void cancel_early_sync_drain(ChipTaskSlotState &slot_state) {
        uint8_t previous =
            slot_state.payload->early_sync_drain_state.exchange(PTO2_EARLY_SYNC_DRAIN_NONE, std::memory_order_seq_cst);
        if ((previous & PTO2_EARLY_SYNC_DRAIN_OWNER) == 0) return;
        if ((previous & PTO2_EARLY_SYNC_DRAIN_READY) != 0) {
            push_ready_routed(&slot_state);
            return;
        }
        if (slot_state.payload->early_dispatch_state.load(std::memory_order_seq_cst) == PTO2_EARLY_DISPATCH_STAGING) {
            early_sync_start_queue.push_tagged(&slot_state, static_cast<uint64_t>(slot_state.task->task_id.raw));
        }
    }

    static inline void finish_early_sync_drain(PTO2TaskPayload &payload) {
        uint8_t state = payload.early_sync_drain_state.load(std::memory_order_seq_cst);
        while ((state & PTO2_EARLY_SYNC_DRAIN_OWNER) != 0 && (state & PTO2_EARLY_SYNC_DRAIN_COMPLETE) == 0) {
            uint8_t desired = state | PTO2_EARLY_SYNC_DRAIN_COMPLETE;
            if (payload.early_sync_drain_state.compare_exchange_weak(
                    state, desired, std::memory_order_seq_cst, std::memory_order_seq_cst
                )) {
                return;
            }
        }
    }
    // sync_start rendezvous: a sync_start consumer's gated cores launch as an atomic
    // cohort, so their doorbells are held until BOTH halves hold — every gated core
    // occupies a running slot (running_slot_count == popcount(staged_core_mask)) AND the
    // producer released (early_dispatch_state == DISPATCHED). Counting CORES (not blocks) makes
    // this shape-agnostic: an AIC/AIV block is one core, a MIX block is a cluster whose
    // cores promote pending->running independently. Called from both halves (the producer
    // release and each pending->running promotion); whichever observes the second half
    // wins the launch latch and rings exactly once. Returns true only to that winner,
    // which may then expose the cohort to its fanout.
    inline bool maybe_rendezvous_ring(ChipTaskSlotState &slot_state) {
        // running_slot_count is the publication seed: every staged_core_mask OR
        // happens-before its final store. Read the seed first, then the mask, so
        // observing the final count cannot be paired with a partially published
        // mask when producer release races staging.
        int32_t running_cores = slot_state.payload->running_slot_count.load(std::memory_order_seq_cst);
        int32_t staged_cores = 0;
        for (int w = 0; w < PTO2_EARLY_DISPATCH_CORE_MASK_WORDS; w++)
            staged_cores +=
                __builtin_popcountll(slot_state.payload->staged_core_mask[w].load(std::memory_order_seq_cst));
        if (staged_cores == 0) return false;
        if (running_cores != staged_cores) return false;
        if (slot_state.payload->early_dispatch_state.load(std::memory_order_seq_cst) != PTO2_EARLY_DISPATCH_DISPATCHED)
            return false;
        if (!try_claim_early_dispatch_launch(*slot_state.payload)) return false;
        ring_all_staged_doorbells(slot_state);
        wmb();
        slot_state.payload->early_dispatch_launch_state.store(
            PTO2_EARLY_DISPATCH_LAUNCH_COMPLETE, std::memory_order_release
        );
        return true;
    }

    inline bool retry_sync_start_rendezvous_after_staging(ChipTaskSlotState &slot_state) {
        if (!maybe_rendezvous_ring(slot_state)) return false;
        propagate_dispatch_fanin(slot_state);
        return true;
    }

    // Milestone 1: early-dispatch (predicated / allow_early_resolve) is stubbed.
    // This producer-push propagation walked the wiring fanout list bumping each
    // consumer's dispatch_fanin to pre-stage early-dispatch candidates — all of
    // which (fanout_head, dispatch_fanin, fanin_actual_count, dispatch_propagated)
    // are gone under polling. Nothing pre-stages into early_dispatch_queues /
    // early_sync_start_queue, so tasks reach cores only through the normal ready
    // path (wake drain -> push_ready_routed). Milestone 2 replaces this with the
    // consumer-pull publish_flags design. sync_start cohorts still launch via
    // ready_sync_queues (unaffected).
    void propagate_dispatch_fanin(ChipTaskSlotState & /*p*/) {}

    // Orch owns dependency discovery and saves the immutable fanin wire.
    // Scheduler polling only chooses which already-wired producer a consumer
    // waits on at this instant; it never recomputes producer relationships.
    int32_t graph_first_unmet_producer(const GraphExecution &execution, const ChipTaskSlotState &consumer) const {
        const uint32_t node_index = static_cast<uint32_t>(consumer.graph_node_index);
        const uint32_t begin = execution.fanin_offsets[node_index];
        const uint32_t end = execution.fanin_offsets[node_index + 1];
        for (uint32_t edge = begin; edge < end; ++edge) {
            const uint16_t producer_index = execution.fanin_indices[edge];
            const ChipTaskSlotState &producer = execution.node_at(producer_index).slot;
            if (producer.task_state.load(std::memory_order_acquire) != PTO2_TASK_COMPLETED) {
                return static_cast<int32_t>(producer_index);
            }
        }
        return -1;
    }

    void register_graph_wake(GraphExecution &execution, ChipTaskSlotState *producer, ChipTaskSlotState *consumer) {
        while (true) {
            ChipTaskSlotState *expected = producer->wake_list_head.load(std::memory_order_relaxed);
            while (expected != WAKE_LIST_SENTINEL) {
                consumer->next_in_wake_list = expected;
                if (producer->wake_list_head.compare_exchange_weak(
                        expected, consumer, std::memory_order_acq_rel, std::memory_order_relaxed
                    )) {
                    return;
                }
            }

            // The producer completed between fanin classification and the CAS.
            // Retry against acquire-loaded task states until cache coherence
            // exposes the completed producer, then route or retarget the waiter.
            const int32_t unmet_producer = graph_first_unmet_producer(execution, *consumer);
            if (unmet_producer < 0) {
                push_ready_routed(consumer);
                return;
            }
            producer = &execution.node_at(unmet_producer).slot;
        }
    }

    uint32_t drain_graph_wake_list(GraphExecution &execution, ChipTaskSlotState &producer) {
        uint32_t consumers_rescanned = 0;
        ChipTaskSlotState *waiter = producer.wake_list_head.exchange(WAKE_LIST_SENTINEL, std::memory_order_acq_rel);
        while (waiter != nullptr && waiter != WAKE_LIST_SENTINEL) {
            ChipTaskSlotState *next = waiter->next_in_wake_list;
            const int32_t unmet_producer = graph_first_unmet_producer(execution, *waiter);
            if (unmet_producer < 0) {
                push_ready_routed(waiter);
            } else {
                register_graph_wake(execution, &execution.node_at(unmet_producer).slot, waiter);
            }
            consumers_rescanned++;
            waiter = next;
        }
        return consumers_rescanned;
    }

    // Push every materialized-and-published root that has not been routed yet,
    // once the outer Graph task's external dependencies are ready.
    // route_cursor makes this idempotent, so it composes across the per-slice
    // calls during materialization and the final call at the activation meet;
    // each root reaches the ready queue exactly once. Non-roots are never pushed
    // here — they reach the ready queue through their producers' wake list.
    int32_t graph_route_ready_roots(GraphExecution &execution) {
        if (execution.outer_slot == nullptr || !graph_execution_external_ready(execution)) return 0;
        const int32_t published = execution.published_nodes.load(std::memory_order_acquire);
        int32_t routed = 0;
        while (true) {
            int32_t i = execution.route_cursor.load(std::memory_order_relaxed);
            if (i >= published) break;
            if (!execution.route_cursor.compare_exchange_weak(
                    i, i + 1, std::memory_order_acq_rel, std::memory_order_relaxed
                )) {
                continue;
            }
            if (execution.fanin_offsets[i] == execution.fanin_offsets[i + 1]) {
                push_ready_routed(&execution.node_at(i).slot);
                routed++;
            }
        }
        return routed;
    }

    // Register each newly materialized node [first, last) on its first unmet
    // producer (or route it immediately when every producer already completed),
    // publish the range for routing, and route any roots the external gate now
    // admits. Runs single-owner per graph via the prepare-queue slot, so the
    // range never overlaps another thread's. register_graph_wake and
    // graph_first_unmet_producer are safe against a producer completing
    // concurrently, which is what lets a node dispatch before the whole graph is
    // materialized.
    void graph_incremental_publish(GraphExecution &execution, int32_t first, int32_t last) {
        for (int32_t i = first; i < last; ++i) {
            if (execution.fanin_offsets[i] == execution.fanin_offsets[i + 1]) continue;  // root
            ChipTaskSlotState &node = execution.node_at(i).slot;
            const int32_t unmet = graph_first_unmet_producer(execution, node);
            if (unmet < 0) {
                push_ready_routed(&node);
            } else {
                register_graph_wake(execution, &execution.node_at(unmet).slot, &node);
            }
        }
        execution.published_nodes.store(last, std::memory_order_release);
        graph_route_ready_roots(execution);
    }

    int32_t activate_prepared_graph(GraphExecution &execution) {
        if (!graph_execution_transition(execution, GraphExecutionState::PREPARED, GraphExecutionState::ACTIVE)) {
            return 0;
        }
        return graph_route_ready_roots(execution);
    }

    GraphMaterializeResult prepare_graph_task(
        ChipTaskSlotState &outer_slot, int32_t max_nodes = GRAPH_MATERIALIZE_SLICE_NODES,
        int32_t *nodes_materialized = nullptr
    ) {
        GraphExecution *execution = graph_execution_from_outer_slot(outer_slot);
        if (execution == nullptr) return GraphMaterializeResult::INVALID;
        const int32_t before = execution->materialized_nodes;
        const GraphMaterializeResult result =
            graph_execution_materialize_slice(outer_slot, *execution, max_nodes, nodes_materialized);
        if (result == GraphMaterializeResult::PENDING || result == GraphMaterializeResult::PREPARED) {
            graph_incremental_publish(*execution, before, execution->materialized_nodes);
        }
        if (result == GraphMaterializeResult::PREPARED && graph_execution_external_ready(*execution)) {
            activate_prepared_graph(*execution);
        }
        return result;
    }

    int32_t activate_graph_task(ChipTaskSlotState &outer_slot) {
        GraphExecution *execution = graph_execution_from_outer_slot(outer_slot);
        if (execution == nullptr) return 0;
        graph_execution_signal_external_ready(*execution);
        return activate_prepared_graph(*execution);
    }

    struct TaskCompletionOutcome {
        uint32_t fanout_edges{0};
        int32_t stream_tasks_completed{0};
        int32_t error_code{SIMPLER_ERROR_NONE};
    };

    TaskCompletionOutcome complete_task(ChipTaskSlotState &slot_state, int thread_idx) {
        TaskCompletionOutcome outcome;
        if (slot_state.task_kind != TaskKind::GRAPH_NODE) {
#if SIMPLER_SCHED_PROFILING
            CompletionStats stats = on_task_complete(slot_state, thread_idx);
            outcome.fanout_edges = static_cast<uint32_t>(stats.fanout_edges);
#else
            outcome.fanout_edges = on_task_complete(slot_state, thread_idx);
#endif
            outcome.stream_tasks_completed = 1;
            return outcome;
        }

        GraphExecution *execution = graph_execution_from_slot(slot_state);
        if (execution == nullptr || execution->definition == nullptr || execution->nodes == nullptr) {
            outcome.error_code = SIMPLER_ERROR_INVALID_ARGS;
            return outcome;
        }
        // Incremental activation routes a node before the graph reaches ACTIVE, so a
        // node can legitimately complete while the graph is still MATERIALIZING or
        // PREPARED. Only SUBMITTED (execution not yet bound) and COMPLETED
        // (execution already retired) are invalid states for a node completion.
        const GraphExecutionState graph_state = graph_execution_state(*execution);
        if (graph_state < GraphExecutionState::MATERIALIZING || graph_state > GraphExecutionState::ACTIVE) {
            outcome.error_code = SIMPLER_ERROR_INVALID_ARGS;
            return outcome;
        }
        const int32_t saved_node_index = slot_state.graph_node_index;
        if (saved_node_index < 0) {
            outcome.error_code = SIMPLER_ERROR_INVALID_ARGS;
            return outcome;
        }
        const uint32_t node_index = static_cast<uint32_t>(saved_node_index);
        if (node_index >= static_cast<uint32_t>(execution->node_count)) {
            outcome.error_code = SIMPLER_ERROR_INVALID_ARGS;
            return outcome;
        }

        // Publish completion before closing the wake list. A consumer that
        // loses registration to the sentinel acquires this state when it
        // rescans the Orch-built fanin wire, so no wakeup can be lost.
        slot_state.mark_completed();
        outcome.fanout_edges = drain_graph_wake_list(*execution, slot_state);

        const bool graph_completed = graph_execution_complete_node(*execution);
        graph_execution_retire_node(*execution);
        if (!graph_completed) return outcome;

        // Internal nodes count as zero stream tasks. The final node publishes
        // the outer ring task exactly once, waking external consumers and
        // contributing the one task the host actually submitted.
        if (execution->outer_slot != nullptr) {
            on_mixed_task_complete(*execution->outer_slot, thread_idx);
            outcome.stream_tasks_completed = 1;
        }
        graph_execution_mark_completed(*execution);
        return outcome;
    }

    /**
     * Subtask completion: atomic counter model.
     * Called when a single subtask (AIC, AIV0, or AIV1) finishes on any block.
     * Atomically increments completed_subtasks and checks whether all subtasks
     * across all blocks are done.
     *
     * @return true if this was the last subtask, completing the entire task.
     */
    bool on_subtask_complete(ChipTaskSlotState &slot_state) {
        int16_t prev = slot_state.completed_subtasks.fetch_add(1, std::memory_order_acq_rel);
        return (prev + 1) == slot_state.total_required_subtasks;
    }

    /**
     * Two-stage completion: second stage.
     * Called exactly once when all subtasks of a task are done (i.e.,
     * on_subtask_complete returned true). Walks the consumer (fanout) list,
     * decrements each consumer's fanin, pushes newly-ready ones, and rings
     * doorbells for early-dispatch hits.
     *
     * Non-PROFILING returns the consumer-walk count (= edges traversed). The
     * Resolve swimlane bar reads it to label the bar with how many successors
     * actually got resolved. PROFILING returns the richer CompletionStats
     * whose `fanout_edges` carries the same number.
     */
#if SIMPLER_SCHED_PROFILING
    CompletionStats
#else
    uint32_t
#endif
    on_task_complete(ChipTaskSlotState &slot_state, int thread_idx) {
        // Polling completion: publish the host-visible task_state mirror + the
        // device-visible completion_flags byte, drain the wake list (route or
        // re-register each waiter), and advance the watermark. Replaces the
        // fanout-list walk + fanin_refcount decrements of the wiring model.
        on_mixed_task_complete(slot_state, thread_idx);
#if SIMPLER_SCHED_PROFILING
        // Resolved-successor accounting is not tracked on the polling path (the
        // producer no longer enumerates its consumers); report 0 for the DFX bar.
        return CompletionStats{0, 0, 0, true};
#else
        return 0;
#endif
    }

    // on_task_release is gone under polling. It existed to bump each producer's
    // fanout_refcount so the host wait_for_consumers could observe consumer
    // retirement; that signal is now the per-ring completed_watermark advanced by
    // on_mixed_task_complete. There is likewise no self CONSUMED flip (host-orch
    // never reclaimed slots on device).

    // === Cold-path API (defined in scheduler.cpp) ===

    // Phase 1: declare every sub-region (ready_queue slots, dummy queue slots,
    // per-ring dep_pool entries) on the supplied arena.
    // Capacities are baked into the returned layout; init_data_from_layout uses
    // the same values.
    static PTO2SchedulerLayout reserve_layout(DeviceArena &arena);

    // Phase 3a: write everything *except* arena-internal pointer fields.
    // `sm_dev_base` is the device address of the SM (only stored, never
    // dereferenced here). Safe to call on a host arena that holds the
    // prebuilt image buffer. (The orchestrator counterpart takes
    // task_window_size for ring task_descriptors address arithmetic; the
    // scheduler only needs the SM header / ring header base addresses,
    // both window-size-independent.)
    bool init_data_from_layout(const PTO2SchedulerLayout &layout, DeviceArena &arena, void *sm_dev_base);

    // Phase 3b: write the arena-internal pointer fields
    // (ready_queues[].slots, dummy_ready_queue.slots, dep_pool.base for each
    // ring). Called on both host and device sides.
    void wire_arena_pointers(const PTO2SchedulerLayout &layout, DeviceArena &arena);

    // Phase 3c, device only: bring every queue's slots[] to its empty ramp. The
    // slot arrays sit past the uploaded range, so this is the only thing that
    // establishes the sequence values push compares against. Runs once per arena
    // after wire_arena_pointers and before any push; a drained queue needs no
    // repeat, since a free slot's sequence tracks the position it serves.
    void seed_queue_slots();

    // Forget per-region pointers; arena owns the backing memory.
    void destroy();
    void print_stats();
    void print_queues();
};

// Scheduler cold-path API is declared as PTO2SchedulerState member functions.
// See init()/destroy()/print_stats()/print_queues() below the struct definition.

// try_inline_complete_locked: short-circuit NotDeferred completions seen during
// drain so they don't grow entries[]. Defined here (not in async_wait.h)
// because PTO2SchedulerState's on_task_complete signature is only known
// after its full definition above.
//
// Polling: on_task_complete publishes completion + drains the wake list inline,
// so the async-drain path no longer buffers producer releases.
inline bool
AsyncWaitList::try_inline_complete_locked(AsyncWaitList::DrainCompletionSink &sink, ChipTaskSlotState &slot_state) {
    // Return value (CompletionStats / consumer-walk count) discarded:
    // async-wait drain path has no Resolve swimlane bar attached.
    PTO2SchedulerState::TaskCompletionOutcome outcome = sink.sched->complete_task(slot_state, sink.thread_idx);
    if (outcome.error_code != SIMPLER_ERROR_NONE) {
        sink.error_code = outcome.error_code;
        return false;
    }
    sink.inline_completed += outcome.stream_tasks_completed;
    sink.inline_resolved++;
    return true;
}

template <bool Profiling>
inline AsyncPollResult
AsyncWaitList::poll_and_complete(AICoreCompletionMailbox *aicore_mailbox, PTO2SchedulerState *sched, int thread_idx) {
    AsyncPollResult result;
    if (!try_lock()) return result;

    AsyncWaitList::DrainCompletionSink sink{};
    sink.sched = sched;
    sink.thread_idx = thread_idx;

    int32_t drain_err = SIMPLER_ERROR_NONE;
    drain_aicore_completion_mailbox_locked(aicore_mailbox, sink, drain_err);
    if (drain_err != SIMPLER_ERROR_NONE) {
        result.error_code = drain_err;
        unlock();
        return result;
    }
    result.completed += sink.inline_completed;
    result.resolved += sink.inline_resolved;

    for (int32_t i = count - 1; i >= 0; --i) {
        AsyncWaitEntry &entry = entries[i];
        uintptr_t last_invalidated_counter_line = static_cast<uintptr_t>(-1);
        for (int32_t c = 0; c < entry.condition_count; c++) {
            CompletionCondition &cond = entry.conditions[c];
            if (cond.satisfied) continue;
            if (cond.completion_type == COMPLETION_TYPE_COUNTER && cond.counter_addr != nullptr) {
                uintptr_t counter_line = mailbox_cache_line(cond.counter_addr);
                if (counter_line != last_invalidated_counter_line) {
                    cache_invalidate_range(reinterpret_cast<const void *>(counter_line), sizeof(uint32_t));
                    last_invalidated_counter_line = counter_line;
                }
            }
            CompletionPollResult poll = cond.test();
            if (poll.state == CompletionPollState::FAILED) {
                result.error_code = poll.error_code;
                result.failed_slot_state = entry.slot_state;
                unlock();
                return result;
            }
            if (poll.state == CompletionPollState::READY) {
                cond.satisfied = true;
                cond.retire();
                entry.waiting_completion_count--;
            }
        }

        if (entry.normal_done && entry.waiting_completion_count <= 0) {
            // Return value (CompletionStats / consumer-walk count) discarded:
            // deferred-completion drain has no Resolve swimlane bar attached.
            PTO2SchedulerState::TaskCompletionOutcome outcome = sched->complete_task(*entry.slot_state, thread_idx);
            if (outcome.error_code != SIMPLER_ERROR_NONE) {
                result.error_code = outcome.error_code;
                result.failed_slot_state = entry.slot_state;
                unlock();
                return result;
            }
            // Polling: completion is fully published inline; no deferred release.
            result.completed += outcome.stream_tasks_completed;
            result.resolved++;

            int32_t last = count - 1;
            if (i != last) entries[i] = entries[last];
            count = last;
        }
    }

    unlock();
    return result;
}

// =============================================================================
// Scheduler Profiling Data
// =============================================================================

#if SIMPLER_SCHED_PROFILING
struct PTO2SchedProfilingData {
    // Sub-phase cycle breakdown within on_task_complete
    uint64_t lock_cycle;           // lock_fanout + state store + unlock
    uint64_t fanout_cycle;         // fanout traversal
    uint64_t fanin_cycle;          // fanin traversal
    uint64_t self_consumed_cycle;  // self check_and_handle_consumed

    // Wait times
    uint64_t lock_wait_cycle;  // Legacy (wiring): fanout_lock spin-wait; polling has no such lock
    uint64_t push_wait_cycle;  // CAS contention in push()
    uint64_t pop_wait_cycle;   // CAS contention in pop()

    // Atomic counts per sub-phase
    uint64_t lock_atomic_count;
    uint64_t fanout_atomic_count;
    uint64_t fanin_atomic_count;
    uint64_t self_atomic_count;
    uint64_t pop_atomic_count;

    int64_t complete_count;
};

/**
 * Get and reset scheduler profiling data for a specific thread.
 * Returns accumulated profiling data and resets counters.
 */
PTO2SchedProfilingData scheduler_get_profiling(int thread_idx);
#endif
