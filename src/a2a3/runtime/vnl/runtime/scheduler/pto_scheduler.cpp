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
 * PTO Runtime2 - Scheduler Implementation
 *
 * Implements scheduler state management, ready queues, and task lifecycle.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_scheduler.h"
#include <inttypes.h>
#include <stdlib.h>
#include "common/unified_log.h"


// =============================================================================
// Ready Queue Implementation
// =============================================================================

size_t ready_queue_reserve_layout(DeviceArena &arena, uint64_t capacity) {
    // Align the slots[] base to a full cache line so MPMC CAS traffic on the
    // first slot cannot false-share with whatever region sits in front of us
    // (e.g. orchestrator tensormap heads written by the orch thread).
    return arena.reserve(capacity * sizeof(PTO2ReadyQueueSlot), PTO2_ALIGN_SIZE);
}

bool ready_queue_init_from_layout(PTO2ReadyQueue *queue, DeviceArena &arena, size_t slots_off, uint64_t capacity) {
    queue->slots = static_cast<PTO2ReadyQueueSlot *>(arena.region_ptr(slots_off));
    queue->capacity = capacity;
    queue->mask = capacity - 1;
    queue->enqueue_pos.store(0, std::memory_order_relaxed);
    queue->dequeue_pos.store(0, std::memory_order_relaxed);

    for (uint64_t i = 0; i < capacity; i++) {
        queue->slots[i].sequence.store((int64_t)i, std::memory_order_relaxed);
        queue->slots[i].slot_state = nullptr;
    }

    return true;
}

void ready_queue_destroy(PTO2ReadyQueue *queue) {
    // Arena owns the slots[] buffer; just forget the pointer.
    queue->slots = nullptr;
}

// =============================================================================
// Scheduler Initialization
// =============================================================================

bool PTO2SchedulerState::RingSchedState::init(PTO2SharedMemoryHeader *sm_header, int32_t ring_id) {
    ring = &sm_header->rings[ring_id];
    last_task_alive = 0;
    advance_lock.store(0, std::memory_order_relaxed);

    // Initialize all per-task slot state fields.
    // bind() sets payload, task, ring_id — immutable after init, bound once
    // to their fixed shared-memory addresses.
    // reset_for_reuse() sets dynamic fields to reclaim defaults (fanout_count=1,
    // rest zero) so the first submit needs no reset.
    for (uint64_t i = 0; i < ring->task_window_size; i++) {
        ring->slot_states[i].bind(&ring->task_payloads[i], &ring->task_descriptors[i], static_cast<uint8_t>(ring_id));
        ring->slot_states[i].reset_for_reuse();
        ring->slot_states[i].fanin_count = 0;
        ring->slot_states[i].active_mask = ActiveMask{};
    }

    return true;
}

void PTO2SchedulerState::RingSchedState::destroy() { ring = nullptr; }

PTO2SchedulerLayout PTO2SchedulerState::reserve_layout(DeviceArena &arena, int32_t dep_pool_capacity) {
    PTO2SchedulerLayout layout{};
    layout.ready_queue_capacity = PTO2_READY_QUEUE_SIZE;
    layout.spsc_capacity = PTO2_WRIRING_QUEUE_SIZE;
    layout.dep_pool_capacity = dep_pool_capacity;

    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        layout.off_ready_queue_slots[i] = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
    }
    layout.off_dummy_ready_queue_slots = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        // Force a cache-line base so writes from scheduler thread 0 (sole
        // writer of this ring's dep_pool) do not invalidate adjacent
        // multi-threaded regions like ready_queue.slots.
        layout.off_dep_pool_entries[r] =
            arena.reserve(static_cast<size_t>(dep_pool_capacity) * sizeof(PTO2DepListEntry), PTO2_ALIGN_SIZE);
    }
    layout.off_wiring_spsc_buffer = PTO2SpscQueue::reserve_layout(arena, PTO2_WRIRING_QUEUE_SIZE);
    return layout;
}

bool PTO2SchedulerState::init_from_layout(
    const PTO2SchedulerLayout &layout, DeviceArena &arena, PTO2SharedMemoryHeader *sm_header_arg
) {
    PTO2SchedulerState *sched = this;
    sched->sm_header = sm_header_arg;

    // Per-ring scheduler state — no arena buffers, just field init.
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        if (!sched->ring_sched_states[r].init(sm_header_arg, r)) {
            return false;
        }
    }

    // Ready queues — one per resource shape plus DUMMY.
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        if (!ready_queue_init_from_layout(
                &sched->ready_queues[i], arena, layout.off_ready_queue_slots[i], layout.ready_queue_capacity
            )) {
            return false;
        }
    }
    if (!ready_queue_init_from_layout(
            &sched->dummy_ready_queue, arena, layout.off_dummy_ready_queue_slots, layout.ready_queue_capacity
        )) {
        return false;
    }

    // Per-ring dep_pool: PTO2DepListPool::init takes an externally-allocated
    // base + capacity, so we just plumb the arena region into it.
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        auto *dep_entries = static_cast<PTO2DepListEntry *>(arena.region_ptr(layout.off_dep_pool_entries[r]));
        // calloc-equivalent: pool expects entries zeroed at construction.
        memset(dep_entries, 0, static_cast<size_t>(layout.dep_pool_capacity) * sizeof(PTO2DepListEntry));
        sched->ring_sched_states[r].dep_pool.init(
            dep_entries, layout.dep_pool_capacity, &sm_header_arg->orch_error_code
        );
    }

    // Wiring SPSC queue (orchestrator push, scheduler thread 0 pop).
    if (!sched->wiring.queue.init_from_layout(arena, layout.off_wiring_spsc_buffer, layout.spsc_capacity)) {
        return false;
    }
    sched->wiring.batch_count = 0;
    sched->wiring.batch_index = 0;
    sched->wiring.backoff_counter = 0;

    return true;
}

void PTO2SchedulerState::destroy() {
    PTO2SchedulerState *sched = this;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        sched->ring_sched_states[r].destroy();
        sched->ring_sched_states[r].dep_pool.base = nullptr;
    }

    sched->wiring.queue.destroy();

    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        ready_queue_destroy(&sched->ready_queues[i]);
    }
    ready_queue_destroy(&sched->dummy_ready_queue);
}
