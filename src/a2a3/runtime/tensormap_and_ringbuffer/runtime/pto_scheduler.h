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

#pragma once

#include <atomic>

#include "common/core_type.h"
#include "utils/device_arena.h"
#include "pto_async_wait.h"
#include "pto_ring_buffer.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"

struct PTO2ReadyQueueSlot
{
    std::atomic<int64_t> sequence;
    PTO2TaskSlotState *slot_state;
};

// Number of CoreType values eligible for local dispatch (AIC=0, AIV=1)
static constexpr int PTO2_LOCAL_DISPATCH_TYPE_NUM = 2;

struct PTO2LocalReadyBuffer
{
    PTO2TaskSlotState **slot_states = nullptr;
    int count = 0;
    int capacity = 0;

    void reset(PTO2TaskSlotState **buf, int cap)
    {
        slot_states = buf;
        count = 0;
        capacity = cap;
    }

    bool try_push(PTO2TaskSlotState *s)
    {
        if (slot_states && count < capacity)
        {
            slot_states[count++] = s;
            return true;
        }
        return false;
    }

    PTO2TaskSlotState *pop()
    {
        return (count > 0) ? slot_states[--count] : nullptr;
    }
};

struct alignas(64) PTO2ReadyQueue
{
    PTO2ReadyQueueSlot *slots;
    uint64_t capacity;
    uint64_t mask;        // capacity - 1
    char _pad0[64 - 24];  // Pad to own cache line

    std::atomic<uint64_t> enqueue_pos;
    char _pad1[64 - sizeof(std::atomic<uint64_t>)];  // Own cache line

    std::atomic<uint64_t> dequeue_pos;
    char _pad2[64 - sizeof(std::atomic<uint64_t>)];  // Own cache line

    uint64_t size()
    {
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        return (e >= d) ? (e - d) : 0;
    }

    bool push(PTO2TaskSlotState *slot_state)
    {
        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
        while (true)
        {
            pos = enqueue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - static_cast<int64_t>(pos);
            if (diff == 0)
            {
                if (enqueue_pos.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed)) break;
            }
            else if (diff < 0)
            {
                return false;  // Queue full
            }
        }

        slot->slot_state = slot_state;
        slot->sequence.store(static_cast<int64_t>(pos + 1), std::memory_order_release);
        return true;
    }

    // Batch push: reserve count slots with a single CAS after confirming
    // every target slot is available under the usual Vyukov sequence check.
    void push_batch(PTO2TaskSlotState **items, int count)
    {
        if (count == 0) return;

        uint64_t pos;
        while (true)
        {
            pos = enqueue_pos.load(std::memory_order_relaxed);
            bool ready = true;
            for (int i = 0; i < count; i++)
            {
                PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
                int64_t seq = slot->sequence.load(std::memory_order_acquire);
                int64_t diff = seq - static_cast<int64_t>(pos + i);
                if (diff != 0)
                {
                    ready = false;
                    break;
                }
            }
            if (!ready) continue;
            if (enqueue_pos.compare_exchange_weak(pos, pos + count, std::memory_order_relaxed, std::memory_order_relaxed)) break;
        }

        for (int i = 0; i < count; i++)
        {
            PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
            slot->slot_state = items[i];
            slot->sequence.store(static_cast<int64_t>(pos + i + 1), std::memory_order_release);
        }
    }

    PTO2TaskSlotState *pop()
    {
        // Fast-path: skip slot load when queue is clearly empty
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        if (d >= e) return nullptr;

        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
        while (true)
        {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - static_cast<int64_t>(pos + 1);
            if (diff == 0)
            {
                if (dequeue_pos.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed)) break;
            }
            else if (diff < 0)
            {
                return nullptr;  // Queue empty
            }
        }

        PTO2TaskSlotState *result = slot->slot_state;
        slot->sequence.store(static_cast<int64_t>(pos + mask + 1), std::memory_order_release);
        return result;
    }

    // Batch pop: reserve a contiguous run of ready slots with a single CAS.
    // Returns actual number of items popped (may be less than max_count).
    int pop_batch(PTO2TaskSlotState **out, int max_count)
    {
        uint64_t pos;
        int count;
        while (true)
        {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            count = 0;
            while (count < max_count)
            {
                PTO2ReadyQueueSlot *slot = &slots[(pos + count) & mask];
                int64_t seq = slot->sequence.load(std::memory_order_acquire);
                int64_t diff = seq - static_cast<int64_t>(pos + count + 1);
                if (diff == 0)
                {
                    count++;
                    continue;
                }
                if (diff < 0) break;
                count = -1;
                break;
            }
            if (count == 0) return 0;
            if (count < 0) continue;
            if (dequeue_pos.compare_exchange_weak(pos, pos + count, std::memory_order_relaxed, std::memory_order_relaxed)) break;
        }

        for (int i = 0; i < count; i++)
        {
            PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
            out[i] = slot->slot_state;
            slot->sequence.store(static_cast<int64_t>(pos + i + mask + 1), std::memory_order_release);
        }
        return count;
    }
};

inline size_t ready_queue_reserve_layout(DeviceArena &arena, uint64_t capacity)
{
    return arena.reserve(capacity * sizeof(PTO2ReadyQueueSlot), PTO2_ALIGN_SIZE);
}
inline bool ready_queue_init_data_from_layout(PTO2ReadyQueue *queue, DeviceArena &arena, size_t slots_off, uint64_t capacity)
{
    // Address the slots region for data writes without storing the pointer in
    // queue->slots — that field is set by ready_queue_wire_arena_pointers.
    auto *slots_arena = static_cast<PTO2ReadyQueueSlot *>(arena.region_ptr(slots_off));
    queue->capacity = capacity;
    queue->mask = capacity - 1;
    queue->enqueue_pos.store(0, std::memory_order_relaxed);
    queue->dequeue_pos.store(0, std::memory_order_relaxed);

    for (uint64_t i = 0; i < capacity; i++)
    {
        slots_arena[i].sequence.store((int64_t)i, std::memory_order_relaxed);
        slots_arena[i].slot_state = nullptr;
    }

    return true;
}
// Stores queue->slots = arena.region_ptr(slots_off). Idempotent.
inline void ready_queue_wire_arena_pointers(PTO2ReadyQueue *queue, DeviceArena &arena, size_t slots_off)
{
    queue->slots = static_cast<PTO2ReadyQueueSlot *>(arena.region_ptr(slots_off));
}
inline void ready_queue_destroy(PTO2ReadyQueue *queue)
{
    // Arena owns the slots[] buffer; just forget the pointer.
    queue->slots = nullptr;
}

struct alignas(64) PTO2SpscQueue
{
    // --- Producer cache lines (orchestrator thread) ---
    alignas(64) std::atomic<uint64_t> head_{0};
    alignas(64) uint64_t tail_cached_{0};

    // --- Consumer cache lines (scheduler thread 0) ---
    alignas(64) std::atomic<uint64_t> tail_{0};
    alignas(64) uint64_t head_cached_{0};

    // --- Shared Cacheline (read only) with mask and data ptr (immutable after init) ---
    alignas(64) PTO2TaskSlotState **buffer_{nullptr};
    uint64_t mask_{0};

    // Padding to exactly 5 cache lines
    char padding[64 - sizeof(PTO2TaskSlotState **) - sizeof(uint64_t)];

    static size_t reserve_layout(DeviceArena &arena, uint64_t capacity)
    {
        return arena.reserve(capacity * sizeof(PTO2TaskSlotState *), PTO2_ALIGN_SIZE);
    }

    bool init_data_from_layout(DeviceArena &arena, size_t buffer_off, uint64_t capacity)
    {
        if (capacity == 0 || (capacity & (capacity - 1)) != 0) return false;
        auto *buf = static_cast<PTO2TaskSlotState **>(arena.region_ptr(buffer_off));
        // calloc'd-equivalent: zero the slot pointers so spurious early pops
        // observe nullptr.
        for (uint64_t i = 0; i < capacity; i++) buf[i] = nullptr;
        mask_ = capacity - 1;
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
        tail_cached_ = 0;
        head_cached_ = 0;
        return true;
    }

    // Wire the arena-internal pointer. Called by both host (with host arena)
    // and AICPU (with device arena attached to the prebuilt image).
    void wire_arena_pointers(DeviceArena &arena, size_t buffer_off)
    {
        buffer_ = static_cast<PTO2TaskSlotState **>(arena.region_ptr(buffer_off));
    }

    // Arena owns the buffer; here we only forget our pointer.
    void destroy()
    {
        buffer_ = nullptr;
    }

    bool push(PTO2TaskSlotState *item)
    {
        uint64_t h = head_.load(std::memory_order_relaxed);
        uint64_t next_h = h + 1;
        if (next_h - tail_cached_ > mask_)
        {
            tail_cached_ = tail_.load(std::memory_order_acquire);
            if (next_h - tail_cached_ > mask_) return false;
        }
        buffer_[h & mask_] = item;
        head_.store(next_h, std::memory_order_release);
        return true;
    }

    // Pop up to max_count items (consumer only). Returns actual count.
    int pop_batch(PTO2TaskSlotState **out, int max_count)
    {
        uint64_t t = tail_.load(std::memory_order_relaxed);
        uint64_t avail = head_cached_ - t;
        if (avail < static_cast<uint64_t>(max_count))
        {
            head_cached_ = head_.load(std::memory_order_acquire);
            avail = head_cached_ - t;
            if (avail == 0) return 0;
        }
        int count = (avail < static_cast<uint64_t>(max_count)) ? static_cast<int>(avail) : max_count;
        for (int i = 0; i < count; i++) out[i] = buffer_[(t + i) & mask_];
        tail_.store(t + count, std::memory_order_release);
        return count;
    }

    // Approximate size (used for backoff decisions, not exact).
    uint64_t size() const
    {
        uint64_t h = head_.load(std::memory_order_acquire);
        uint64_t t = tail_.load(std::memory_order_acquire);
        return h - t;
    }
};

static_assert(sizeof(PTO2SpscQueue) == 5 * 64, "PTO2SpscQueue must be exactly 5 cache lines (320B)");
// =============================================================================

struct CompletionStats
{
    int32_t fanout_edges;       // Number of fanout edges traversed (notify consumers)
    int32_t tasks_enqueued;     // Number of consumers that became READY
    int32_t fanin_edges;        // Number of fanin edges traversed (release producers)
    bool mixed_task_completed;  // True only when this callback completed a mixed task
};

struct PTO2SchedulerLayout
{
    size_t off_ready_queue_slots[PTO2_NUM_RESOURCE_SHAPES];
    size_t off_dummy_ready_queue_slots;
    size_t off_dep_pool_entries[PTO2_MAX_RING_DEPTH];
    size_t off_wiring_spsc_buffer;
    uint64_t ready_queue_capacity;
    uint64_t spsc_capacity;
    int32_t dep_pool_capacity;
};

struct PTO2SchedulerState
{
    // Shared memory access
    PTO2SharedMemoryHeader *sm_header;

    // Per-ring state
    struct alignas(64) RingSchedState
    {
        // --- Cache Line 0: ring pointer (read-only) + hot path (read-write) ---
        PTO2SharedMemoryRingHeader *ring;
        int32_t last_task_alive;
        std::atomic<int32_t> advance_lock;  // multi-thread CAS

        // --- Cache Line 1+: Thread 0 only (wiring dep_pool) ---
        alignas(64) PTO2DepListPool dep_pool;

        bool init_data_from_layout(void *sm_dev_base, int32_t ring_id)
        {
            ring = pto2_sm_layout::ring_header_addr(sm_dev_base, ring_id);
            last_task_alive = 0;
            advance_lock.store(0, std::memory_order_relaxed);
            return true;
        }

        void destroy() { ring = nullptr; }

        void sync_to_sm()
        {
            ring->fc.last_task_alive.store(last_task_alive, std::memory_order_release);
        }

        void advance_ring_pointers()
        {
            int32_t current_task_index = ring->fc.current_task_index.load(std::memory_order_acquire);
            int32_t old_last_task_alive = last_task_alive;

            while (last_task_alive < current_task_index)
            {
                PTO2TaskSlotState &slot_state = ring->get_slot_state_by_task_id(last_task_alive);
                if (slot_state.task_state.load(std::memory_order_acquire) != PTO2_TASK_CONSUMED) break;
                last_task_alive++;
            }

            for (int32_t id = old_last_task_alive; id < last_task_alive; id++) ring->get_slot_state_by_task_id(id).reset_for_reuse();

            sync_to_sm();
        }
    } ring_sched_states[PTO2_MAX_RING_DEPTH];

    // Ready queues remain global (scheduling is ring-agnostic)
    PTO2ReadyQueue ready_queues[PTO2_NUM_RESOURCE_SHAPES];

    // Dependency-only tasks (active_mask is empty, shape == DUMMY). Drained by
    // the dispatch loop and completed inline -- never goes to AICore.
    PTO2ReadyQueue dummy_ready_queue;

    struct alignas(64) WiringState
    {
        static constexpr uint64_t BATCH_SIZE = 30;
        static constexpr int BACKOFF_LIMIT = 32;

        // --- Thread 0 exclusive: local batch buffer + backoff ---
        int batch_count = 0;
        int batch_index = 0;
        int backoff_counter = 0;
        PTO2TaskSlotState *batch[BATCH_SIZE];

        // --- SPSC queue: orchestrator (push) ↔ thread 0 (pop) ---
        PTO2SpscQueue queue;

        // --- Orchestrator write, thread 0 read ---
        alignas(64) std::atomic<bool> orch_needs_drain{false};
    } wiring;

    static_assert(offsetof(WiringState, queue) == 256, "WiringState: batch region must be exactly 4 cache lines before queue");
    static_assert(sizeof(WiringState) == 640, "WiringState must be exactly 10 cache lines (640B)");

    alignas(64) AsyncWaitList async_wait_list;

    int drain_wiring_queue(bool force_drain = false)
    {
        int wired = 0;

        // Refill local batch buffer when exhausted.
        if (wiring.batch_index >= wiring.batch_count)
        {
            // Backoff: defer pop when queue holds fewer than a full batch,
            // unless force_drain, orch_needs_drain, or backoff limit reached.
            if (!force_drain && wiring.queue.size() < WiringState::BATCH_SIZE)
            {
                if (!wiring.orch_needs_drain.load(std::memory_order_acquire) && wiring.backoff_counter < WiringState::BACKOFF_LIMIT)
                {
                    wiring.backoff_counter++;
                    return 0;
                }
            }
            wiring.backoff_counter = 0;
            wiring.batch_count = wiring.queue.pop_batch(wiring.batch, WiringState::BATCH_SIZE);
            wiring.batch_index = 0;
            if (wiring.batch_count == 0) return 0;
        }

        // Process tasks from local buffer in strict FIFO order.
        while (wiring.batch_index < wiring.batch_count)
        {
            PTO2TaskSlotState *ws = wiring.batch[wiring.batch_index];
            int ring_id = ws->ring_id;
            auto &rss = ring_sched_states[ring_id];
            int32_t wfanin = ws->payload->fanin_actual_count;

            if (wfanin > 0 && rss.dep_pool.available() < wfanin)
            {
                rss.dep_pool.reclaim(*rss.ring, rss.last_task_alive);
                if (rss.dep_pool.available() < wfanin) break;  // not enough dep_pool space — keep remainder for next call
            }

            wiring.batch_index++;
            wire_task(rss, ws, wfanin);
            wired++;
        }

        return wired;
    }

    void push_ready_routed(PTO2TaskSlotState *slot_state)
    {
        PTO2ResourceShape shape = slot_state->active_mask.to_shape();
        if (shape == PTO2ResourceShape::DUMMY) dummy_ready_queue.push(slot_state);
        else ready_queues[static_cast<int32_t>(shape)].push(slot_state);
    }

    void wire_task(RingSchedState &rss, PTO2TaskSlotState *ws, int32_t wfanin)
    {
        PTO2TaskPayload *wp = ws->payload;
        ws->fanin_count = wfanin + 1;

        if (wfanin != 0)
        {
            int32_t early_finished = 0;
            for_each_fanin_slot_state(*wp, [&](PTO2TaskSlotState *producer) {
                producer->lock_fanout();
                int32_t pstate = producer->task_state.load(std::memory_order_acquire);
                if (pstate >= PTO2_TASK_COMPLETED) early_finished++;
                else producer->fanout_head = rss.dep_pool.prepend(producer->fanout_head, ws);
                producer->unlock_fanout();
            });

            int32_t init_rc = early_finished + 1;
            int32_t new_rc = ws->fanin_refcount.fetch_add(init_rc, std::memory_order_acq_rel) + init_rc;
            if (new_rc >= ws->fanin_count) push_ready_routed(ws);
        }
        else
        {
            ws->fanin_refcount.fetch_add(1, std::memory_order_acq_rel);
            push_ready_routed(ws);
        }

        ws->dep_pool_mark = rss.dep_pool.top;
    }

    void check_and_handle_consumed(PTO2TaskSlotState &slot_state)
    {
        if (slot_state.fanout_refcount.load(std::memory_order_acquire) != slot_state.fanout_count) return;

        PTO2TaskState expected = PTO2_TASK_COMPLETED;
        if (!slot_state.task_state.compare_exchange_strong(expected, PTO2_TASK_CONSUMED, std::memory_order_acq_rel, std::memory_order_acquire)) return;

        int32_t ring_id = slot_state.ring_id;
        // Try-lock — if another thread is advancing this ring, it will scan our CONSUMED task
        int32_t expected_lock = 0;
        if (ring_sched_states[ring_id].advance_lock.compare_exchange_strong(expected_lock, 1, std::memory_order_acquire, std::memory_order_relaxed))
        {
            ring_sched_states[ring_id].advance_ring_pointers();
            ring_sched_states[ring_id].advance_lock.store(0, std::memory_order_release);
        }
    }

    void release_producer(PTO2TaskSlotState &slot_state)
    {
        slot_state.fanout_refcount.fetch_add(1, std::memory_order_acq_rel);
        check_and_handle_consumed(slot_state);
    }

    bool release_fanin_and_check_ready(PTO2TaskSlotState &slot_state, PTO2LocalReadyBuffer *local_bufs = nullptr)
    {
        int32_t new_refcount = slot_state.fanin_refcount.fetch_add(1, std::memory_order_acq_rel) + 1;

        if (new_refcount == slot_state.fanin_count)
        {
            PTO2ResourceShape shape = slot_state.active_mask.to_shape();
            if (shape == PTO2ResourceShape::DUMMY) dummy_ready_queue.push(&slot_state);
            else if (!local_bufs || !local_bufs[static_cast<int32_t>(shape)].try_push(&slot_state)) ready_queues[static_cast<int32_t>(shape)].push(&slot_state);
            return true;
        }
        return false;
    }

    int get_ready_tasks_batch(PTO2ResourceShape shape, PTO2LocalReadyBuffer &local_buf, PTO2TaskSlotState **out, int max_count)
    {
        int count = 0;
        while (count < max_count && local_buf.count > 0) out[count++] = local_buf.slot_states[--local_buf.count];
        int remaining = max_count - count;
        if (remaining > 0) count += ready_queues[static_cast<int32_t>(shape)].pop_batch(out + count, remaining);
        return count;
    }

    void on_scope_end(PTO2TaskSlotState **task_slot_states, int32_t count)
    {
        if (count > 0) __builtin_prefetch(task_slot_states[0], 1, 0);
        for (int32_t i = 0; i < count; i++)
        {
            if (i + 1 < count) __builtin_prefetch(task_slot_states[i + 1], 1, 0);
            release_producer(*task_slot_states[i]);
        }
    }

    bool on_subtask_complete(PTO2TaskSlotState &slot_state)
    {
        int16_t prev = slot_state.completed_subtasks.fetch_add(1, std::memory_order_acq_rel);
        return (prev + 1) == slot_state.total_required_subtasks;
    }

    void on_mixed_task_complete(
        PTO2TaskSlotState &slot_state,

        PTO2LocalReadyBuffer *local_bufs = nullptr
    )
    {
        slot_state.lock_fanout();
        slot_state.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);
        PTO2DepListEntry *current = slot_state.fanout_head;  // Protected by fanout_lock
        slot_state.unlock_fanout();

        // Fanout: notify consumers
        while (current != nullptr)
        {
            PTO2TaskSlotState &consumer_slot = *current->slot_state;
            release_fanin_and_check_ready(consumer_slot, local_bufs);
            current = current->next;
        }
    }

    int32_t on_task_release(PTO2TaskSlotState &slot_state)
    {
        PTO2TaskPayload *payload = slot_state.payload;
        for_each_fanin_slot_state(*payload, [&](PTO2TaskSlotState *producer_slot_state) {
            release_producer(*producer_slot_state);
        });

        // Self consumed check
        check_and_handle_consumed(slot_state);
        return payload->fanin_actual_count;
    }

    // === Cold-path API ===

    static PTO2SchedulerLayout reserve_layout(DeviceArena &arena, int32_t dep_pool_capacity)
    {
        PTO2SchedulerLayout layout{};
        layout.ready_queue_capacity = PTO2_READY_QUEUE_SIZE;
        layout.spsc_capacity = PTO2_WRIRING_QUEUE_SIZE;
        layout.dep_pool_capacity = dep_pool_capacity;

        for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) layout.off_ready_queue_slots[i] = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
        layout.off_dummy_ready_queue_slots = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) layout.off_dep_pool_entries[r] = arena.reserve(static_cast<size_t>(dep_pool_capacity) * sizeof(PTO2DepListEntry), PTO2_ALIGN_SIZE);
        layout.off_wiring_spsc_buffer = PTO2SpscQueue::reserve_layout(arena, PTO2_WRIRING_QUEUE_SIZE);
        return layout;
    }

    bool init_data_from_layout(const PTO2SchedulerLayout &layout, DeviceArena &arena, void *sm_dev_base)
    {
        PTO2SchedulerState *sched = this;
        sched->sm_header = reinterpret_cast<PTO2SharedMemoryHeader *>(sm_dev_base);

        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
            if (!sched->ring_sched_states[r].init_data_from_layout(sm_dev_base, r)) return false;

        for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++)
            if (!ready_queue_init_data_from_layout(&sched->ready_queues[i], arena, layout.off_ready_queue_slots[i], layout.ready_queue_capacity)) return false;
        if (!ready_queue_init_data_from_layout(&sched->dummy_ready_queue, arena, layout.off_dummy_ready_queue_slots, layout.ready_queue_capacity)) return false;

        auto *orch_err = pto2_sm_layout::orch_error_code_addr(sm_dev_base);
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
        {
            auto *dep_entries = static_cast<PTO2DepListEntry *>(arena.region_ptr(layout.off_dep_pool_entries[r]));
            memset(dep_entries, 0, static_cast<size_t>(layout.dep_pool_capacity) * sizeof(PTO2DepListEntry));
            sched->ring_sched_states[r].dep_pool.init(dep_entries, layout.dep_pool_capacity, orch_err);
        }

        if (!sched->wiring.queue.init_data_from_layout(arena, layout.off_wiring_spsc_buffer, layout.spsc_capacity)) return false;
        sched->wiring.batch_count = 0;
        sched->wiring.batch_index = 0;
        sched->wiring.backoff_counter = 0;

        return true;
    }

    void wire_arena_pointers(const PTO2SchedulerLayout &layout, DeviceArena &arena)
    {
        PTO2SchedulerState *sched = this;
        for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) ready_queue_wire_arena_pointers(&sched->ready_queues[i], arena, layout.off_ready_queue_slots[i]);
        ready_queue_wire_arena_pointers(&sched->dummy_ready_queue, arena, layout.off_dummy_ready_queue_slots);
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) sched->ring_sched_states[r].dep_pool.base = static_cast<PTO2DepListEntry *>(arena.region_ptr(layout.off_dep_pool_entries[r]));
        sched->wiring.queue.wire_arena_pointers(arena, layout.off_wiring_spsc_buffer);
    }

    // Forget per-region pointers; arena owns the backing memory.
    void destroy()
    {
        PTO2SchedulerState *sched = this;
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
        {
            sched->ring_sched_states[r].destroy();
            sched->ring_sched_states[r].dep_pool.base = nullptr;
        }
        sched->wiring.queue.destroy();
        for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) ready_queue_destroy(&sched->ready_queues[i]);
        ready_queue_destroy(&sched->dummy_ready_queue);
    }
    void print_stats()
    {}
    void print_queues()
    {}
};

// Scheduler cold-path API is declared as PTO2SchedulerState member functions.
// See init()/destroy()/print_stats()/print_queues() below the struct definition.

inline bool AsyncWaitList::try_inline_complete_locked(AsyncWaitList::DrainCompletionSink &sink, PTO2TaskSlotState &slot_state)
{
    sink.sched->on_mixed_task_complete(slot_state, sink.local_bufs);
    if (*sink.deferred_release_count >= sink.deferred_release_capacity)
        while (*sink.deferred_release_count > 0) sink.sched->on_task_release(*sink.deferred_release_slot_states[--(*sink.deferred_release_count)]);
    sink.deferred_release_slot_states[(*sink.deferred_release_count)++] = &slot_state;
    sink.inline_completed++;
    return true;
}

template <bool Profiling>
inline AsyncPollResult AsyncWaitList::poll_and_complete(AICoreCompletionMailbox *aicore_mailbox, PTO2SchedulerState *sched, PTO2LocalReadyBuffer *local_bufs, PTO2TaskSlotState **deferred_release_slot_states, int32_t &deferred_release_count, int32_t deferred_release_capacity)
{
    AsyncPollResult result;
    if (!try_lock()) return result;

    AsyncWaitList::DrainCompletionSink sink{};
    sink.sched = sched;
    sink.local_bufs = local_bufs;
    sink.deferred_release_slot_states = deferred_release_slot_states;
    sink.deferred_release_count = &deferred_release_count;
    sink.deferred_release_capacity = deferred_release_capacity;

    int32_t drain_err = PTO2_ERROR_NONE;
    drain_aicore_completion_mailbox_locked(aicore_mailbox, sink, drain_err);
    if (drain_err != PTO2_ERROR_NONE)
    {
        result.error_code = drain_err;
        unlock();
        return result;
    }
    result.completed += sink.inline_completed;

    for (int32_t i = count - 1; i >= 0; --i)
    {
        AsyncWaitEntry &entry = entries[i];
        uintptr_t last_invalidated_counter_line = static_cast<uintptr_t>(-1);
        for (int32_t c = 0; c < entry.condition_count; c++)
        {
            CompletionCondition &cond = entry.conditions[c];
            if (cond.satisfied) continue;
            if (cond.completion_type == COMPLETION_TYPE_COUNTER && cond.counter_addr != nullptr)
            {
                uintptr_t counter_line = mailbox_cache_line(cond.counter_addr);
                if (counter_line != last_invalidated_counter_line)
                {
                    cache_invalidate_range(reinterpret_cast<const void *>(counter_line), sizeof(uint32_t));
                    last_invalidated_counter_line = counter_line;
                }
            }
            CompletionPollResult poll = cond.test();
            if (poll.state == CompletionPollState::FAILED)
            {
                result.error_code = poll.error_code;
                result.failed_slot_state = entry.slot_state;
                unlock();
                return result;
            }
            if (poll.state == CompletionPollState::READY)
            {
                cond.satisfied = true;
                cond.retire();
                entry.waiting_completion_count--;
            }
        }

        if (entry.normal_done && entry.waiting_completion_count <= 0)
        {
            sched->on_mixed_task_complete(*entry.slot_state, local_bufs);
            if (deferred_release_count >= deferred_release_capacity)
                while (deferred_release_count > 0) sched->on_task_release(*deferred_release_slot_states[--deferred_release_count]);
            deferred_release_slot_states[deferred_release_count++] = entry.slot_state;
            result.completed++;

            int32_t last = count - 1;
            if (i != last) entries[i] = entries[last];
            count = last;
        }
    }

    unlock();
    return result;
}
