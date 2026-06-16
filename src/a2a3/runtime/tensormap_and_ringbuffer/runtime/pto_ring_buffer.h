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

#ifndef PTO_RING_BUFFER_H
#define PTO_RING_BUFFER_H

#include <algorithm>
#include <inttypes.h>
#include <type_traits>

#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"

// Block notification interval (in spin counts)
#define PTO2_BLOCK_NOTIFY_INTERVAL 10000
// Alloc spin limit - after this, report deadlock and exit
#define PTO2_ALLOC_SPIN_LIMIT 100000

// Dep pool spin limit - if exceeded, dep pool capacity too small for workload
#define PTO2_DEP_POOL_SPIN_LIMIT 100000

inline void latch_pool_error(std::atomic<int32_t> *error_code_ptr, int32_t error_code)
{
    if (error_code_ptr == nullptr) return;
    int32_t expected = PTO2_ERROR_NONE;
    error_code_ptr->compare_exchange_strong(expected, error_code, std::memory_order_acq_rel);
}

class PTO2TaskAllocator
{
public:
    void init(PTO2TaskDescriptor *descriptors, int32_t window_size, std::atomic<int32_t> *current_index_ptr, std::atomic<int32_t> *last_alive_ptr, void *heap_base, uint64_t heap_size, std::atomic<int32_t> *error_code_ptr, int32_t initial_local_task_id = 0)
    {
        descriptors_ = descriptors;
        window_size_ = window_size;
        window_mask_ = window_size - 1;
        current_index_ptr_ = current_index_ptr;
        last_alive_ptr_ = last_alive_ptr;
        heap_base_ = heap_base;
        heap_size_ = heap_size;
        error_code_ptr_ = error_code_ptr;
        local_task_id_ = initial_local_task_id;
        heap_top_ = 0;
        heap_tail_ = 0;
        last_alive_seen_ = 0;
    }

    PTO2TaskAllocResult alloc(int32_t output_size)
    {
        uint64_t aligned_size = output_size > 0 ? PTO2_ALIGN_UP(static_cast<uint64_t>(output_size), PTO2_ALIGN_SIZE) : 0;

        int spin_count = 0;
        int32_t prev_last_alive = last_alive_ptr_->load(std::memory_order_acquire);
        int32_t last_alive = prev_last_alive;
        update_heap_tail(last_alive);
        bool blocked_on_heap = false;

        while (true)
        {
            // Check both resources; commit only if both available
            if (local_task_id_ - last_alive + 1 < window_size_)
            {
                void *heap_ptr = try_bump_heap(aligned_size);
                if (heap_ptr)
                {
                    int32_t task_id = commit_task();
                    return {task_id, task_id & window_mask_, heap_ptr, static_cast<char *>(heap_ptr) + aligned_size};
                }
                blocked_on_heap = true;
            }
            else
            {
                blocked_on_heap = false;
            }

            // Spin: wait for scheduler to advance last_task_alive
            spin_count++;
            last_alive = last_alive_ptr_->load(std::memory_order_acquire);
            update_heap_tail(last_alive);
            if (last_alive > prev_last_alive)
            {
                spin_count = 0;
                prev_last_alive = last_alive;
            }
            else
            {
                if (spin_count % PTO2_BLOCK_NOTIFY_INTERVAL == 0)
                {}
                if (spin_count >= PTO2_ALLOC_SPIN_LIMIT)
                {
                    report_deadlock(blocked_on_heap);
                    return {-1, -1, nullptr, nullptr};
                }
            }
            SPIN_WAIT_HINT();
        }
    }

    int32_t active_count() const
    {
        int32_t last_alive = last_alive_ptr_->load(std::memory_order_acquire);
        return local_task_id_ - last_alive;
    }

    // Task ring start/end: tail = oldest live task (last_task_alive), head =
    // next task id to allocate. head - tail == active_count().
    int32_t task_tail() const
    {
        return last_alive_ptr_->load(std::memory_order_acquire);
    }
    int32_t task_head() const
    {
        return local_task_id_;
    }

    int32_t window_size() const
    {
        return window_size_;
    }

    uint64_t heap_available() const
    {
        uint64_t tail = heap_tail_;
        if (heap_top_ >= tail)
        {
            uint64_t at_end = heap_size_ - heap_top_;
            uint64_t at_begin = tail;
            return at_end > at_begin ? at_end : at_begin;
        }
        return tail - heap_top_;
    }

    uint64_t heap_top() const
    {
        return heap_top_;
    }
    // Heap ring start: reclaim pointer (oldest byte still live). heap_top() is
    // the end (next allocation). heap_top - heap_tail == heap_used_bytes().
    uint64_t heap_tail() const
    {
        return heap_tail_;
    }
    uint64_t heap_capacity() const
    {
        return heap_size_;
    }
    uint64_t heap_used_bytes() const
    {
        if (heap_size_ == 0) return 0;
        return (heap_top_ + heap_size_ - heap_tail_) % heap_size_;
    }

private:
    // --- Task Ring ---
    PTO2TaskDescriptor *descriptors_ = nullptr;
    int32_t window_size_ = 0;
    int32_t window_mask_ = 0;
    std::atomic<int32_t> *current_index_ptr_ = nullptr;
    std::atomic<int32_t> *last_alive_ptr_ = nullptr;

    // --- Heap ---
    void *heap_base_ = nullptr;
    uint64_t heap_size_ = 0;

    // --- Local state (single-writer, no atomics needed) ---
    int32_t local_task_id_ = 0;    // Next task ID to allocate
    uint64_t heap_top_ = 0;        // Current heap allocation pointer
    uint64_t heap_tail_ = 0;       // Heap reclamation pointer (derived from consumed tasks)
    int32_t last_alive_seen_ = 0;  // last_task_alive at last heap_tail derivation

    // --- Shared ---
    std::atomic<int32_t> *error_code_ptr_ = nullptr;

    int32_t commit_task()
    {
        int32_t task_id = local_task_id_++;
        current_index_ptr_->store(local_task_id_, std::memory_order_release);
        return task_id;
    }

    void update_heap_tail(int32_t last_alive)
    {
        if (last_alive <= last_alive_seen_) return;
        last_alive_seen_ = last_alive;

        PTO2TaskDescriptor &desc = descriptors_[(last_alive - 1) & window_mask_];
        heap_tail_ = static_cast<uint64_t>(static_cast<char *>(desc.packed_buffer_end) - static_cast<char *>(heap_base_));
    }

    void *try_bump_heap(uint64_t alloc_size)
    {
        uint64_t top = heap_top_;
        if (alloc_size == 0) return static_cast<char *>(heap_base_) + top;
        uint64_t tail = heap_tail_;
        void *result;

        if (top >= tail)
        {
            uint64_t space_at_end = heap_size_ - top;
            if (space_at_end >= alloc_size)
            {
                result = static_cast<char *>(heap_base_) + top;
                heap_top_ = top + alloc_size;
            }
            else if (tail > alloc_size)
            {
                result = heap_base_;
                heap_top_ = alloc_size;
            }
            else
            {
                return nullptr;
            }
        }
        else if (tail - top > alloc_size)
        {
            result = static_cast<char *>(heap_base_) + top;
            heap_top_ = top + alloc_size;
        }
        else
        {
            return nullptr;
        }

        return result;
    }

    void report_deadlock(bool heap_blocked)
    {
        if (error_code_ptr_)
        {
            int32_t code = heap_blocked ? PTO2_ERROR_HEAP_RING_DEADLOCK : PTO2_ERROR_FLOW_CONTROL_DEADLOCK;
            error_code_ptr_->store(code, std::memory_order_release);
        }
    }
};

template <typename Fn>
using PTO2FaninCallbackResult = std::invoke_result_t<Fn &, PTO2TaskSlotState *>;

template <typename Fn>
using PTO2FaninForEachReturn = std::conditional_t<std::is_same_v<PTO2FaninCallbackResult<Fn>, void>, void, bool>;

template <typename Slots, typename Fn>
inline PTO2FaninForEachReturn<Fn> for_each_fanin_in(Slots &&slot_states, int32_t fanin_count, Fn &&fn)
{
    using FaninCallbackResult = PTO2FaninCallbackResult<Fn>;
    static_assert(std::is_same_v<FaninCallbackResult, void> || std::is_same_v<FaninCallbackResult, bool>, "fanin callback must return void or bool");

    if constexpr (std::is_void_v<FaninCallbackResult>)
    {
        for (int32_t i = 0; i < fanin_count; i++) fn(slot_states[i]);
    }
    else
    {
        for (int32_t i = 0; i < fanin_count; i++)
            if (!fn(slot_states[i])) return false;
        return true;
    }
}

template <typename Fn>
inline PTO2FaninForEachReturn<Fn> for_each_fanin_slot_state(const PTO2TaskPayload &payload, Fn &&fn)
{
    return for_each_fanin_in(payload.fanin_slot_states, payload.fanin_count, static_cast<Fn &&>(fn));
}

struct PTO2DepListPool
{
    PTO2DepListEntry *base;     // Pool base address
    int32_t capacity;           // Total number of entries
    int32_t top;                // Linear next-allocation counter (starts from 1)
    int32_t tail;               // Linear first-alive counter (entries before this are dead)
    int32_t high_water;         // Peak concurrent usage (top - tail)
    int32_t last_reclaimed{0};  // last_task_alive at last successful reclamation

    // Error code pointer for fatal error reporting (→ sm_header->orch_error_code)
    std::atomic<int32_t> *error_code_ptr = nullptr;

    void init(PTO2DepListEntry *in_base, int32_t in_capacity, std::atomic<int32_t> *in_error_code_ptr)
    {
        base = in_base;
        capacity = in_capacity;
        top = 1;   // Start from 1, 0 means NULL/empty
        tail = 1;  // Match initial top (no reclaimable entries yet)
        high_water = 0;
        last_reclaimed = 0;

        // Initialize entry 0 as NULL marker
        base[0].slot_state = nullptr;
        base[0].next = nullptr;

        error_code_ptr = in_error_code_ptr;
    }

    void reclaim(PTO2SharedMemoryRingHeader &ring, int32_t sm_last_task_alive)
    {
        if (sm_last_task_alive >= last_reclaimed + PTO2_DEP_POOL_CLEANUP_INTERVAL && sm_last_task_alive > 0)
        {
            int32_t mark = ring.get_slot_state_by_task_id(sm_last_task_alive - 1).dep_pool_mark;
            if (mark > 0) advance_tail(mark);
            last_reclaimed = sm_last_task_alive;
        }
    }

    bool ensure_space(PTO2SharedMemoryRingHeader &ring, int32_t needed)
    {
        if (available() >= needed) return true;

        int spin_count = 0;
        int32_t prev_last_alive = ring.fc.last_task_alive.load(std::memory_order_acquire);
        while (available() < needed)
        {
            reclaim(ring, prev_last_alive);
            if (available() >= needed) return true;

            spin_count++;

            // Progress detection: reset spin counter if last_task_alive advances
            int32_t cur_last_alive = ring.fc.last_task_alive.load(std::memory_order_acquire);
            if (cur_last_alive > prev_last_alive)
            {
                spin_count = 0;
                prev_last_alive = cur_last_alive;
            }

            if (spin_count >= PTO2_DEP_POOL_SPIN_LIMIT)
            {
                latch_pool_error(error_code_ptr, PTO2_ERROR_DEP_POOL_OVERFLOW);
                return false;
            }
            SPIN_WAIT_HINT();
        }
        return true;
    }

    PTO2DepListEntry *alloc()
    {
        int32_t used = top - tail;
        if (used >= capacity)
        {
            if (error_code_ptr) error_code_ptr->store(PTO2_ERROR_DEP_POOL_OVERFLOW, std::memory_order_release);
            return nullptr;
        }
        int32_t idx = top % capacity;
        top++;
        used++;
        if (used > high_water) high_water = used;
        return &base[idx];
    }

    void advance_tail(int32_t new_tail)
    {
        if (new_tail > tail) tail = new_tail;
    }

    PTO2DepListEntry *prepend(PTO2DepListEntry *cur, PTO2TaskSlotState *slot_state)
    {
        PTO2DepListEntry *new_entry = alloc();
        if (!new_entry) return nullptr;
        new_entry->slot_state = slot_state;
        new_entry->next = cur;
        return new_entry;
    }

    int32_t used() const
    {
        return top - tail;
    }

    int32_t available() const
    {
        return capacity - used();
    }
};

struct PTO2RingSet
{
    PTO2TaskAllocator task_allocator;
};

#endif  // PTO_RING_BUFFER_H
