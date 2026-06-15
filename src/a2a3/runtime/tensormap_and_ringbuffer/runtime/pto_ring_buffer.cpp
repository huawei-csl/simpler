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

#include "pto_ring_buffer.h"
#include <inttypes.h>
#include <string.h>
#include "scheduler/pto_scheduler.h"

static void latch_pool_error(std::atomic<int32_t> *error_code_ptr, int32_t error_code)
{
    if (error_code_ptr == nullptr) return;
    int32_t expected = PTO2_ERROR_NONE;
    error_code_ptr->compare_exchange_strong(expected, error_code, std::memory_order_acq_rel);
}

void PTO2FaninPool::reclaim(PTO2SharedMemoryRingHeader &ring, int32_t sm_last_task_alive)
{
    if (sm_last_task_alive <= reclaim_task_cursor) return;

    int32_t scan_end = sm_last_task_alive;
    for (int32_t task_id = reclaim_task_cursor; task_id < scan_end; ++task_id)
    {
        PTO2TaskPayload &payload = ring.get_payload_by_task_id(task_id);
        if (payload.fanin_spill_pool != this) continue;

        int32_t inline_count = std::min(payload.fanin_actual_count, PTO2_FANIN_INLINE_CAP);
        int32_t spill_edge_count = payload.fanin_actual_count - inline_count;
        if (spill_edge_count > 0) advance_tail(payload.fanin_spill_start + spill_edge_count);
    }
    reclaim_task_cursor = scan_end;
}

bool PTO2FaninPool::ensure_space(PTO2SharedMemoryRingHeader &ring, int32_t needed)
{
    if (available() >= needed) return true;

    int spin_count = 0;
    int32_t prev_last_alive = ring.fc.last_task_alive.load(std::memory_order_acquire);
    while (available() < needed)
    {
        reclaim(ring, prev_last_alive);
        if (available() >= needed) return true;

        spin_count++;

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

void PTO2DepListPool::reclaim(PTO2SharedMemoryRingHeader &ring, int32_t sm_last_task_alive)
{
    if (sm_last_task_alive >= last_reclaimed + PTO2_DEP_POOL_CLEANUP_INTERVAL && sm_last_task_alive > 0)
    {
        int32_t mark = ring.get_slot_state_by_task_id(sm_last_task_alive - 1).dep_pool_mark;
        if (mark > 0) advance_tail(mark);
        last_reclaimed = sm_last_task_alive;
    }
}

bool PTO2DepListPool::ensure_space(PTO2SharedMemoryRingHeader &ring, int32_t needed)
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
