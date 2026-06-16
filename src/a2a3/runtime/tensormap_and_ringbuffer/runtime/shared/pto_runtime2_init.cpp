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

#include <stdlib.h>
#include <string.h>

#include "pto_orchestrator.h"
#include "pto_runtime2.h"
#include "pto_ring_buffer.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "scheduler/pto_scheduler.h"

bool PTO2SchedulerState::RingSchedState::init_data_from_layout(void *sm_dev_base, int32_t ring_id)
{
    // ring stores the device address of the SM ring header — pure offset
    // arithmetic, no SM load.
    ring = pto2_sm_layout::ring_header_addr(sm_dev_base, ring_id);
    last_task_alive = 0;
    advance_lock.store(0, std::memory_order_relaxed);

    return true;
}

void PTO2SchedulerState::RingSchedState::destroy()
{
    ring = nullptr;
}
