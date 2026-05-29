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
 * Host/AICPU shared runtime-arena layout, init_data and wire implementations.
 *
 * Lives under runtime/shared/ so it is included in both the host_runtime.so
 * build (host pre-populates the prebuilt arena image) and the aicpu_runtime
 * build (AICPU runs wire_arena_pointers + destroy after attach). The
 * device-only parts of pto_runtime2.cpp / pto_orchestrator.cpp / pto_scheduler.cpp
 * (ops table, scope/submit/dispatch business logic, profiling) stay in their
 * original files and the aicpu build only.
 */

#include <stdlib.h>
#include <string.h>

#include "pto_orchestrator.h"
#include "pto_runtime2.h"
#include "pto_ring_buffer.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "scheduler/pto_scheduler.h"


// =============================================================================
// Top-level runtime arena
// =============================================================================

PTO2RuntimeArenaLayout
runtime_reserve_layout(DeviceArena &arena, uint64_t task_window_size, int32_t dep_pool_capacity) {
    PTO2RuntimeArenaLayout layout{};
    layout.task_window_size = task_window_size;
    layout.dep_pool_capacity = dep_pool_capacity;

    int32_t task_window_sizes[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        task_window_sizes[r] = static_cast<int32_t>(task_window_size);
    }

    layout.off_sm_handle = arena.reserve(sizeof(PTO2SharedMemoryHandle), alignof(PTO2SharedMemoryHandle));
    layout.orch = PTO2OrchestratorState::reserve_layout(arena, task_window_sizes, dep_pool_capacity);
    layout.sched = PTO2SchedulerState::reserve_layout(arena, dep_pool_capacity);
    layout.off_runtime = arena.reserve(sizeof(PTO2Runtime), PTO2_ALIGN_SIZE);
    layout.off_mailbox = arena.reserve(sizeof(AICoreCompletionMailbox), alignof(AICoreCompletionMailbox));

    layout.arena_size = arena.total_size();
    return layout;
}

PTO2Runtime *runtime_init_data_from_layout(
    DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2RuntimeMode mode, void *sm_dev_base,
    uint64_t /*sm_size*/, void *gm_heap_dev_base, uint64_t heap_size
) {
    PTO2Runtime *rt = static_cast<PTO2Runtime *>(arena.region_ptr(layout.off_runtime));
    memset((void*)rt, 0, sizeof(PTO2Runtime));

    auto *sm_wrap = static_cast<PTO2SharedMemoryHandle *>(arena.region_ptr(layout.off_sm_handle));
    memset(sm_wrap, 0, sizeof(*sm_wrap));

    // rt->ops is filled by the AICPU at boot.
    rt->mode = mode;
    rt->gm_heap = gm_heap_dev_base;
    rt->gm_heap_size = heap_size > 0 ? heap_size * PTO2_MAX_RING_DEPTH : 0;
    rt->gm_heap_owned = false;
    rt->total_cycles = 0;

    if (!rt->orchestrator.init_data_from_layout(
            layout.orch, arena, sm_dev_base, gm_heap_dev_base, heap_size, layout.task_window_size
        )) {
        return nullptr;
    }
    if (!rt->scheduler.init_data_from_layout(layout.sched, arena, sm_dev_base)) {
        return nullptr;
    }

    auto *mailbox = static_cast<AICoreCompletionMailbox *>(arena.region_ptr(layout.off_mailbox));
    memset(mailbox, 0, sizeof(*mailbox));

    return rt;
}

void runtime_wire_arena_pointers(DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2Runtime *rt) {
    rt->sm_handle = static_cast<PTO2SharedMemoryHandle *>(arena.region_ptr(layout.off_sm_handle));
    rt->aicore_mailbox = static_cast<AICoreCompletionMailbox *>(arena.region_ptr(layout.off_mailbox));
    rt->orchestrator.wire_arena_pointers(layout.orch, arena, &rt->scheduler);
    rt->scheduler.wire_arena_pointers(layout.sched, arena);
}

void runtime_destroy(PTO2Runtime *rt, DeviceArena & /*arena*/) {
    // Arena buffer is pooled across runs by DeviceRunner — never freed here.
    if (!rt) return;
    rt->scheduler.destroy();
    rt->orchestrator.destroy();
    rt->aicore_mailbox = nullptr;
    rt->sm_handle = nullptr;
}
