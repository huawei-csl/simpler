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
 * TraCR instrumentation surface for simpler's AICPU-side runtime.
 *
 * <tracr/tracr.hpp> lives in the optional tools/tracr submodule, so it is
 * included only when ENABLE_TRACR is defined. Without the define this header
 * supplies no-op macros under the same names, and a build needs neither the
 * submodule nor BUILD_TRACR.
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstring>
#include <string_view>

#ifdef ENABLE_TRACR
#include <tracr/tracr.hpp>
#else
#define INSTRUMENTATION_ACTIVE false
#define INSTRUMENTATION_START()
#define INSTRUMENTATION_END()
#define INSTRUMENTATION_THREAD_INIT()
#define INSTRUMENTATION_THREAD_FINALIZE()
#define INSTRUMENTATION_MARK_SET(channelId, eventId, extraId) ((void)(channelId), (void)(eventId), (void)(extraId))
#define INSTRUMENTATION_MARK_RESET(channelId) ((void)(channelId))
#endif

// sched_getcpu() is glibc/Linux-only, and the sim host runtime also builds on
// macOS; there the caller is told "no CPU" rather than failing to compile.
#if defined(__linux__)
#include <sched.h>
inline int tracr_getcpu() { return sched_getcpu(); }
#else
inline int tracr_getcpu() { return -1; }
#endif

// Lane index of the calling AICPU thread, -1 outside a traced run. Lanes are
// handed out in ascending order from 0 so they index the per-thread device
// trace buffer directly.
inline thread_local int g_tracr_thread_idx{-1};
inline std::atomic<int> g_tracr_thread_counter{0};

/*
 * Scheduler-loop markers carry each runtime's own SchedPhaseKind vocabulary
 * (src/common/{host_build_graph,tensormap_and_ringbuffer}/sched_phase_kind.h),
 * so a TraCR lane and a chip-swimlane bar covering the same code call it the
 * same thing. Every name here is published in a capture's metadata.json, so a
 * name that no runtime emits advertises a phase that cannot occur.
 */
#define MARKER_TYPES       \
    X(Orchestrating)       \
    X(Read_Dimensions)     \
    X(Reshape_Kernels)     \
    X(Pre_Loop_Info)       \
    X(Scoping)             \
    X(Scheduling)          \
    X(Complete)            \
    X(Drain_Sync_Start)    \
    X(Dummy)               \
    X(Dispatch)            \
    X(Idle)                \
    X(Initializing)        \
    X(De_Initializing)     \
    X(DLL_loading)         \
    X(Allocating)          \
    X(Running_Task_Single) \
    X(Running_Task_Pair)   \
    X(Resolving)           \
    X(Release)

enum MarkerType {
#define X(name) name,
    MARKER_TYPES
#undef X

        MARKERTYPE_COUNT
};

constexpr std::string_view MarkerTypeNames[] = {
#define X(name) #name,
    MARKER_TYPES
#undef X
};

/**
 * Claim this thread's TraCR lane. The thread that draws lane 0 starts the
 * process-wide capture; the others attach to it.
 */
inline void tracr_thread_start() {
#ifdef ENABLE_TRACR
    g_tracr_thread_idx = g_tracr_thread_counter.fetch_add(1, std::memory_order_relaxed);
    if (g_tracr_thread_idx == 0) {
        INSTRUMENTATION_START();
    } else {
        INSTRUMENTATION_THREAD_INIT();
    }
#endif
}

/**
 * Copy this thread's traces into the device buffer the host sized for
 * aicpu_thread_num lanes, then release the lane. Returns false when this
 * thread's lane falls outside that buffer, the one case where traces are
 * dropped rather than stored.
 */
template <typename RuntimeT>
inline bool tracr_thread_finalize(RuntimeT *runtime) {
    (void)runtime;
#ifdef ENABLE_TRACR
    const bool stored = (g_tracr_thread_idx >= 0 && g_tracr_thread_idx < runtime->get_aicpu_thread_num());
    if (stored && runtime->get_tracr_data() != 0 && tracrThread->_traceIdx > 0) {
        auto *data = reinterpret_cast<TraCR::Payload *>(runtime->get_tracr_data());
        std::memcpy(
            &data[g_tracr_thread_idx * TraCR::CAPACITY], tracrThread->_traces.data(),
            tracrThread->_traceIdx * sizeof(TraCR::Payload)
        );
    }
    if (stored && runtime->get_tracr_data_sizes() != 0) {
        reinterpret_cast<size_t *>(runtime->get_tracr_data_sizes())[g_tracr_thread_idx] = tracrThread->_traceIdx;
    }

    if (g_tracr_thread_idx == 0) {
        INSTRUMENTATION_END();
        g_tracr_thread_counter.store(0, std::memory_order_relaxed);
    } else {
        INSTRUMENTATION_THREAD_FINALIZE();
    }
    g_tracr_thread_idx = -1;
    return stored;
#else
    return true;
#endif
}
