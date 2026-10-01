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
 * TraCR Simpler Marker Types
 */

#pragma once

#include <atomic>
#include <string_view>

// sched_getcpu() is a glibc/Linux-only API, but the simulator/host build also
// compiles on non-Linux targets (e.g. the macOS packaging CI). Route the TraCR
// call sites through this portable shim instead of calling sched_getcpu directly.
#if defined(__linux__)
#include <sched.h>
inline int tracr_getcpu() { return sched_getcpu(); }
#else
inline int tracr_getcpu() { return -1; }
#endif

// Global TraCR thread idx counter
inline std::atomic<int> g_TraCR_thread_idx_counter{0};

// Global thread local thread idx placeholder
inline thread_local int g_TraCR_thread_idx{-1};

/*
 * Scheduler-loop markers are named after each runtime's own SchedPhaseKind
 * vocabulary (src/common/{host_build_graph,tensormap_and_ringbuffer}/sched_phase_kind.h),
 * so a TraCR lane and a chip-swimlane bar covering the same code call it the
 * same thing.
 *
 * These replaced Phase1..Phase4, which named a position in the loop rather than
 * the work it does. Two of the old names were actively misleading:
 *
 *   old name  new name           what it actually wraps
 *   --------  -----------------  ----------------------------------------------
 *   Phase1    Complete           Observe FINs and run completion work. In
 *                                group_queue on a simulated device this is one
 *                                watermark read retiring a contiguous prefix
 *                                rather than a per-core MMIO sweep -- same role
 *                                in the loop, different mechanism.
 *   Phase2    Drain_Sync_Start   The sync_start stop-the-world drain attempt
 *                                (SchedPhaseKind::Drain). This, not the marker
 *                                that used to be called Drain.
 *   Phase3    Dummy              Dependency-only dummy / false-predicate
 *                                retirement. Emitted by tensormap_and_ringbuffer;
 *                                host_build_graph does this on its resolution
 *                                thread instead.
 *   Phase4    Dispatch           Publish ready tasks to AICore
 *                                (dispatch_ready_tasks).
 *   Drain     Idle               host_build_graph / group_queue: a loop
 *                                iteration that made no progress.
 *   Drain     Release            tensormap_and_ringbuffer: the deferred-release
 *                                drain (SchedPhaseKind::Release).
 *
 * The last two are why Drain had to go: one marker name stood for two unrelated
 * phases depending on which runtime emitted it, so a trace could not be read
 * without knowing which binary produced it.
 *
 * The group_queue runtime carries one more (Group_Feed) on the asim-tracr
 * branch. It is deliberately absent here: this branch has no group_queue, and
 * every name in this list is published in each capture's metadata.json, so
 * carrying it would advertise a phase that can never occur in these traces.
 */
#define MARKER_TYPES       \
    X(Orchestrating)       \
    X(Read_Dimensions)     \
    X(Reshape_Kernels)     \
    X(Pre_Loop_Info)       \
    X(PTO2_SCOPE_)         \
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
