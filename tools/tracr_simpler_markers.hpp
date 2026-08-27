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

// Marker ids are POSITIONAL (this X-macro generates both the enum and the name
// table), so editing this list renumbers events in the .bts payloads.
//
// That is safe for the current consumers, which was checked before editing:
// every call site names markers via the macro (so they renumber with the enum),
// and the trace analysers key only on RESET == 0xFFFF, treating any other event
// on a core lane as opening a task span -- none of them hardcode a SET id.
// Re-check that before renumbering again.
//
// Removed as dead in every runtime (zero call sites in hbg, tmr, sac, platform
// or common): Read_Dimensions, Reshape_Kernels, Pre_Loop_Info, PTO2_SCOPE_.
//
// Which runtime uses what, so an unused-looking marker is not deleted by
// mistake: Orchestrating / Phase3 / DLL_loading / Allocating / Barrier are
// tensormap_and_ringbuffer's (it orchestrates on device); Resolving is
// host_build_graph's dedicated resolution thread (3S+1P). scan_and_claim uses
// neither -- it has no device orchestrator and no P thread.
#define MARKER_TYPES       \
    X(Orchestrating)       \
    X(Scheduling)          \
    X(Phase1)              \
    X(Phase2)              \
    X(Phase3)              \
    X(Phase4)              \
    X(Drain)               \
    X(Initializing)        \
    X(De_Initializing)     \
    X(DLL_loading)         \
    X(Allocating)          \
    X(Running_Task_Single) \
    X(Running_Task_Pair)   \
    X(Barrier)             \
    X(Resolving)           \
    /* --- scan_and_claim --- appended, so existing ids stay put --- */ \
    X(Scanning)            \
    X(Retiring_Inline)     \
    X(Idle)

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
