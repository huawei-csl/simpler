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

// X-Macro approach. Pretty cool
//
// A marker's position IS its wire id: `payload.eventId` is the 0-based
// MarkerType value, and AICore-written lanes hard-code those ids as integers
// because the emitter cannot see this enum (see
// `src/common/platform/include/aicore/tracr_aicore_emit.h`). So removing or
// reordering an entry silently renames every lane after it. Append new markers
// at the end; do not reorder. Host-side callers use the names and are immune.
#define MARKER_TYPES       \
    X(Orchestrating)       \
    X(Read_Dimensions)     \
    X(Reshape_Kernels)     \
    X(Pre_Loop_Info)       \
    X(PTO2_SCOPE_)         \
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
    X(CommNotify)          \
    X(CommWait)

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
