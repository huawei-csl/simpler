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
 * A marker is a sticky state, not a span: a SET replaces whatever its lane held,
 * and that state runs until the next SET on the same lane. Nothing closes it, so
 * any region of the scheduler loop that sets no marker is billed to the region
 * before it -- across the iteration boundary included. The loop therefore sets
 * one at the head of every region, and the five Sched_* states partition an AICPU
 * lane's time exactly.
 *
 * A state names the role the AICPU is playing, not the function it is in, so one
 * state can be entered from two different code paths. That is what lets a lane be
 * read the same way in both runtimes, whose loops agree on role and differ on
 * mechanism:
 *
 *   Sched_Complete Finding what finished and retiring it: the core sweep, the
 *                  async wait list, and the dummy queue. host_build_graph sweeps
 *                  each running core's COND register with serialized
 *                  Device-nGnRE loads; group_queue reads one queue status and
 *                  walks the retired watermark. The sweep reads a core and
 *                  retires that core's task in the same step, so the read and
 *                  the retirement are one state rather than two; extraId carries
 *                  the count that separates them. This is also the only state
 *                  that mutates ready-queue depth.
 *   Sched_Submit   Handing work onward. host_build_graph writes the chosen core's
 *                  register; group_queue hands the entry to a controller that
 *                  chooses the core, reached either from dispatch or from a
 *                  group, silo or partition feed. Every such route is this state,
 *                  because a split would leave host_build_graph a row it can
 *                  never enter.
 *   Sched_Supply   Making work locally available without handing it out:
 *                  incremental classification and graph materialization.
 *   Sched_Sync     Cross-thread rendezvous: the sync_start drain, and in
 *                  group_queue the cohort service.
 *   Sched_Idle     An iteration that made no progress.
 *
 * Init and Teardown cover the thread on either side of the loop.
 *
 * Running_Task_Single and Running_Task_Pair are core lanes rather than AICPU
 * lanes, and are the one pair that must stay SET/RESET-paired: a core lane holds
 * one task at a time, and interleaved spans on it cannot be reconstructed.
 *
 * The list is capped at 22 entries. tracr_process's Paraver header defines 23
 * state colours, colour 0 being RESET, and a marker's colour is its id plus one.
 */
#define MARKER_TYPES       \
    X(Sched_Complete)      \
    X(Sched_Submit)        \
    X(Sched_Supply)        \
    X(Sched_Sync)          \
    X(Sched_Idle)          \
    X(Init)                \
    X(Teardown)            \
    X(Running_Task_Single) \
    X(Running_Task_Pair)

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

// On a simulated device the model runs on this AICPU inside the window it
// measures, so the thread's clock is real time LESS its accumulated overrun.
// Markers stamped from the real counter therefore sit on a different epoch from
// the intervals the model decides, and the gap grows through a run. These
// stamp on the model clock instead, so a trace reads as the simulated timeline.
//
// The ledger is in PLATFORM_PROF_SYS_CNT_FREQ ticks while NanoTimer reports
// nanoseconds, so the offset is converted before it is subtracted. Both derive
// from cntvct, so the difference is exact rather than an approximation.
//
// The overrun grows by at most the real time that elapsed, so model time is
// still non-decreasing: the host-side sort in StoreTracrData and the
// post-processor's k-way merge both keep their sorted-input contract.
#if defined(__SIMULATED_DEVICE__) && SIM_TRACR_MODEL_CLOCK

#include "aicpu/device_phase_aicpu.h"
#include "aicpu/device_time.h"
#include "common/platform_config.h"

inline uint64_t tracr_model_now_ns() {
    const uint64_t real_ns = TraCR::NanoTimer::now();
    const uint64_t ofs_ns = sys_cnt_ticks_to_ns(simulated_device_self_overrun_ticks(), PLATFORM_PROF_SYS_CNT_FREQ);
    return real_ns > ofs_ns ? real_ns - ofs_ns : 0;
}

inline void tracr_model_mark_set(uint16_t channelId, uint16_t eventId, uint32_t extraId = UINT32_MAX) {
    if (TRACR_UNLIKELY(!TraCR::enable_tracr)) {
        return;
    }
    TraCR::tracrThread->store_trace(TraCR::Payload{channelId, eventId, extraId, tracr_model_now_ns()});
}

inline void tracr_model_mark_reset(uint16_t channelId) {
    if (TRACR_UNLIKELY(!TraCR::enable_tracr)) {
        return;
    }
    TraCR::tracrThread->store_trace(
        TraCR::Payload{channelId, TraCR::EVENTID_RESET, UINT32_MAX, tracr_model_now_ns()}
    );
}

#undef INSTRUMENTATION_MARK_SET
#undef INSTRUMENTATION_MARK_RESET
#define INSTRUMENTATION_MARK_SET(channelId, eventId, extraId) tracr_model_mark_set(channelId, eventId, extraId)
#define INSTRUMENTATION_MARK_RESET(channelId) tracr_model_mark_reset(channelId)

#endif
