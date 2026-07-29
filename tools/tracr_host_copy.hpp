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
 * Host-side TraCR recorder for H2D / D2H copy cost.
 *
 * `host_runtime` is compiled with TRACR_DISABLE_FLUSH (tools/tracr.cmake), so the
 * INSTRUMENTATION_* macros keep their traces in memory and never write them out —
 * traces are normally produced on the device and only serialized by the host. Host
 * copies therefore record into the buffers here, and `StoreTracrData()` serializes
 * them into the per-device proc as extra lanes beside the AICPU/AICore ones. The
 * host reads the same hardware counter as the device (USE_HW_COUNTER), so the
 * timestamps already share the device timeline.
 *
 * One lane per recording thread, mirroring the device's thread-per-lane layout: a
 * span must be a SET/RESET pair with no third payload in between, which a single
 * shared lane could not guarantee once two threads copy concurrently. Lane
 * creation takes the registry lock once per thread; recording itself is lock-free.
 *
 * `channelId` is left 0 while recording and stamped at dump time, because the
 * device lane count it has to sit after is only known from the Runtime.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <tracr/tracr.hpp>

#include "tracr_simpler_markers.hpp"

namespace tracr_host_copy {

struct Lane {
    std::vector<TraCR::Payload> traces;
};

inline std::mutex &registry_mutex() {
    static std::mutex m;
    return m;
}

// Owns the lanes. Held by unique_ptr so appending never invalidates a Lane* that
// a recording thread already cached.
inline std::vector<std::unique_ptr<Lane>> &lanes() {
    static std::vector<std::unique_ptr<Lane>> l;
    return l;
}

inline thread_local Lane *g_lane = nullptr;

// Copies issued while this is set are the profiler downloading its own device
// trace buffers (tens of MB per AICPU thread): profiling overhead, not workload
// data movement, so they stay out of the lanes.
inline thread_local bool g_suppressed = false;

inline Lane &lane_for_this_thread() {
    if (g_lane != nullptr) {
        return *g_lane;
    }
    std::lock_guard<std::mutex> guard(registry_mutex());
    lanes().push_back(std::make_unique<Lane>());
    g_lane = lanes().back().get();
    return *g_lane;
}

/**
 * Suppresses copy recording on the calling thread for its lifetime.
 */
class Suppress {
public:
    Suppress() : previous_(g_suppressed) { g_suppressed = true; }
    ~Suppress() { g_suppressed = previous_; }
    Suppress(const Suppress &) = delete;
    Suppress &operator=(const Suppress &) = delete;

private:
    bool previous_;
};

/**
 * Records one copy as a SET/RESET span: `marker` opens it with the transferred
 * byte count in extraId, and the destructor closes it once the copy has returned.
 */
class Span {
public:
    Span(MarkerType marker, std::size_t bytes)
        : marker_(marker), bytes_(bytes), active_(!g_suppressed),
          start_(active_ ? TraCR::NanoTimer::now() : uint64_t{0}) {}

    ~Span() {
        if (!active_) {
            return;
        }
        const uint64_t stop = TraCR::NanoTimer::now();
        std::vector<TraCR::Payload> &traces = lane_for_this_thread().traces;
        // Both payloads or neither: a lone SET would leave the lane's last span
        // open to the end of the trace.
        if (traces.size() + 2 > TraCR::CAPACITY) {
            return;
        }
        traces.push_back(TraCR::Payload{0, static_cast<uint16_t>(marker_), static_cast<uint32_t>(bytes_), start_});
        traces.push_back(TraCR::Payload{0, TraCR::EVENTID_RESET, UINT32_MAX, stop});
    }

    Span(const Span &) = delete;
    Span &operator=(const Span &) = delete;

private:
    MarkerType marker_;
    std::size_t bytes_;
    bool active_;
    uint64_t start_;
};

/**
 * Drops every recorded lane. Called once the lanes have been serialized so the
 * next run starts empty (each run writes its own `~/ascend/tracr_<N>/`).
 */
inline void clear() {
    std::lock_guard<std::mutex> guard(registry_mutex());
    for (std::unique_ptr<Lane> &lane : lanes()) {
        lane->traces.clear();
    }
}

}  // namespace tracr_host_copy
