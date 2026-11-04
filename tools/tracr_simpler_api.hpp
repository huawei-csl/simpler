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
 * Host side of the TraCR capture: allocate the device trace buffer before a
 * run, and retrieve it into ~/ascend/tracr_<sample>/ afterwards.
 *
 * Both entry points are no-ops returning 0 unless ENABLE_TRACR is defined, so
 * the call sites in each platform's device_runner carry no preprocessor.
 */

#pragma once

#include <cstdint>

#include "common/unified_log.h"

#include <tracr_simpler_markers.hpp>

#ifdef ENABLE_TRACR

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

/**
 * Resolve a leading ~ against HOME; other paths pass through unchanged.
 */
inline std::filesystem::path tracr_expand_user_path(const std::string &path) {
    if (!path.empty() && path[0] == '~') {
        const char *home = std::getenv("HOME");
        if (home == nullptr) throw std::runtime_error("HOME not set");

        std::string sub = path.substr(1);
        if (!sub.empty() && sub[0] == '/') sub = sub.substr(1);

        return std::filesystem::path(home) / sub;
    }
    return std::filesystem::path(path);
}

/**
 * Write one traces.bts per lane under base_dir, in the binary trace format the
 * tracr_process post-processor reads. Lanes that recorded nothing are skipped.
 */
inline int tracr_write_bts(
    const std::filesystem::path &base_dir, const TraCR::Payload *traces, const size_t *trace_counts, int num_threads
) {
    std::filesystem::create_directories(base_dir);

    for (int t = 0; t < num_threads; ++t) {
        const size_t num_traces = trace_counts[t];
        if (num_traces == 0) continue;

        if (num_traces > TraCR::CAPACITY) {
            LOG_ERROR("TraCR lane %d recorded %zu traces, past the %zu capacity", t, num_traces, TraCR::CAPACITY);
            return -1;
        }

        const std::filesystem::path thread_dir = base_dir / ("thread." + std::to_string(t + 1));
        std::filesystem::create_directories(thread_dir);

        const std::filesystem::path file_path = thread_dir / "traces.bts";
        std::ofstream out(file_path, std::ios::binary);
        if (!out) {
            LOG_ERROR("Cannot open %s", file_path.c_str());
            return -1;
        }

        out.write(
            reinterpret_cast<const char *>(traces + static_cast<size_t>(t) * TraCR::CAPACITY),
            static_cast<std::streamsize>(num_traces * sizeof(TraCR::Payload))
        );
        if (!out) {
            LOG_ERROR("Write failed for %s", file_path.c_str());
            return -1;
        }
    }
    return 0;
}

/**
 * Write the capture's metadata.json: one channel name per AICPU lane and per
 * AICore, plus the marker-id to marker-name table the lanes reference.
 */
template <typename RuntimeT>
int tracr_write_metadata(const std::filesystem::path &base_dir, RuntimeT &runtime) {
    nlohmann::json channel_names = nlohmann::json::array();
    for (int i = 0; i < runtime.get_aicpu_thread_num(); ++i) {
        channel_names.push_back("AICPU_" + std::to_string(i));
    }
    for (int i = 0; i < int(runtime.get_worker_count() / 3); ++i) {
        channel_names.push_back("AICube_" + std::to_string(i));
    }
    for (int i = 0; i < int(2 * runtime.get_worker_count() / 3); ++i) {
        channel_names.push_back("AIVector_" + std::to_string(i));
    }
    channel_names.push_back("INVALID");

    nlohmann::json metadata;
    metadata["channel_names"] = channel_names;
    metadata["num_channels"] = channel_names.size();
    metadata["markerTypes"] = nlohmann::json::object();
    for (int i = 0; i < MARKERTYPE_COUNT; ++i) {
        std::ostringstream oss;
        oss << std::setw(2) << std::setfill('0') << (i + 1);
        metadata["markerTypes"][oss.str()] = MarkerTypeNames[i];
    }
    metadata["pid"] = 1;
    metadata["tid"] = 0;

    std::ofstream file(base_dir / "metadata.json");
    if (!file.is_open()) {
        LOG_ERROR("%s", "Failed to open metadata.json for writing");
        return -1;
    }
    file << metadata.dump(4);
    return 0;
}

/**
 * Reserve the device trace buffer for this run: one TraCR::CAPACITY-entry lane
 * per AICPU thread, plus the per-lane fill counts the AICPU writes back.
 */
template <typename DeviceRunnerT, typename RuntimeT>
int tracr_alloc_device_buffers(DeviceRunnerT *device_runner, RuntimeT &runtime) {
    const size_t traces_bytes =
        sizeof(TraCR::Payload) * static_cast<size_t>(runtime.get_aicpu_thread_num()) * TraCR::CAPACITY;
    void *traces = device_runner->allocate_tensor(traces_bytes);
    if (traces == nullptr) {
        LOG_ERROR("TraCR trace buffer: alloc %zu bytes failed", traces_bytes);
        return -1;
    }

    const size_t counts_bytes = static_cast<size_t>(runtime.get_aicpu_thread_num()) * sizeof(size_t);
    void *counts = device_runner->allocate_tensor(counts_bytes);
    if (counts == nullptr) {
        LOG_ERROR("TraCR lane counts: alloc %zu bytes failed", counts_bytes);
        device_runner->free_tensor(traces);
        return -1;
    }

    runtime.set_tracr_data(reinterpret_cast<uint64_t>(traces));
    runtime.set_tracr_data_sizes(reinterpret_cast<uint64_t>(counts));
    return 0;
}

/**
 * Copy this run's traces back from the device, write them under
 * ~/ascend/tracr_<sample>/proc.<1000 + device id>/, and release the device
 * buffer. The caller must have synchronized the run's streams first, so the
 * AICPU has finished writing into the buffer.
 */
template <typename DeviceRunnerT, typename RuntimeT>
int tracr_store_traces_impl(DeviceRunnerT *device_runner, RuntimeT &runtime) {
    static_assert(
        std::is_trivially_copyable_v<TraCR::Payload>, "TraCR::Payload must be trivially copyable for raw binary dump"
    );

    // Reserved per prepare and released here, but a Runtime object is shared
    // across overlapping pipeline slots, so a predecessor's drain can already
    // have released the buffer a successor reserved. There is then nothing to
    // capture, which must not fail the successor's run.
    const int num_threads = runtime.get_aicpu_thread_num();
    if (runtime.get_tracr_data() == 0 || runtime.get_tracr_data_sizes() == 0 || num_threads <= 0) {
        LOG_WARN("No TraCR buffer is reserved for this run (aicpu_thread_num=%d); capturing nothing", num_threads);
        return 0;
    }

    std::vector<TraCR::Payload> traces(TraCR::CAPACITY * static_cast<size_t>(num_threads));
    int rc = device_runner->copy_from_device(
        traces.data(), reinterpret_cast<void *>(runtime.get_tracr_data()), traces.size() * sizeof(TraCR::Payload)
    );
    if (rc != 0) {
        LOG_ERROR("copy_from_device of the TraCR trace buffer failed rc=%d", rc);
        return rc;
    }

    std::vector<size_t> trace_counts(static_cast<size_t>(num_threads));
    rc = device_runner->copy_from_device(
        trace_counts.data(), reinterpret_cast<void *>(runtime.get_tracr_data_sizes()),
        trace_counts.size() * sizeof(size_t)
    );
    if (rc != 0) {
        LOG_ERROR("copy_from_device of the TraCR lane counts failed rc=%d", rc);
        return rc;
    }

    // One capture directory per run. PYPTO_RUN_SAMPLE_ID seeds the counter so a
    // benchmark's repeated runs land in sibling directories rather than
    // overwriting each other.
    static size_t sample_id = []() {
        const char *env = std::getenv("PYPTO_RUN_SAMPLE_ID");
        return env != nullptr ? std::stoul(env) : 0UL;
    }();
    const std::filesystem::path base_dir = tracr_expand_user_path(
        "~/ascend/tracr_" + std::to_string(sample_id++) + "/proc." + std::to_string(1000 + device_runner->device_id())
    );

    rc = tracr_write_bts(base_dir, traces.data(), trace_counts.data(), num_threads);
    if (rc != 0) return rc;

    device_runner->free_tensor(reinterpret_cast<void *>(runtime.get_tracr_data()));
    device_runner->free_tensor(reinterpret_cast<void *>(runtime.get_tracr_data_sizes()));
    runtime.set_tracr_data(0);
    runtime.set_tracr_data_sizes(0);

    return tracr_write_metadata(base_dir, runtime);
}

/**
 * The trace capture runs inside the device drain path, where an escaping
 * exception is caught blind and reported as an internal device error. A
 * filesystem that cannot take the traces must fail the capture, not the run's
 * diagnosis, so every throw is converted to a return code here.
 */
template <typename DeviceRunnerT, typename RuntimeT>
int tracr_store_traces(DeviceRunnerT *device_runner, RuntimeT &runtime) {
    try {
        return tracr_store_traces_impl(device_runner, runtime);
    } catch (const std::exception &e) {
        LOG_ERROR("TraCR capture threw: %s", e.what());
        return -1;
    } catch (...) {
        LOG_ERROR("%s", "TraCR capture threw a non-std exception");
        return -1;
    }
}

#else /* ENABLE_TRACR */

template <typename DeviceRunnerT, typename RuntimeT>
inline int tracr_alloc_device_buffers(DeviceRunnerT *, RuntimeT &) {
    return 0;
}

template <typename DeviceRunnerT, typename RuntimeT>
inline int tracr_store_traces(DeviceRunnerT *, RuntimeT &) {
    return 0;
}

#endif /* ENABLE_TRACR */
