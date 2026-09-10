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
 * TraCR API functions for Simpler A2A3, A2A3sim, A5, A5sim
 *
 * TODO: A5 not yet able to test
 */

#pragma once

#include <filesystem>  // C++17 or newer
#include <fstream>
#include <array>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include <tracr/tracr.hpp>
#include <tracr_simpler_markers.hpp>
#include "aicore/tracr_aicore_layout.h"
#include "common/platform_config.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

// TraCR profiling/benchmarking stuff
size_t getSampleID() {
    const auto env = std::getenv("PYPTO_RUN_SAMPLE_ID");
    return env ? std::stoul(env) : 0;
}
size_t sampleID = getSampleID();

std::string tracr_dir = "~/ascend/tracr/proc.1";

/**
 * A function for defining the path of the TraCR traces in home
 */
fs::path expand_user_path(const std::string &path) {
    if (!path.empty() && path[0] == '~') {
        const char *home = std::getenv("HOME");
        if (!home) throw std::runtime_error("HOME not set");

        std::string sub = path.substr(1);                        // remove ~
        if (!sub.empty() && sub[0] == '/') sub = sub.substr(1);  // remove leading slash

        return fs::path(home) / sub;
    }
    return fs::path(path);
}

/**
 *
 */
inline int TracrData2BTS(const TraCR::Payload *tracrData, const size_t *tracrDataSizes, const size_t num_threads) {
    fs::path base_dir = expand_user_path(tracr_dir);

    fs::create_directories(base_dir);

    for (uint32_t t = 0; t < num_threads; ++t) {
        size_t num_traces = tracrDataSizes[t];

        if (num_traces == 0) continue;

        if (num_traces > TraCR::CAPACITY) {
            LOG_ERROR("Thread %u exceeds CAPACITY", t);
            return -1;
        }

        fs::path thread_dir = base_dir / ("thread." + std::to_string(t + 1));

        fs::create_directories(thread_dir);

        fs::path file_path = thread_dir / "traces.bts";

        std::ofstream out(file_path, std::ios::binary);
        if (!out) {
            LOG_ERROR("Cannot open %s", file_path);
            return -1;
        }

        const TraCR::Payload *thread_ptr = tracrData + t * TraCR::CAPACITY;

        out.write(reinterpret_cast<const char *>(thread_ptr), num_traces * sizeof(TraCR::Payload));

        if (!out) {
            LOG_ERROR("Write failed for %s", file_path);
            return -1;
        }
    }
    return 0;
}

/**
 * Number of channels the device lanes occupy, and therefore the id of the first
 * host lane. MUST match the device half of the channel_names array built by
 * StoreTracrMetaData(), which checks the two agree.
 */
template <typename RuntimeT>
inline size_t DeviceChannelCount(RuntimeT &runtime) {
    return static_cast<size_t>(runtime.get_aicpu_thread_num()) +        // AICPU_i
           static_cast<size_t>(runtime.get_worker_count() / 3) +        // AICube_i
           static_cast<size_t>(2 * runtime.get_worker_count() / 3) +    // AIVector_i
           1;                                                          // INVALID
}

/**
 * Serialize the AICore record slices as `.bts` lanes.
 *
 * One slice per core; a slice whose count word is 0 wrote nothing and is
 * skipped, which is why the region is zeroed at allocation. `first_thread_index`
 * continues the `thread.<n>` numbering after the AICPU threads, and each payload
 * is stamped with the channel the writer's identity word resolves to -- the
 * kernel cannot know that index, because the channel table is sized by the run's
 * core count.
 *
 * Returns the number of lanes written, or -1 on failure.
 */
template <typename DeviceRunnerT, typename RuntimeT>
int TracrAicoreLanes2BTS(
    DeviceRunnerT *device_runner, RuntimeT &runtime, size_t first_thread_index, const fs::path &proc_dir
) {
    const uint64_t base = device_runner->get_tracr_aicore_base();
    if (base == 0) return 0;

    const size_t words = static_cast<size_t>(PLATFORM_MAX_CORES) * kTracrAicoreWordsPerCore;
    std::vector<int64_t> host(words, 0);
    if (device_runner->copy_from_device(host.data(), reinterpret_cast<void *>(base), words * sizeof(int64_t)) !=
        0) {
        LOG_ERROR("TraCR AICore region: readback failed");
        return -1;
    }

    // Channel table layout, mirroring StoreTracrMetaData: aicpu_thread_num
    // AICPU_i, then worker_count/3 AICube_i, then 2*worker_count/3 AIVector_i.
    const int aicpu = runtime.get_aicpu_thread_num();
    const int cube_count = static_cast<int>(runtime.get_worker_count() / 3);

    int lanes = 0;
    for (int core = 0; core < PLATFORM_MAX_CORES; ++core) {
        const int64_t *slice = host.data() + static_cast<size_t>(core) * kTracrAicoreWordsPerCore;
        const int64_t count = slice[0];
        if (count <= 0) continue;

        const int64_t dropped = slice[1];
        if (dropped > 0) {
            LOG_TIMING("[TraCR] AICore slice %d dropped %lld record(s): buffer too small", core,
                       static_cast<long long>(dropped));
        }

        const int core_type = tracr_identity_core_type(slice[2]);
        const int block_idx = tracr_identity_block_idx(slice[2]);
        // CoreType::AIC == 0, AIV == 1.
        const int channel = (core_type == 0) ? (aicpu + block_idx) : (aicpu + cube_count + block_idx);

        const int64_t capacity = tracr_capacity_for_words(kTracrAicoreWordsPerCore);
        const int64_t usable = (count < capacity) ? count : capacity;

        std::vector<TraCR::Payload> payloads;
        payloads.reserve(static_cast<size_t>(usable));
        for (int64_t i = 0; i < usable; ++i) {
            const int64_t word0 = slice[kTracrHeaderWords + i * kTracrWordsPerPayload];
            const int64_t ts = slice[kTracrHeaderWords + i * kTracrWordsPerPayload + 1];
            TraCR::Payload payload;
            payload.channelId = static_cast<uint16_t>(channel);
            payload.eventId = static_cast<uint16_t>((static_cast<uint64_t>(word0) >> 16) & 0xFFFFu);
            payload.extraId = static_cast<uint32_t>(static_cast<uint64_t>(word0) >> 32);
            payload.timestamp = static_cast<uint64_t>(ts);
            payloads.push_back(payload);
        }

        fs::path thread_dir = proc_dir / ("thread." + std::to_string(first_thread_index + lanes + 1));
        fs::create_directories(thread_dir);
        std::ofstream out(thread_dir / "traces.bts", std::ios::binary);
        if (!out) {
            LOG_ERROR("Cannot open %s", (thread_dir / "traces.bts").c_str());
            return -1;
        }
        out.write(reinterpret_cast<const char *>(payloads.data()), payloads.size() * sizeof(TraCR::Payload));
        if (!out) {
            LOG_ERROR("Write failed for %s", (thread_dir / "traces.bts").c_str());
            return -1;
        }
        ++lanes;
    }
    return lanes;
}

/**
 * A method for storing the TraCR metadata.json
 */
template <typename RuntimeT>
int StoreTracrMetaData(RuntimeT &runtime) {
    fs::path base_dir = expand_user_path(tracr_dir);

    // Add the metadata.json
    nlohmann::json metadata;

    // channel_names
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

    if (channel_names.size() != DeviceChannelCount(runtime)) {
        LOG_ERROR(
            "TraCR channel bookkeeping drift: %zu device channel names vs %zu counted",
            channel_names.size(), DeviceChannelCount(runtime)
        );
        return -1;
    }

    metadata["channel_names"] = channel_names;
    metadata["num_channels"] = channel_names.size();

    // markerTypes
    metadata["markerTypes"] = nlohmann::json::object();

    for (int i = 0; i < MARKERTYPE_COUNT; ++i) {
        std::ostringstream oss;
        oss << std::setw(2) << std::setfill('0') << (i + 1);
        metadata["markerTypes"][oss.str()] = MarkerTypeNames[i];
    }

    metadata["pid"] = 1;
    metadata["start_time"] = 0;
    metadata["tid"] = 0;

    fs::path metadata_dir = base_dir / ("metadata.json");

    std::ofstream file(metadata_dir);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open file for writing.\n");
        return -1;
    }

    // Dump JSON into file
    file << metadata.dump(4);

    // Close the file
    file.close();

    return 0;
}

/**
 * A function for extracting the TraCR data from the Device to Host
 */
template <typename DeviceRunnerT, typename RuntimeT>
int StoreTracrData(DeviceRunnerT *device_runner, RuntimeT &runtime) {
    static_assert(
        std::is_trivially_copyable_v<TraCR::Payload>, "TraCR::Payload must be trivially copyable for raw binary dump"
    );

    if (runtime.get_tracr_data() == nullptr) {
        LOG_ERROR("runtime.tracrData_ is a nullptr");
        return -1;
    }

    if (runtime.get_tracr_data_sizes() == nullptr) {
        LOG_ERROR("runtime.tracrDataSizes_ is a nullptr");
        return -1;
    }

    if (runtime.get_aicpu_thread_num() <= 0) {
        LOG_ERROR("runtime.aicpu_thread_num is zero or negative: %d", runtime.get_aicpu_thread_num());
        return -1;
    }

    // Download the tracrData_ from Device to Host
    size_t size = sizeof(TraCR::Payload) * TraCR::CAPACITY * runtime.get_aicpu_thread_num();
    std::vector<TraCR::Payload> tracrData(TraCR::CAPACITY * runtime.get_aicpu_thread_num());
    int rc = device_runner->copy_from_device(
        reinterpret_cast<void *>(tracrData.data()), reinterpret_cast<void *>(runtime.get_tracr_data()), size
    );
    if (rc != 0) {
        LOG_ERROR("device_runner->copy_from_device 'tracrData' failed rc=%d", rc);
        return rc;
    }

    // Download the tracrDataSizes_ from Device to Host
    size = sizeof(size_t) * runtime.get_aicpu_thread_num();
    std::vector<size_t> tracrDataSizes(runtime.get_aicpu_thread_num());
    rc = device_runner->copy_from_device(
        reinterpret_cast<void *>(tracrDataSizes.data()), reinterpret_cast<void *>(runtime.get_tracr_data_sizes()), size
    );
    if (rc != 0) {
        LOG_ERROR("device_runner->copy_from_device 'tracrDataSizes' failed rc=%d", rc);
        return rc;
    }

    // Now, store the traces into '~/ascend/tracr/'
    tracr_dir =
        "~/ascend/tracr_" + std::to_string(sampleID++) + "/proc." + std::to_string(1000 + device_runner->device_id());
    rc = TracrData2BTS(tracrData.data(), tracrDataSizes.data(), runtime.get_aicpu_thread_num());
    if (rc != 0) {
        LOG_ERROR("TracrData2BTS() failed");
        return rc;
    }

    // AICore lanes continue the thread numbering after the AICPU threads, so
    // the merged trace shows the cores beside the threads that dispatched them.
    const int aicore_lanes = TracrAicoreLanes2BTS(
        device_runner, runtime, static_cast<size_t>(runtime.get_aicpu_thread_num()),
        expand_user_path(tracr_dir)
    );
    if (aicore_lanes < 0) {
        LOG_ERROR("TracrAicoreLanes2BTS() failed");
        return -1;
    }
    if (aicore_lanes > 0) {
        LOG_TIMING("[TraCR] wrote %d AICore lane(s)", aicore_lanes);
    }

    // Free device TraCR memory data placeholder
    device_runner->free_tensor(runtime.get_tracr_data());
    device_runner->free_tensor(runtime.get_tracr_data_sizes());
    if (device_runner->get_tracr_aicore_base() != 0) {
        device_runner->free_tensor(reinterpret_cast<void *>(device_runner->get_tracr_aicore_base()));
        device_runner->set_tracr_aicore_base(0);
    }

    rc = StoreTracrMetaData(runtime);
    if (rc != 0) {
        LOG_ERROR("StoreTracrMetaData failed: %d", rc);
        return rc;
    }

    return 0;
}

/**
 * A method for allocating memory on the device
 *
 * Polymorphic to A2A3 and A5 (should be)
 */
/**
 * Allocate the AICore TraCR record region and hand back its device address.
 *
 * One slice of `kTracrAicoreWordsPerCore` int64 words per core, indexed by the
 * kernel entry as `base + block_idx * kTracrAicoreWordsPerCore`. Cores never
 * share a slice, so the count word is a plain load/store and two concurrent
 * kernels cannot corrupt each other's records.
 *
 * **Zeroed here, and that is not optional.** The device writes a record count
 * into word 0 and the host decodes that many records back. A buffer the device
 * never touches -- TraCR compiled out of the kernel, or simply no comm on that
 * core -- would otherwise present stale GM as a count, and the host would decode
 * garbage rather than nothing. Zeroing makes "did not record" read as "no
 * records" instead of as a large number of nonsense ones.
 *
 * Returns 0 on failure, which the caller stores as the "off" value.
 */
template <typename DeviceRunnerT>
uint64_t DevAllocTracrAicore(DeviceRunnerT *device_runner) {
    const size_t bytes = static_cast<size_t>(PLATFORM_MAX_CORES) * kTracrAicoreWordsPerCore * sizeof(int64_t);
    void *dev_ptr = device_runner->allocate_tensor(bytes);
    if (dev_ptr == nullptr) {
        LOG_ERROR("TraCR AICore region: alloc %zu bytes failed", bytes);
        return 0;
    }
    if (device_runner->device_memset(dev_ptr, 0, bytes) != 0) {
        LOG_ERROR("TraCR AICore region: zeroing %zu bytes failed", bytes);
        device_runner->free_tensor(dev_ptr);
        return 0;
    }
    const uint64_t base = reinterpret_cast<uint64_t>(dev_ptr);
    device_runner->set_tracr_aicore_base(base);
    return base;
}

template <typename DeviceRunnerT, typename RuntimeT>
int DevAllocTraCR(DeviceRunnerT *device_runner, RuntimeT &runtime) {
    const size_t size = sizeof(TraCR::Payload) * runtime.get_aicpu_thread_num() * TraCR::CAPACITY;
    // LOG_INFO("Device alloc start of size=%u, %p", size, runtime.get_tracr_data());
    runtime.set_tracr_data(device_runner->allocate_tensor(size));
    if (runtime.get_tracr_data() == nullptr) {
        LOG_ERROR("runtime.tracrData_: alloc %zu bytes failed", size);
        return -1;
    }
    // LOG_INFO("Device alloc start of size=%u, %p", size, runtime.get_tracr_data());
    runtime.set_tracr_data_sizes(device_runner->allocate_tensor(runtime.get_aicpu_thread_num() * sizeof(size_t)));
    if (runtime.get_tracr_data_sizes() == nullptr) {
        const size_t sizes_bytes = runtime.get_aicpu_thread_num() * sizeof(size_t);
        LOG_ERROR("runtime.tracrDataSizes_: alloc %zu bytes failed", sizes_bytes);
        device_runner->free_tensor(runtime.get_tracr_data());
        runtime.set_tracr_data(nullptr);
        return -1;
    }
    return 0;
}
