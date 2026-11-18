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
#pragma once

#include <cstdint>

namespace simpler::hbg {

enum class ClusterPlacement : uint8_t { Invalid, RoundRobin, DieLocal };

struct ClusterDies {
    uint8_t aic;
    uint8_t aiv0;
    uint8_t aiv1;
};

// The full cluster remains indivisible. Physical locality is used only when
// every actual participant is mapped and both dies have equal participation.
template <int32_t ClusterCapacity, int32_t ThreadCapacity>
ClusterPlacement plan_cluster_assignment(
    const ClusterDies *clusters, int32_t cluster_count, const int32_t *thread_cpus, const int8_t *thread_dies,
    int32_t thread_count, int32_t (&owners)[ClusterCapacity], int32_t (&counts)[ThreadCapacity]
) {
    for (int32_t t = 0; t < ThreadCapacity; ++t)
        counts[t] = 0;
    if (cluster_count <= 0 || cluster_count > ClusterCapacity || thread_count <= 0 || thread_count > ThreadCapacity)
        return ClusterPlacement::Invalid;

    for (int32_t c = 0; c < cluster_count; ++c) {
        owners[c] = c % thread_count;
        ++counts[owners[c]];
    }

    int32_t die_threads[2][ThreadCapacity]{};
    int32_t die_thread_count[2]{};
    for (int32_t t = 0; t < thread_count; ++t) {
        if (thread_cpus[t] < 0 || thread_dies[t] < 0 || thread_dies[t] >= 2) return ClusterPlacement::RoundRobin;
        for (int32_t previous = 0; previous < t; ++previous)
            if (thread_cpus[previous] == thread_cpus[t]) return ClusterPlacement::RoundRobin;
        const int32_t die = thread_dies[t];
        die_threads[die][die_thread_count[die]++] = t;
    }
    if (die_thread_count[0] == 0 || die_thread_count[0] != die_thread_count[1]) return ClusterPlacement::RoundRobin;

    int32_t die_clusters[2]{};
    for (int32_t c = 0; c < cluster_count; ++c) {
        const ClusterDies &cluster = clusters[c];
        if (cluster.aic >= 2 || cluster.aiv0 != cluster.aic || cluster.aiv1 != cluster.aic)
            return ClusterPlacement::RoundRobin;
        ++die_clusters[cluster.aic];
    }
    if (die_clusters[0] == 0 || die_clusters[1] == 0) return ClusterPlacement::RoundRobin;

    for (int32_t t = 0; t < thread_count; ++t)
        counts[t] = 0;
    int32_t next[2]{};
    for (int32_t c = 0; c < cluster_count; ++c) {
        const int32_t die = clusters[c].aic;
        owners[c] = die_threads[die][next[die]++ % die_thread_count[die]];
        ++counts[owners[c]];
    }
    return ClusterPlacement::DieLocal;
}

}  // namespace simpler::hbg
