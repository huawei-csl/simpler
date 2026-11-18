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

namespace simpler::a5 {

inline constexpr uint32_t kUnknownAicpuDieMap = UINT32_MAX;
inline constexpr int32_t kAicpuDieMapCpuCount = 16;

// Two bits per logical CPU; three denotes an unknown physical die.
inline constexpr int32_t aicpu_die_from_map(uint32_t map, int32_t cpu) {
    if (cpu < 0 || cpu >= kAicpuDieMapCpuCount) return -1;
    const uint32_t die = (map >> (2 * cpu)) & 3u;
    return die < 2 ? static_cast<int32_t>(die) : -1;
}

inline constexpr uint32_t aicpu_die_map_set(uint32_t map, int32_t cpu, int32_t die) {
    if (cpu < 0 || cpu >= kAicpuDieMapCpuCount || die < 0 || die >= 2) return map;
    const uint32_t shift = 2 * cpu;
    return (map & ~(3u << shift)) | (static_cast<uint32_t>(die) << shift);
}

}  // namespace simpler::a5
