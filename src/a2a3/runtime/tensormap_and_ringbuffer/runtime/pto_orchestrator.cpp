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

#include "pto_orchestrator.h"

#include "aicpu/dep_gen_collector_aicpu.h"
#include "common/dep_gen.h"
#include "tensor.h"

static_assert(sizeof(Tensor) == DEP_GEN_TENSOR_SIZE, "DepGenRecord::tensors slot size out of sync with sizeof(Tensor)");

extern "C" __attribute__((weak, visibility("hidden"))) bool is_dep_gen_enabled()
{
    return false;
}
__attribute__((weak, visibility("hidden"))) void dep_gen_aicpu_record_submit(uint64_t, bool, int, const void *const *, const uint8_t *, int, const uint64_t *, const int32_t[3])
{}

extern "C" __attribute__((weak, visibility("hidden"))) bool is_scope_stats_enabled()
{
    return false;
}
