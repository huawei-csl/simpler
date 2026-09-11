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

/**
 * Base of the AICore TraCR record region (= KernelArgs::tracr_aicore_data_base),
 * or 0 when TraCR is off.
 *
 * Published before the scheduler's cold path runs — onboard from `k_args` at
 * the AICPU kernel entry, sim through the host's dlsym'd setter — and read
 * there to hand every core the address of its own slice through
 * `GlobalContext::tracr_aicore_slice`.
 *
 * The AICPU has no use for the region itself; it carries the address purely
 * because a generated kernel cannot get it any other way. Such a kernel links
 * into a self-contained AICore image (see `_link_incore` in
 * `simpler_setup/kernel_compiler.py`), so it cannot call the platform's
 * `get_tracr_aicore_buffer()` the way a platform-resident kernel does, and
 * `args[]` is its only channel to the runtime.
 */
extern "C" void set_platform_tracr_aicore_base(uint64_t base);
extern "C" uint64_t get_platform_tracr_aicore_base();
