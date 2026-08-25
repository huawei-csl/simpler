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
#include "host/platform_compile_info.h"
#include "host/runtime_compile_info.h"
#include <string.h>

extern "C" {

ToolchainType get_incore_compiler(void) {
    if (strcmp(get_platform(), "a2a3") == 0) return TOOLCHAIN_CCEC;
    return TOOLCHAIN_HOST_GXX_15;
}

ToolchainType get_orchestration_compiler(void) {
    // scan_and_claim orchestrates on the HOST (like host_build_graph), so the
    // orchestration .so is built with the host toolchain on every platform --
    // no aarch64 cross-compile. This matches what KernelCompiler actually does
    // (simpler_setup/kernel_compiler.py::_orchestration_toolchain).
    //
    // NOTE: host_build_graph's copy of this file still carries tensormap's
    // TOOLCHAIN_AARCH64_GXX branch, which contradicts the Python. It is
    // harmless there only because nothing calls this hook -- the Python
    // if-chain is the live path -- but it should not be propagated.
    return TOOLCHAIN_HOST_GXX;
}
}
