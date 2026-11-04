# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Host-side offline trace post-processor. Included only under BUILD_TRACR=ON:
# it needs the tools/tracr submodule, and it pulls in Linux-only APIs that do
# not compile on the macOS packaging build.

set(TRACR_PROCESS_OUTPUT_DIR "${CMAKE_CURRENT_LIST_DIR}/../build/output/bin")

configure_file(
    ${TRACR_SIMPLER_DIR}/tracr/postprocessing/paraver/state.cfg
    ${TRACR_PROCESS_OUTPUT_DIR}/state.cfg
    COPYONLY
)

add_executable(tracr_process ${TRACR_SIMPLER_DIR}/tracr/postprocessing/tracr_process.cpp)
tracr_enable(tracr_process)
set_target_properties(tracr_process PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TRACR_PROCESS_OUTPUT_DIR})
