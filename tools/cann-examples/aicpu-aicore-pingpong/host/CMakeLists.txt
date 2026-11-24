# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(aicpu_aicore_pingpong_launch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH must be set")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})

add_executable(launch_pp_census launch.cpp)
set(SIMPLER_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")

target_include_directories(launch_pp_census PRIVATE
    "${SIMPLER_REPO_ROOT}/src/common/platform/include"
    "${SIMPLER_REPO_ROOT}/src/common/log/include"
    "${ASCEND_HOME_PATH}/include"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/runtime"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/runtime/runtime"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/profiling"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/driver"
)

# libascend_dump.so, pulled in by libascendcl, resolves drvHdc*/drvDevice*
# against libascend_hal.so, which ships with the driver rather than the toolkit
# and is not on the default search path on every host. Override with
# -DASCEND_DRIVER_LIB_DIR=... where the driver lives elsewhere.
set(ASCEND_DRIVER_LIB_DIR "/usr/local/Ascend/driver/lib64/driver" CACHE PATH
    "Directory holding libascend_hal.so")

target_link_directories(launch_pp_census PRIVATE
    "${ASCEND_HOME_PATH}/lib64"
    "${ASCEND_HOME_PATH}/runtime/lib64"
    "${ASCEND_DRIVER_LIB_DIR}"
)
set_target_properties(launch_pp_census PROPERTIES
    BUILD_RPATH "${ASCEND_HOME_PATH}/lib64;${ASCEND_DRIVER_LIB_DIR}"
)

# GM-only census: no COND/DMB MMIO, so no halResMap and no direct ascend_hal link.
target_link_libraries(launch_pp_census PRIVATE ascendcl runtime dl pthread)
