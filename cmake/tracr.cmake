# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# TraCR trace instrumentation for the AICPU runtime.
#
# BUILD_TRACR is an environment variable, default OFF:
#   BUILD_TRACR=ON pip install --no-build-isolation -e '.[test]'
#
# tools/ is on the include path in both modes, because the runtime sources
# include tracr_simpler_markers.hpp unconditionally and that header supplies
# no-op markers when ENABLE_TRACR is undefined. Only BUILD_TRACR=ON reaches
# into the tools/tracr submodule, so an ordinary build needs neither the
# submodule nor a recursive checkout.
set(TRACR_SIMPLER_DIR "${CMAKE_CURRENT_LIST_DIR}/../tools")

option(BUILD_TRACR "Enable TraCR" OFF)
if(DEFINED ENV{BUILD_TRACR})
    set(BUILD_TRACR $ENV{BUILD_TRACR})
endif()

function(tracr_enable target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "tracr_enable: target '${target}' does not exist.")
    endif()

    # SYSTEM: TraCR and its vendored third-party headers (extern/nlohmann json)
    # are external to simpler and do not compile cleanly under the build's
    # -Wall -Wextra -Werror (modern clang flags nlohmann's deprecated literal
    # operators). Marking them system suppresses warnings from those headers.
    if(NOT BUILD_TRACR)
        target_include_directories(${target} SYSTEM PRIVATE ${TRACR_SIMPLER_DIR})
        return()
    endif()

    if(NOT EXISTS "${TRACR_SIMPLER_DIR}/tracr/include/tracr/tracr.hpp")
        message(FATAL_ERROR
            "BUILD_TRACR=ON but the TraCR submodule is absent. Run:\n"
            "    git submodule update --init tools/tracr"
        )
    endif()

    message(STATUS "Enabling TraCR for target: ${target}")
    target_include_directories(${target} SYSTEM PRIVATE
        ${TRACR_SIMPLER_DIR}
        ${TRACR_SIMPLER_DIR}/tracr/include
        ${TRACR_SIMPLER_DIR}/tracr/extern
    )

    # Traces are collected on the Ascend device and copied back by the host, so
    # the device never writes them to a filesystem of its own.
    target_compile_definitions(${target} PRIVATE ENABLE_TRACR TRACR_DISABLE_FLUSH USE_HW_COUNTER)

    # Per-thread trace capacity; empty keeps TraCR's own default of 1<<20
    # entries, about 17MB per thread.
    set(TRACR_CAPACITY "" CACHE STRING "TraCR per-thread buffer capacity (empty = TraCR default)")
    if(NOT "${TRACR_CAPACITY}" STREQUAL "")
        if(NOT TRACR_CAPACITY MATCHES "^[0-9]+$")
            message(FATAL_ERROR "TRACR_CAPACITY must be a positive integer")
        endif()
        target_compile_definitions(${target} PRIVATE TRACR_CAPACITY=${TRACR_CAPACITY})
    endif()

    # What a full buffer does: empty aborts, PERIODIC overwrites from the
    # start, IGNORE_IF_FULL drops incoming traces.
    set(TRACR_POLICY "" CACHE STRING "TraCR full-buffer policy (empty = TraCR default)")
    set_property(CACHE TRACR_POLICY PROPERTY STRINGS "" TRACR_POLICY_PERIODIC TRACR_POLICY_IGNORE_IF_FULL)
    if(NOT "${TRACR_POLICY}" STREQUAL "")
        if(NOT TRACR_POLICY MATCHES "^TRACR_POLICY_(PERIODIC|IGNORE_IF_FULL)$")
            message(FATAL_ERROR "Unknown TRACR_POLICY: ${TRACR_POLICY}")
        endif()
        target_compile_definitions(${target} PRIVATE ${TRACR_POLICY})
    endif()
endfunction()
