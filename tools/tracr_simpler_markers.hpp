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
 * TraCR Simpler Marker Types
 */

#pragma once

#include <atomic>
#include <string_view>

// Global TraCR thread idx counter
std::atomic<int> g_TraCR_thread_idx_counter{0};

// Global thread local thread idx placeholder to capture traces
thread_local int g_TraCR_thread_idx{-1};

#define MARKER_TYPES    \
    X(Orchestrating)    \
    X(Scheduling)       \
    X(Initializing)     \
    X(De_Initializing)  \
    X(DLL_loading)      \
    X(Allocating)       \
    X(Running_Task_Single) \
    X(Running_Task_Pair)
    

enum MarkerType {
#define X(name) name,
    MARKER_TYPES
#undef X

    MARKERTYPE_COUNT
};

constexpr std::string_view MarkerTypeNames[] = {
#define X(name) #name,
    MARKER_TYPES
#undef X
};