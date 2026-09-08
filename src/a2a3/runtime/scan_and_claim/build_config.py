# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# scan_and_claim Runtime build configuration
# All paths are relative to this file's directory (src/a2a3/runtime/scan_and_claim/)
#
# Started life as a verbatim copy of host_build_graph (61 files) and shares its
# entire front end: the host runs the orchestrator to completion, populating
# SM + arena, then H2Ds the image to device; AICPU threads 0..N-1 all run
# schedulers (no on-device orchestrator thread); AICore executes tasks via an
# aligned PTO2DispatchPayload + pre-built dispatch_args.
#
# The intended divergence is device-side task DISCOVERY only, confined to
# runtime/scheduler/: wake lists + ready queues + the dedicated resolution
# thread are to be replaced by a bounded scan over a task-id-indexed state
# array with a single completed-prefix cursor. As of M1 the scheduler is still
# host_build_graph's, unchanged, so this tree is behaviourally identical to it.
#
# NOTE: "../../../common/host_build_graph" below is the SHARED host-state /
# recorded-graph layer (src/common/host_build_graph/: graph_execution,
# graph_cache, graph_host_state, self_relative_ptr). It is deliberately shared
# with host_build_graph, not forked -- likewise the #include "host_build_graph/..."
# lines inside runtime/. Do not rename either; src/common is an include root.
#
# The "orchestration" directory contains source files compiled into both
# runtime targets AND the orchestration .so (e.g., tensor methods needed
# by the ChipTensor constructor's validation logic).
# "host_orchestration_support" is linked only into that host-loaded .so; its
# recorder prewarm entry must not instantiate host threads in runtime targets.

# This runtime does not compile or include src/common/host_build_graph.
#
# It forked host_build_graph before that layer was split per runtime and
# redesigned (TaskId by id-space rather than ring slot, tensors carrying their
# own runtime's TaskId, in-graph-task naming, a sorted CSR). Its own frozen
# copies of the types it forked against live in runtime/ -- task_id.h, tensor.h,
# graph_execution.h, graph_cache.h, graph_host_state.h -- and win a flat include
# by the same-directory rule. That keeps this runtime's shared-memory image
# self-consistent and decoupled from a layer it shares no scheduler with.
#
# Recorded Graphs are not executable here regardless: the window scan skips
# TaskKind::GRAPH and GRAPH_NODE, and bind_graph_definitions refuses them.

BUILD_CONFIG = {
    "aicore": {"include_dirs": ["runtime", "common", ".."], "source_dirs": ["aicore", "orchestration"]},
    "aicpu": {
        "include_dirs": ["runtime", "common", ".."],
        "source_dirs": ["aicpu", "runtime", "orchestration"],
    },
    "host": {
        "include_dirs": ["runtime", "common", ".."],
        "source_dirs": ["host", "runtime/orchestrator_core", "runtime/shared", "orchestration"],
    },
    "orchestration": {
        "include_dirs": ["runtime", "orchestration", "common", ".."],
        "source_dirs": ["orchestration", "host_orchestration_support"],
    },
}
