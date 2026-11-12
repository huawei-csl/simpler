# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# host_build_graph_symmetric Runtime build configuration
# All paths are relative to this file's directory (src/a5/runtime/host_build_graph_symmetric/)
#
# Symmetric-AICPU-thread variant of host_build_graph. host_build_graph splits its
# AICPU threads into (aicpu_thread_num - 1) core-owning schedulers plus one
# core-less resolution thread that drains per-scheduler SPSC queues and performs
# every completion -> ready transition. This runtime gives every AICPU thread a
# slice of the cores and resolves completions inline on the thread that observed
# them, so the resolution thread and its queues do not exist.
#
# Only the aicpu target compiles the scheduler, so this directory carries the
# five scheduler/executor translation units that differ and borrows every other
# source from ../host_build_graph. The aicore, host and orchestration targets
# name no local source at all: their binaries are built from exactly the sources
# host_build_graph builds them from, which is what makes an A/B between the two
# runtimes differ in libaicpu_kernel.so alone.
#
# scheduler.cpp is copied without modification. CMake collects each source_dirs
# entry with a recursive glob, so a directory is the smallest unit that can be
# taken from one tree or the other, and scheduler.cpp shares a directory with the
# three scheduler sources that do change.
#
# Include order matters: this runtime's "runtime" and "runtime/scheduler" precede
# host_build_graph's copies so scheduler_context.h resolves here, while
# scheduler.h, scheduler_types.h, dispatch_payload.h and the rest resolve there.
# Only the four translation units in this directory include scheduler_context.h,
# so no borrowed source can see a different SchedulerContext layout.
HBG = "../host_build_graph"
SHARED = "../../../common/host_build_graph"

BUILD_CONFIG = {
    "aicore": {
        "include_dirs": [f"{HBG}/runtime", f"{HBG}/common", "..", SHARED],
        "source_dirs": [f"{HBG}/aicore", f"{HBG}/orchestration"],
    },
    "aicpu": {
        "include_dirs": [
            "runtime",
            "runtime/scheduler",
            f"{HBG}/runtime",
            f"{HBG}/runtime/scheduler",
            f"{HBG}/common",
            "..",
            SHARED,
        ],
        "source_dirs": [
            "aicpu",
            "runtime",
            f"{HBG}/orchestration",
            f"{SHARED}/device",
            f"{SHARED}/shared",
        ],
    },
    "host": {
        "include_dirs": [f"{HBG}/runtime", f"{HBG}/common", "..", SHARED],
        "source_dirs": [
            f"{HBG}/host",
            f"{HBG}/orchestration",
            f"{SHARED}/host",
            f"{SHARED}/shared",
        ],
    },
    "orchestration": {
        "include_dirs": [
            f"{HBG}/runtime",
            f"{HBG}/orchestration",
            f"{HBG}/common",
            "..",
            SHARED,
        ],
        "source_dirs": [f"{HBG}/orchestration"],
    },
}
