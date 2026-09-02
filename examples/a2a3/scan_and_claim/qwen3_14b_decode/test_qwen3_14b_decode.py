#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B 40-layer decode on scan_and_claim, as ORDINARY ring tasks.

Deliberately subclasses the **tensormap_and_ringbuffer** scene, not the
host_build_graph one. That choice is the whole point of this file:

* hbg's qwen3 is built on recorded Graphs (`rt_submit_graph`, one Definition per
  layer, case named `GraphExecutionBatch16Seq3500`). Graph Execution is an
  hbg-only feature, and scan_and_claim does not implement it -- the scan skips
  `TaskKind::GRAPH` / `GRAPH_NODE`, because graph-internal nodes hold no ring
  task-window slot and no ring completion flag, so there is nothing for a
  flag-scanning scheduler to find. Pointing sac at the hbg parent produces a
  `SCHEDULER_TIMEOUT`: the graph tasks are never dispatched.
* The tmr orchestration submits the same 40 layers as plain ring tasks -- zero
  `rt_submit_graph` calls, ~19,208 tasks per round. That is precisely the large
  flat graph the scan handles well, and it is a far better scheduler benchmark
  than the graph variant, whose device wall is trivial next to its host bind.

Sizing (neither this file nor the hbg/tmr ones set `runtime_env`, so the env
vars apply identically across runtimes -- size symmetrically, in the environment):

    PTO2_RING_TASK_WINDOW=32768        # ~19.2K tasks exceeds the 16384 default
    PTO2_RING_HEAP=$((1024*1024*1024)) # host-orch holds all 40 layers' live
                                       # intermediates at once; 512 MiB carries
                                       # it, 1 GiB leaves margin

Expect this to be AICore-bound: earlier measurement put AICore task execution at
~99.9% of device wall (~1.9 us per task vs ~0.34 us for PA Case1), and five
different scheduler configurations landed within 0.1% of each other. Treat it as
a realism check -- proof the runtime drives a real model end to end -- not as the
workload for judging scheduler changes.
"""

import copy
import os
from pathlib import Path

from simpler_setup import scene_test
from examples.a2a3.tensormap_and_ringbuffer.qwen3_14b_decode.test_qwen3_14b_decode import (
    TestQwen314BDecode as _TmrBase,
)

# Shared scene directory. The base CALLABLE names its sources RELATIVE to its own
# directory, so every one has to be re-anchored there -- a subclass living
# elsewhere would otherwise resolve them against this file's directory.
_SHARED_DIR = Path(__file__).resolve().parents[2] / "tensormap_and_ringbuffer/qwen3_14b_decode"

# Sweep the AICPU thread count without editing this file: SAC_THREADS=1|2|3|4
_SAC_THREADS = int(os.environ.get("SAC_THREADS", "4"))


def _callable():
    cfg = copy.deepcopy(_TmrBase.CALLABLE)
    cfg["orchestration"]["source"] = str(_SHARED_DIR / cfg["orchestration"]["source"])
    for incore in cfg["incores"]:
        incore["source"] = str(_SHARED_DIR / incore["source"])
    return cfg


@scene_test(level=2, runtime="scan_and_claim")
class TestQwen314BDecodeScanAndClaim(_TmrBase):
    CALLABLE = _callable()

    CASES = [
        {
            "name": "StressBatch16Seq3500",
            "platforms": ["a2a3"],
            "manual": True,
            "config": {
                "aicpu_thread_num": _SAC_THREADS,
                # ~19.2K tasks exceed the 16384 default window; host-orch holds all
                # 40 layers' intermediates live at once. Env sizing is retired on main.
                "runtime_env": {"ring_task_window": 32768, "ring_heap": 1073741824},
            },
            "params": {"seed": 1234, "seq_len": 3500},
        },
    ]


if __name__ == "__main__":
    TestQwen314BDecodeScanAndClaim.run_module(__name__)
