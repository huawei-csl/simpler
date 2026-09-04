#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 FLASH decode on scan_and_claim, as ordinary ring tasks.

The host_build_graph copy of this network expresses each layer as a recorded
Graph (`rt_submit_graph`), which scan_and_claim does not implement. The
tensormap_and_ringbuffer copy submits the same computation as ordinary ring
tasks, so subclassing it is the flattened form of the same program — no graph
support required.

Distributed: EP2/TP2 over N_RANKS devices, so this needs `-d <d0>,<d1>`.
"""

import copy
import os

from pathlib import Path

from simpler_setup import scene_test
from examples.a2a3.tensormap_and_ringbuffer.deepseek_v4_flash_decode.test_deepseek_v4_flash_decode import (
    TestDeepseekV4FlashDecode as _TmrBase,
)

# 4 is the measured optimum; 1 is the single-threaded correctness gate.
_SAC_THREADS = int(os.environ.get("SAC_THREADS", "4"))

# Kernel and orchestration sources are written relative to the tmr scene dir.
_SHARED_DIR = Path(__file__).resolve().parents[2] / "tensormap_and_ringbuffer/deepseek_v4_flash_decode"


def _anchored():
    cfg = copy.deepcopy(_TmrBase.CALLABLE)
    for entry in cfg["callables"]:
        entry["orchestration"]["source"] = str(_SHARED_DIR / entry["orchestration"]["source"])
        for incore in entry["incores"]:
            incore["source"] = str(_SHARED_DIR / incore["source"])
    return cfg


@scene_test(level=3, runtime="scan_and_claim")
class TestDeepseekV4FlashDecodeScanAndClaim(_TmrBase):
    CALLABLE = _anchored()
    CASES = [
        {
            **copy.deepcopy(case),
            "config": {**copy.deepcopy(case.get("config", {})), "aicpu_thread_num": _SAC_THREADS},
        }
        for case in _TmrBase.CASES
    ]


if __name__ == "__main__":
    TestDeepseekV4FlashDecodeScanAndClaim.run_module(__name__)
