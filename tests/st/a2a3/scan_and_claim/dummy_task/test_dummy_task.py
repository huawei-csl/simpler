#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""dummy_task on scan_and_claim: dep-only tasks must block consumers and never run a kernel.

Ported from the tensormap_and_ringbuffer scene by subclassing it, so the
orchestration and kernels are shared verbatim rather than duplicated — the
orchestration API is identical across runtimes, which is precisely what makes
this a fair cross-runtime comparison.

Why this scene matters for scan_and_claim specifically: a dependency-only task
occupies no AICore, so it is retired *inside the scan* at the moment its fanin
is satisfied, rather than being pushed to a queue and drained. Case 2 (a long
back-to-back dummy chain) is the direct test of that path resolving
transitively within a single scan pass: retiring id k sets its completion flag
before the same pass reaches k+1, so a chain should collapse in one pass
instead of one link per pass. Case 4 (18-way dummy fanout then fanin) covers
the dense-dependency variant.
"""

import os

from copy import deepcopy
from pathlib import Path

from simpler_setup import scene_test
from tests.st.a2a3.tensormap_and_ringbuffer.dummy_task.test_dummy_task import TestDummyTask as _TmrBase


# M4 sweeps the AICPU thread count without editing this file:
#   SAC_THREADS=1|2|3|4  (default 1, the M2/M3 single-threaded baseline)
_SAC_THREADS = int(os.environ.get("SAC_THREADS", "1"))

@scene_test(level=2, runtime="scan_and_claim")
class TestDummyTaskScanAndClaim(_TmrBase):
    _SHARED_DIR = Path(__file__).resolve().parents[2] / "tensormap_and_ringbuffer/dummy_task"

    CALLABLE = deepcopy(_TmrBase.CALLABLE)
    CALLABLE["orchestration"]["source"] = str(_SHARED_DIR / CALLABLE["orchestration"]["source"])
    for _incore in CALLABLE["incores"]:
        _incore["source"] = str(_SHARED_DIR / _incore["source"])

    # Onboard only, and single-threaded: M3 keeps N=1 so any failure is a
    # state-machine or retire-ordering bug rather than a race. The config is
    # merged, not replaced, so anything else the base case carries survives.
    CASES = [
        {
            **deepcopy(case),
            "platforms": ["a2a3"],
            "config": {**deepcopy(case.get("config", {})), "aicpu_thread_num": _SAC_THREADS},
        }
        for case in _TmrBase.CASES
    ]


if __name__ == "__main__":
    TestDummyTaskScanAndClaim.run_module(__name__)
