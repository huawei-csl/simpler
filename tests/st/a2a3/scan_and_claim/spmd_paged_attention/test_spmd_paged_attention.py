#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Manual scan_and_claim coverage for the disabled SPMD paged-attention workload."""

import os

from copy import deepcopy
from pathlib import Path

from simpler_setup import scene_test
from tests.st.a2a3.tensormap_and_ringbuffer.spmd_paged_attention.test_spmd_paged_attention import (
    TestPagedAttentionUnrollTpushPop as _TmrBase,
)


# M4 sweeps the AICPU thread count without editing this file:
#   SAC_THREADS=1|2|3|4  (default 4; set 1 for the single-threaded correctness gate)
_SAC_THREADS = int(os.environ.get("SAC_THREADS", "4"))

@scene_test(level=2, runtime="scan_and_claim")
class TestSpmdPagedAttentionScanAndClaim(_TmrBase):
    _SHARED_DIR = Path(__file__).resolve().parents[2] / "tensormap_and_ringbuffer/spmd_paged_attention"
    CALLABLE = deepcopy(_TmrBase.CALLABLE)
    CALLABLE["orchestration"]["source"] = str(_SHARED_DIR / "kernels/orchestration/spmd_paged_attention_orch.cpp")
    for _incore in CALLABLE["incores"]:
        _incore["source"] = str(_SHARED_DIR / "kernels/mix/paged_attention_parallel.cpp")

    # The TMR shapes, run through scan_and_claim. Deep-copied so the two
    # classes do not share a params dict. Onboard only: the MIX kernel's
    # TPUSH/TPOP path does not build for the CPU simulator (PTO-ISA rejects the
    # ColMajor TMULS in the softmax tail), so a2a3sim is out of scope. See
    # #1832. Every HBG case stays manual; the Per-PR sweep covers these shapes
    # on the TMR side.
    CASES = [
        {
            **deepcopy(case),
            "platforms": ["a2a3"],
            "manual": True,
            # M3 runs single-threaded: merged, not replaced, so any
            # ring-sizing runtime_env on the base case survives.
            "config": {**deepcopy(case.get("config", {})), "aicpu_thread_num": _SAC_THREADS},
        }
        for case in _TmrBase.CASES
    ]


if __name__ == "__main__":
    TestSpmdPagedAttentionScanAndClaim.run_module(__name__)
