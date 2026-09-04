#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""scan_and_claim coverage for native_run_lifecycle: the host_build_graph scene, same kernels
and same shapes, dispatched by the scan/eager scheduler.

The workload is GRAPH-free, so it needs nothing sac does not implement.
"""

import os

from copy import deepcopy
from pathlib import Path

from simpler_setup import scene_test
from tests.st.a2a3.host_build_graph.native_run_lifecycle.test_native_run_lifecycle import TestNativeRunLifecycle as _HbgBase

# Thread count under test; 4 is the measured optimum, 1 is the single-threaded
# correctness gate.
_SAC_THREADS = int(os.environ.get("SAC_THREADS", "4"))

# Kernel and orchestration sources in the base CALLABLE may be written relative
# to the base scene's directory, which is not this file's directory.
_SHARED_DIR = Path(__file__).resolve().parents[2] / "host_build_graph/native_run_lifecycle"


def _anchor(source):
    path = Path(source)
    return str(path if path.is_absolute() else (_SHARED_DIR / path).resolve())


def _anchored(spec):
    spec = deepcopy(spec)
    orch = spec.get("orchestration")
    if isinstance(orch, dict) and "source" in orch:
        orch["source"] = _anchor(orch["source"])
    for incore in spec.get("incores", []) or []:
        if isinstance(incore, dict) and "source" in incore:
            incore["source"] = _anchor(incore["source"])
    return spec


@scene_test(level=2, runtime="scan_and_claim")
class TestNativeRunLifecycleScanAndClaim(_HbgBase):
    CALLABLE = _anchored(_HbgBase.CALLABLE) if isinstance(_HbgBase.CALLABLE, dict) else _HbgBase.CALLABLE
    CASES = [
        {
            **deepcopy(case),
            "config": {**deepcopy(case.get("config", {})), "aicpu_thread_num": _SAC_THREADS},
        }
        for case in _HbgBase.CASES
    ]


if __name__ == "__main__":
    TestNativeRunLifecycleScanAndClaim.run_module(__name__)
