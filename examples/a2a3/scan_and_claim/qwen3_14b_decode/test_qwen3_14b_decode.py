#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B decode on scan_and_claim, driven by the tensormap_and_ringbuffer driver.

The driver is reused rather than copied so both runtimes execute the *same* graph
-- ~19.2K ordinary ring tasks -- which is what makes the two device walls
comparable. (host_build_graph's copy is a different graph form: 40 recorded
Graphs.) Reusing it also keeps the driver's relative CALLABLE sources resolving
against its own directory, so nothing needs re-anchoring here.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from simpler_setup.scene_test import standalone_pytest_options

_DRIVER_NAME = "_qwen3_14b_a2a3_sac_driver"
_DRIVER_PATH = Path(__file__).resolve().parents[2] / "tensormap_and_ringbuffer" / "qwen3_14b_decode" / "main.py"

# ~19.2K tasks exceed the default window, and host orchestration holds all 40
# layers' intermediates live at once. Env-var sizing is retired; this is the knob.
_RUNTIME_ENV = {"ring_task_window": 32768, "ring_heap": 1024 * 1024 * 1024}
_OVERRIDES = {"runtime": "scan_and_claim", "runtime_env": _RUNTIME_ENV}


def _driver():
    cached = sys.modules.get(_DRIVER_NAME)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(_DRIVER_NAME, _DRIVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the Qwen3-14B decode driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_DRIVER_NAME] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.manual
@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime("scan_and_claim")
@pytest.mark.device_count(1)
def test_qwen3_14b_decode(st_platform, st_device_ids, request):
    assert _driver().run(st_device_ids, st_platform, **_OVERRIDES, **standalone_pytest_options(request)) == 0


if __name__ == "__main__":
    sys.exit(_driver().main(**_OVERRIDES))
