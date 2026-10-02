#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B decode on host_build_graph with device-resident parameters."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TMR_CASE_DIR = HERE.parents[1] / "tensormap_and_ringbuffer/qwen3_14b_decode"
RUNTIME = "host_build_graph"
# Two orchestrations of the same 40 layers. The recorded one submits each layer as
# a Graph for the device Scheduler to expand; the expanded one runs the layer
# definition inline, so the same tasks and edges reach the manager directly. Only
# the expanded one can run under group_queue, which does not expand a Graph, and
# an arm-to-arm comparison needs both sides on the same one.
GRAPH_ORCHESTRATION_SOURCE = HERE / "kernels/orchestration/decode_fwd_layers.cpp"
EXPANDED_ORCHESTRATION_SOURCE = HERE / "kernels/orchestration/decode_fwd_layers_expanded.cpp"
ORCHESTRATION_SOURCE = GRAPH_ORCHESTRATION_SOURCE


def expanded_source_for(n_layers: int, out_dir: Path | None = None) -> Path:
    """The expanded orchestration, rewritten for a shorter decoder stack.

    The layer count is one `constexpr` the orchestration loops over, and it also
    divides the KV pool into per-layer pages, so it has to agree with the
    `n_layers` the host allocates for -- a mismatch misaddresses the pool rather
    than failing. Deriving the source here is what keeps the two from drifting.
    """
    if n_layers == 40:
        return EXPANDED_ORCHESTRATION_SOURCE
    if not 1 <= n_layers <= 40:
        raise ValueError(f"n_layers must be in 1..40, got {n_layers}")
    text = EXPANDED_ORCHESTRATION_SOURCE.read_text(encoding="utf-8")
    needle = "constexpr uint32_t num_layers = 40;"
    if text.count(needle) != 1:
        raise RuntimeError(f"expected exactly one {needle!r} in {EXPANDED_ORCHESTRATION_SOURCE}")
    text = text.replace(needle, f"constexpr uint32_t num_layers = {n_layers};")
    out_dir = out_dir or (HERE / "kernels/orchestration/generated")
    out_dir.mkdir(parents=True, exist_ok=True)
    derived = out_dir / f"decode_fwd_layers_expanded_l{n_layers}.cpp"
    derived.write_text(text, encoding="utf-8")
    return derived
CASE_NAME = "GraphExecutionBatch16Seq3500"


def _load_tmr_driver():
    module_name = "_qwen3_14b_a2a3_tmr_driver"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, TMR_CASE_DIR / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the TMR Qwen3-14B decode driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run(device_ids, platform: str, **kwargs) -> int:
    driver = _load_tmr_driver()
    kwargs.setdefault("runtime", RUNTIME)
    kwargs.setdefault("orchestration_source", ORCHESTRATION_SOURCE)
    kwargs.setdefault("runtime_env", {})
    return driver.run(device_ids, platform, **kwargs)


def main(argv=None) -> int:
    driver = _load_tmr_driver()
    return driver.main(
        argv,
        case_name=CASE_NAME,
        runtime=RUNTIME,
        orchestration_source=ORCHESTRATION_SOURCE,
        runtime_env={},
    )


if __name__ == "__main__":
    sys.exit(main())
