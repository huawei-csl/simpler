#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Paged attention — scan_and_claim runtime.

Tests scan_and_claim runtime with AIC+AIV mixed execution and INOUT tensors.
Templated kernels support variable tile sizes via runtime dispatch.
"""

import os

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.goldens.paged_attention import compute_golden as _pa_compute_golden  # noqa: PLC0415
from simpler_setup.goldens.paged_attention import generate_inputs as _pa_generate_inputs  # noqa: PLC0415


# M4 sweeps the AICPU thread count without editing this file:
#   SAC_THREADS=1|2|3|4  (default 4; set 1 for the single-threaded correctness gate)
_SAC_THREADS = int(os.environ.get("SAC_THREADS", "4"))

@scene_test(level=2, runtime="scan_and_claim")
class TestPagedAttentionScanAndClaim(SceneTestCase):
    """Paged attention with scan_and_claim runtime."""

    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/paged_attention_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "QK",
                "source": "kernels/aic/aic_qk_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "name": "PV",
                "source": "kernels/aic/aic_pv_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "SF",
                "source": "kernels/aiv/aiv_softmax_prepare.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT, D.OUT, D.OUT],
            },
            {
                "func_id": 3,
                "name": "UP",
                "source": "kernels/aiv/aiv_online_update.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.IN, D.INOUT, D.INOUT, D.INOUT, D.INOUT],
            },
        ],
    }

    CASES = [
        {
            # Marked manual for scan_and_claim (as for host_build_graph): this batch=256 case submits
            # ~64K tasks, and host-orchestration populates the whole task graph
            # before the device schedules — so the ring/heap cannot reclaim
            # mid-orchestration and must hold the entire graph at once. That
            # exceeds the default ring window / GM heap. Run it explicitly with
            # a large PTO2_RING_TASK_WINDOW / PTO2_RING_HEAP if needed.
            "name": "Case1",
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": _SAC_THREADS,
                       # whole-graph-resident: window and heap must hold the entire
                       # graph. The PTO2_RING_* env vars are retired on main (warn +
                       # ignore), so sizing lives here, per case.
                       "runtime_env": {"ring_task_window": 131072, "ring_heap": 2147483648}},
            "manual": True,
            "params": {
                "batch": 256,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 128,
                "block_size": 128,
                "context_len": 8100,
                "max_model_len": 32768,
                "dtype": "bfloat16",
            },
        },
        {
            "name": "Case2",
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": _SAC_THREADS,
                       # whole-graph-resident: window and heap must hold the entire
                       # graph. The PTO2_RING_* env vars are retired on main (warn +
                       # ignore), so sizing lives here, per case.
                       "runtime_env": {"ring_task_window": 65536, "ring_heap": 1073741824}},
            "manual": True,
            "params": {
                "batch": 64,
                "num_heads": 64,
                "kv_head_num": 1,
                "head_dim": 128,
                "block_size": 64,
                "context_len": 8150,
                "max_model_len": 32768,
                "dtype": "bfloat16",
            },
        },
        {
            "name": "small1",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": _SAC_THREADS},
            "params": {
                "batch": 1,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 16,
                "block_size": 16,
                "context_len": 16,
                "max_model_len": 256,
                "dtype": "bfloat16",
            },
        },
        {
            "name": "small2",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": _SAC_THREADS},
            "manual": True,
            "params": {
                "batch": 1,
                "num_heads": 16,
                "kv_head_num": 1,
                "head_dim": 16,
                "block_size": 16,
                "context_len": 64,
                "max_model_len": 256,
                "dtype": "bfloat16",
            },
        },
    ]

    def generate_args(self, params):
        inputs = _pa_generate_inputs(params)
        specs = []
        for name, val in inputs:
            if isinstance(val, torch.Tensor):
                specs.append(TensorArg(name, val))
            else:
                specs.append(Scalar(name, val))
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        tensors = {s.name: s.value for s in args.specs if isinstance(s, TensorArg)}
        _pa_compute_golden(tensors, params)
        for s in args.specs:
            if isinstance(s, TensorArg) and s.name in tensors:
                getattr(args, s.name)[:] = tensors[s.name]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
