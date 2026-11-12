#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""host_build_graph vs host_build_graph_symmetric on the batch=256 paged attention graph.

This case submits ~64K small tasks, which is what makes it scheduler-bound:
a chip-swimlane capture of it shows host_build_graph's core-less resolution
thread at 99.9% busy with 44 us of idle across the whole run, while the four
core-owning threads sit between 62% and 87%. It is therefore the workload that
decides whether resolving completions on the observing thread is worth doing.

Both runtimes are declared from one spec in this module, so the graph, the
kernels, the case parameters and the task-window sizing cannot drift apart --
the runtime name is the only difference between the two classes. The kernels
are ``tests/st/a5/host_build_graph/paged_attention``'s, referenced in place.

Running the module runs both back to back in one process on one device lock.
Repeat it to interleave:

    python test_paged_attention_ab.py -p a5 -d 0 --rounds 1 --skip-golden
"""

import copy

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.goldens.paged_attention import compute_golden as _pa_compute_golden
from simpler_setup.goldens.paged_attention import generate_inputs as _pa_generate_inputs

_K = "../../../../tests/st/a5/host_build_graph/paged_attention/kernels"

_CALLABLE = {
    "orchestration": {
        "source": f"{_K}/orchestration/paged_attention_orch.cpp",
        "function_name": "build_paged_attention_graph",
        "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
    },
    "incores": [
        {
            "func_id": 0,
            "name": "QK",
            "source": f"{_K}/aic/aic_qk_matmul.cpp",
            "core_type": "aic",
            "signature": [D.IN, D.IN, D.OUT],
        },
        {
            "func_id": 2,
            "name": "PV",
            "source": f"{_K}/aic/aic_pv_matmul.cpp",
            "core_type": "aic",
            "signature": [D.IN, D.IN, D.OUT],
        },
        {
            "func_id": 1,
            "name": "SF",
            "source": f"{_K}/aiv/aiv_softmax_prepare.cpp",
            "core_type": "aiv",
            "signature": [D.IN, D.OUT, D.OUT, D.OUT],
        },
        {
            "func_id": 3,
            "name": "UP",
            "source": f"{_K}/aiv/aiv_online_update.cpp",
            "core_type": "aiv",
            "signature": [D.IN, D.IN, D.IN, D.INOUT, D.INOUT, D.INOUT, D.INOUT],
        },
    ],
}

# host_build_graph is whole-graph-resident and reclaims nothing mid-run, so the
# task table has to hold all ~64K tasks at once; the default is 16384 slots.
_CASES = [
    {
        "name": "Case1",
        "platforms": ["a5"],
        "manual": True,
        "config": {"runtime_env": {"ring_task_window": 131072}},
        "params": {
            "batch": 256,
            "num_heads": 16,
            "kv_head_num": 1,
            "head_dim": 128,
            "block_size": 128,
            "context_len": 8192,
            "max_model_len": 32768,
            "dtype": "bfloat16",
        },
    },
]


class _PagedAttentionAB(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

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


@scene_test(level=2, runtime="host_build_graph")
class TestPagedAttentionABBaseline(_PagedAttentionAB):
    """Baseline: one core-less resolution thread serves the other four."""

    CALLABLE = copy.deepcopy(_CALLABLE)
    CASES = copy.deepcopy(_CASES)


@scene_test(level=2, runtime="host_build_graph_symmetric")
class TestPagedAttentionABSymmetric(_PagedAttentionAB):
    """Every AICPU thread owns cores and resolves the completions it observes."""

    CALLABLE = copy.deepcopy(_CALLABLE)
    CASES = copy.deepcopy(_CASES)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
