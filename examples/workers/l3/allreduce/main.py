#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Minimal distributed allreduce example — onephase mesh-direct algorithm.

This is the simplest "how to do a collective" feature demo.  Each rank
stages its private vector into the HCCL window, waits for peers via a
signal barrier, then reads every peer's slot and accumulates locally.

For the full algorithm corpus (twophase, ring, bidirectional_ring, ibing)
see the scene tests at ``tests/st/worker/collectives/allreduce/``.

Run:
    python examples/workers/l3/allreduce/main.py -p a2a3sim -d 0-1

"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
from simpler.task_interface import (  # noqa: E402
    ArgDirection,
    CallConfig,
    ChipCallable,
    CommBufferSpec,
    CoreCallable,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import Worker  # noqa: E402

from simpler_setup.elf_parser import extract_text_section  # noqa: E402
from simpler_setup.kernel_compiler import KernelCompiler  # noqa: E402
from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_F32 = DataType.FLOAT32
_I64 = DataType.INT64

_KERNEL_AIV = os.path.join(HERE, "kernels", "aiv", "allreduce_onephase_kernel.cpp")
_KERNEL_ORCH = os.path.join(HERE, "kernels", "orchestration", "allreduce_onephase_orch.cpp")

ALLREDUCE_COUNT = 256
DTYPE_NBYTES = 4  # float32
K_MAX_SUPPORTED_RANKS = 16
# Phase-2 barrier timing slots written by the kernel: [0]=barrier entry,
# [1]=all TNOTIFYs issued, [2+peer]=that peer's TWAIT satisfied.
TIMING_SLOTS = 2 + K_MAX_SUPPORTED_RANKS


def parse_device_range(spec: str) -> list[int]:
    """Parse a device range string like ``0-1`` or a single device id."""
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        ids = list(range(lo, hi + 1))
    else:
        ids = [int(spec)]
    if not (2 <= len(ids) <= K_MAX_SUPPORTED_RANKS):
        raise ValueError(f"allreduce needs between 2 and {K_MAX_SUPPORTED_RANKS} devices, got {len(ids)} ({ids})")
    return ids


def build_chip_callable(platform: str) -> ChipCallable:
    """Compile the onephase allreduce kernel + orchestration shim."""
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = kc.get_orchestration_include_dirs(runtime)

    kernel_include_dirs = list(include_dirs) + [
        str(kc.project_root / "src" / "common"),
        # aicore/aicore.h -> inner_kernel.h (get_sys_cnt_aicore) lives per backend.
        str(kc.platform_dir / ("sim" if platform.endswith("sim") else "onboard") / "aicore"),
    ]
    kernel_bytes = kc.compile_incore(
        source_path=_KERNEL_AIV,
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=kernel_include_dirs,
    )
    if not platform.endswith("sim"):
        kernel_bytes = extract_text_section(kernel_bytes)

    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=_KERNEL_ORCH,
    )
    core_callable = CoreCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT, ArgDirection.OUT],
        binary=kernel_bytes,
    )
    return ChipCallable.build(
        signature=[ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT, ArgDirection.OUT],
        func_name="allreduce_orchestration",
        config_name="allreduce_orchestration_config",
        binary=orch_bytes,
        children=[(0, core_callable)],
    )


# PLATFORM_PROF_SYS_CNT_FREQ (a2a3): the AICore system counter ticks at 50 MHz,
# so one tick is 20 ns.
SYS_CNT_FREQ_HZ = 50_000_000


def report_barrier_timing(host_timings, nranks: int) -> None:
    """Print the Phase-2 barrier cost per rank from the kernel's timestamps.

    Slot layout written by allreduce_onephase_kernel.cpp:
      [0]        barrier entry
      [1]        all TNOTIFYs issued
      [2 + peer] that peer's TWAIT satisfied
    """

    def to_us(ticks: int) -> float:
        return ticks * 1e6 / SYS_CNT_FREQ_HZ

    print("[allreduce] Phase-2 barrier cost (AICore get_sys_cnt_aicore ticks @ 50 MHz):")
    for r in range(nranks):
        t = host_timings[r].tolist()
        peer_done = [(p, t[2 + p]) for p in range(nranks) if p != r and t[2 + p] != 0]
        if t[0] == 0 or not peer_done:
            print(f"  rank {r}: no barrier timing recorded")
            continue
        notify_us = to_us(t[1] - t[0])
        total_us = to_us(max(d for _, d in peer_done) - t[0])
        detail = "  ".join(f"peer{p}={to_us(d - t[1]):.1f}us" for p, d in peer_done)
        print(f"  rank {r}: notify={notify_us:7.1f}us  barrier_total={total_us:7.1f}us  |  {detail}")


def expected_output(nranks: int) -> list[float]:
    """output[i] = sum_r (i + r*100) = nranks*i + 100 * nranks*(nranks-1)/2."""
    return [float(nranks * i + 100 * nranks * (nranks - 1) // 2) for i in range(ALLREDUCE_COUNT)]


def run(device_ids: list[int], platform: str = "a2a3") -> int:
    """Core logic — callable from both CLI and pytest."""
    nranks = len(device_ids)
    if not (2 <= nranks <= K_MAX_SUPPORTED_RANKS):
        raise ValueError(f"allreduce needs between 2 and {K_MAX_SUPPORTED_RANKS} devices, got {nranks}")

    float_elems = ALLREDUCE_COUNT
    signal_tail_nbytes = K_MAX_SUPPORTED_RANKS * DTYPE_NBYTES
    scratch_nbytes = float_elems * DTYPE_NBYTES + signal_tail_nbytes
    window_size = max(scratch_nbytes, 4 * 1024)

    print(f"[allreduce] platform={platform} devices={device_ids} nranks={nranks}")

    host_inputs = [
        torch.tensor([i + rank * 100 for i in range(ALLREDUCE_COUNT)], dtype=torch.float32).share_memory_()
        for rank in range(nranks)
    ]
    host_outputs = [torch.zeros(ALLREDUCE_COUNT, dtype=torch.float32).share_memory_() for _ in range(nranks)]
    host_timings = [torch.zeros(TIMING_SLOTS, dtype=torch.int64).share_memory_() for _ in range(nranks)]

    print("[allreduce] compiling onephase kernel...")
    chip_callable = build_chip_callable(platform)

    worker = Worker(
        level=3,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
    )
    chip_handle = worker.register(chip_callable)

    try:
        print("[allreduce] init worker...")
        worker.init()

        def orch_fn(orch, _args, cfg):
            with orch.allocate_domain(
                name="default",
                workers=list(range(nranks)),
                window_size=window_size,
                buffers=[CommBufferSpec(name="scratch", dtype="float32", count=float_elems, nbytes=scratch_nbytes)],
            ) as handle:
                args_list = []
                for i in range(nranks):
                    domain = handle[i]
                    chip_args = TaskArgs()
                    chip_args.add_tensor(
                        worker.make_tensor_arg(host_inputs[i], shapes=(ALLREDUCE_COUNT,), dtype=_F32),
                        TensorArgType.INPUT,
                    )
                    chip_args.add_tensor(
                        worker.make_tensor_arg(host_outputs[i], shapes=(ALLREDUCE_COUNT,), dtype=_F32),
                        TensorArgType.OUTPUT_EXISTING,
                    )
                    chip_args.add_tensor(domain.buffers["scratch"].tensor((float_elems,), _F32), TensorArgType.INOUT)
                    chip_args.add_tensor(
                        worker.make_tensor_arg(host_timings[i], shapes=(TIMING_SLOTS,), dtype=_I64),
                        TensorArgType.OUTPUT_EXISTING,
                    )
                    chip_args.add_scalar(domain.domain_size)
                    chip_args.add_scalar(domain.device_ctx)
                    args_list.append(chip_args)
                orch.submit_next_level_group(chip_handle, args_list, cfg, workers=list(range(nranks)))

        print(f"[allreduce] running {nranks}-chip allreduce DAG...")
        worker.run(orch_fn, args=None, config=CallConfig())

        report_barrier_timing(host_timings, nranks)

        expected = torch.tensor(expected_output(nranks), dtype=torch.float32)
        ok = True
        for i in range(nranks):
            max_diff = float(torch.max(torch.abs(host_outputs[i] - expected)))
            print(f"[allreduce] chip {i}: max |out - expected| = {max_diff:.3e}")
            if max_diff > 1e-3:
                ok = False

        if not ok:
            print("[allreduce] golden check FAILED")
            return 1
        print("[allreduce] all ranks matched golden ✅")
        return 0
    finally:
        worker.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--platform", default="a2a3", help="Platform backend, e.g. a2a3 or a2a3sim.")
    parser.add_argument(
        "-d", "--device", default="0-1", help="Device range, e.g. '0-1' or '0-3'. 2 to 16 chips required."
    )
    cli = parser.parse_args()
    return run(parse_device_range(cli.device), platform=cli.platform)


if __name__ == "__main__":
    sys.exit(main())
