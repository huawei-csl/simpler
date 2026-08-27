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
import json
import os
import struct
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
# The kernel writes TraCR Payloads (16 B = two int64 words) into this buffer:
# one notify span plus one wait span per peer, each a SET/RESET pair.
TRACR_PAYLOAD_CAP = 1 + 2 + 2 * K_MAX_SUPPORTED_RANKS  # header + notify span + per-peer spans
TRACR_SLOTS = 2 * TRACR_PAYLOAD_CAP


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


# Must match the constants in allreduce_onephase_kernel.cpp. verify_tracr_ids()
# re-checks them against the metadata the runtime actually emitted, so a change
# to the channel layout or the MarkerType X-macro fails the run instead of
# silently relabelling the lane.
TRACR_CHANNEL_NAME = "AIVector_0"
TRACR_EVENT_PHASE2 = 7
TRACR_EVENT_BARRIER = 17
TRACR_EVENT_RESET = 0xFFFF
TRACR_PAYLOAD_FMT = struct.Struct("<HHIQ")


def decode_payloads(words: list[int]) -> list[tuple[int, int, int, int]]:
    """Decode the kernel's buffer into (channel, event, extra, ts_ns) records.

    Word 0 is the record count. It is authoritative: the buffer is
    OUTPUT_EXISTING, so words past the count hold stale device memory rather
    than zeros and must not be read.
    """
    count = words[0] & 0xFFFFFFFFFFFFFFFF if words else 0
    count = min(count, len(words) // 2 - 1)  # word 1 is the drop count
    out = []
    for i in range(1, count + 1):
        w0 = words[i * 2] & 0xFFFFFFFFFFFFFFFF
        ts = words[i * 2 + 1] & 0xFFFFFFFFFFFFFFFF
        out.append((w0 & 0xFFFF, (w0 >> 16) & 0xFFFF, (w0 >> 32) & 0xFFFFFFFF, ts))
    return out


def resolve_tracr_channel(meta: dict) -> int:
    """Index of the lane the AICore payloads belong on.

    channel_names is sized by the run's core count, so this is resolved by name
    per run rather than assumed.
    """
    names = meta["channel_names"]
    if TRACR_CHANNEL_NAME not in names:
        raise RuntimeError(f"TraCR channel {TRACR_CHANNEL_NAME!r} absent from this proc's channel_names")
    return names.index(TRACR_CHANNEL_NAME)


def verify_tracr_events(meta: dict) -> None:
    """Fail loudly if the kernel's hard-coded event ids have drifted."""
    marker_types = [meta["markerTypes"][k] for k in sorted(meta["markerTypes"])]
    for event_id, want in ((TRACR_EVENT_PHASE2, "Phase2"), (TRACR_EVENT_BARRIER, "Barrier")):
        got = marker_types[event_id] if event_id < len(marker_types) else None
        if got != want:
            raise RuntimeError(
                f"TraCR event id drift: kernel writes {event_id} for {want!r}, metadata has {got!r}. "
                f"Update the kTracrEvent* constants in allreduce_onephase_kernel.cpp."
            )


def write_aicore_bts(host_tracr, device_ids: list[int]) -> int:
    """Serialize each rank's AICore payloads as a .bts lane in its device proc.

    The device proc dirs are written by the runtime during run(); this appends a
    thread folder beside them, exactly as HostCopyTraces2BTS does for the host
    copy lanes. Returns the number of lanes written.
    """
    base = os.path.join(os.path.expanduser("~/ascend"), f"tracr_{os.environ.get('PYPTO_RUN_SAMPLE_ID', '0')}")
    written = 0
    for rank, device_id in enumerate(device_ids):
        proc_dir = os.path.join(base, f"proc.{1000 + device_id}")
        meta_path = os.path.join(proc_dir, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        payloads = decode_payloads(host_tracr[rank].tolist())
        if not payloads:
            continue
        with open(meta_path) as fh:
            meta = json.load(fh)
        verify_tracr_events(meta)
        channel = resolve_tracr_channel(meta)
        used = [n for n in os.listdir(proc_dir) if n.startswith("thread.") and n[7:].isdigit()]
        thread_dir = os.path.join(proc_dir, f"thread.{max((int(n[7:]) for n in used), default=0) + 1}")
        os.makedirs(thread_dir, exist_ok=True)
        with open(os.path.join(thread_dir, "traces.bts"), "wb") as fh:
            fh.write(b"".join(TRACR_PAYLOAD_FMT.pack(channel, ev, ex, ts) for _c, ev, ex, ts in payloads))
        written += 1
    return written


def report_barrier_timing(host_tracr, nranks: int) -> None:
    """Print the Phase-2 barrier cost per rank, decoded from the AICore payloads."""
    print("[allreduce] Phase-2 barrier, decoded from AICore TraCR payloads:")
    for rank in range(nranks):
        payloads = decode_payloads(host_tracr[rank].tolist())
        if not payloads:
            print(f"  rank {rank}: no payloads recorded")
            continue
        spans, open_span = [], None
        for _chan, event, extra, ts in payloads:
            if event == TRACR_EVENT_RESET:
                if open_span is not None:
                    spans.append((open_span[0], open_span[1], (ts - open_span[2]) / 1000.0))
                    open_span = None
            else:
                open_span = (event, extra, ts)
        notify = "  ".join(f"notify={d:.1f}us" for e, _x, d in spans if e == TRACR_EVENT_PHASE2)
        waits = "  ".join(f"peer{x}={d:.1f}us" for e, x, d in spans if e == TRACR_EVENT_BARRIER)
        dropped = host_tracr[rank].tolist()[1] & 0xFFFFFFFFFFFFFFFF
        drop_note = f"  [{dropped} DROPPED - buffer too small]" if dropped else ""
        print(f"  rank {rank}: {len(payloads)} payloads / {len(spans)} spans  |  {notify}  {waits}{drop_note}")


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
    host_tracr = [torch.zeros(TRACR_SLOTS, dtype=torch.int64).share_memory_() for _ in range(nranks)]

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
                        worker.make_tensor_arg(host_tracr[i], shapes=(TRACR_SLOTS,), dtype=_I64),
                        TensorArgType.OUTPUT_EXISTING,
                    )
                    chip_args.add_scalar(domain.domain_size)
                    chip_args.add_scalar(domain.device_ctx)
                    args_list.append(chip_args)
                orch.submit_next_level_group(chip_handle, args_list, cfg, workers=list(range(nranks)))

        print(f"[allreduce] running {nranks}-chip allreduce DAG...")
        worker.run(orch_fn, args=None, config=CallConfig())

        report_barrier_timing(host_tracr, nranks)
        lanes = write_aicore_bts(host_tracr, device_ids)
        if lanes:
            print(f"[allreduce] wrote {lanes} AICore TraCR lane(s) into the device procs")

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
