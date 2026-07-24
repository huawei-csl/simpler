# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyTraCR markers for the L3 host-orchestration scheduling timeline.

The ``tracr`` module (PyTraCR) exists only when simpler is built with
``BUILD_TRACR=ON``; when absent every function here is a no-op, so orchestration
code may call them unconditionally. All markers record on the orchestrator's
main thread (channel 0). At finalize the flushed ``proc.<pid>`` folder is moved
beside the device-side ``proc.<1000+device_id>`` folders under
``~/ascend/tracr_<PYPTO_RUN_SAMPLE_ID>/`` so a single ``tracr_process`` run
merges the host scheduling lane with the per-device lanes.
"""
import json
import os
import shutil

try:
    import tracr as _t

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

# The orchestrator main thread is the only PyTraCR-initialized thread, so all
# host markers share this one lane.
_ORCH_CHANNEL = 0
_MARKER_LABELS = ("AllocateDomain", "SubmitNextLevel", "SubmitNextLevelGroup", "Drain")

_started = False
_labels: dict[str, int] = {}
_trace_base = ""


def _base_dir() -> str:
    sample_id = os.getenv("PYPTO_RUN_SAMPLE_ID", "0")
    return os.path.join(os.path.expanduser("~/ascend"), f"tracr_{sample_id}") + os.sep


def active() -> bool:
    return _AVAILABLE and _started


def start() -> None:
    """Create the host proc and register marker labels. Idempotent; must run on
    the orchestrator main thread after ``Worker.init()`` (all forks done)."""
    global _started, _trace_base
    if not _AVAILABLE or _started:
        return
    _trace_base = _base_dir()
    _t.INSTRUMENTATION_TRACE_PATH(_trace_base)
    _t.INSTRUMENTATION_START()
    for name in _MARKER_LABELS:
        _labels[name] = _t.INSTRUMENTATION_MARK_ADD(name)
    _started = True


def end() -> None:
    """Flush the host proc and relocate it beside the device procs."""
    global _started
    if not active():
        return
    _t.INSTRUMENTATION_ADD_CHANNEL_NAMES(["L3_Orchestrator"])
    _t.INSTRUMENTATION_END()
    _started = False
    _relocate_proc()


def _relocate_proc() -> None:
    src = os.path.join(_trace_base, "tracr", f"proc.{os.getpid()}")
    dst = os.path.join(_trace_base, f"proc.{os.getpid()}")
    try:
        if os.path.isdir(src):
            os.makedirs(_trace_base, exist_ok=True)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)
            _anchor_to_device_timeline(os.path.join(dst, "metadata.json"))
    except OSError:
        pass


def _anchor_to_device_timeline(metadata_path: str) -> None:
    # Device procs anchor on the raw hardware-counter timeline (start_time=0);
    # with USE_HW_COUNTER the host reads the same counter, so start_time=0 keeps
    # the host lane on that shared raw timeline instead of being normalized to
    # its own zero (which shifts it far from the device lanes in tracr_process).
    try:
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
        meta["start_time"] = 0
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
    except (OSError, ValueError):
        pass


def mark_set(label: str, extra: int = 0) -> None:
    if not active():
        return
    _t.INSTRUMENTATION_MARK_SET(_ORCH_CHANNEL, _labels[label], int(extra) & 0xFFFFFFFF)


def mark_reset() -> None:
    if not active():
        return
    _t.INSTRUMENTATION_MARK_RESET(_ORCH_CHANNEL)
