#!/usr/bin/env python3
"""Run a pypto example against this worktree's simpler on an allocated device.

The bundled examples hardcode `device_id` (and sometimes a platform), so RunConfig
is wrapped before the example imports it: the device must be the one task-submit
granted, not whatever the file names.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

PYPTO_EXAMPLES = Path("/mount_home/simpler/tmp_pypto_main/examples")
DEV = int(os.environ["PYPTO_DEVICE"])
PLAT = os.environ.get("PYPTO_PLATFORM", "a2a3")

import pypto.runtime as _rt  # noqa: E402

# The device-side [GQ_WORK] counters are emitted at info; the pypto path never
# raises the level on its own.
# ChipWorker.init() forwards a snapshot of the "simpler" logger's effective
# level to the device, so the device-side [GQ_*] counters need that logger at
# INFO -- configuring simpler_setup's logging alone does not reach them.
import logging as _logging  # noqa: E402
_logging.getLogger("simpler").setLevel(_logging.INFO)

# Patch the constructor rather than the class: pypto does
# `isinstance(run_config, RunConfig)`, so replacing the name breaks it.
_orig_init = _rt.RunConfig.__init__


# pypto's platform vocabulary is {a2a3, a2a3sim, a5, a5sim}; the simulated-device
# variants are simpler's, not pypto's. _platform_string is the single place the
# wire spelling is produced, so overriding it there is what lets a pypto program
# run on an aSim build without teaching pypto a platform it does not own.
SIMPLER_PLATFORM = os.environ.get("PYPTO_SIMPLER_PLATFORM")
if SIMPLER_PLATFORM:
    import pypto.runtime.runner as _runner

    _runner._platform_string = lambda arch, mode: SIMPLER_PLATFORM

    # Codegen validates the same string but wants the real arch: an aSim build
    # runs a2a3 kernels, it only replaces the cores underneath them. Map the
    # simulated-device spelling onto the arch it is a variant of.
    import pypto.ir.compile  # noqa: F401  (the name resolves to a function; take the module)

    _compile = sys.modules["pypto.ir.compile"]
    _orig_backend_for = _compile._backend_type_for_platform

    def _backend_for(platform, fallback):
        if platform == SIMPLER_PLATFORM:
            platform = "a5" if SIMPLER_PLATFORM.startswith("a5") else "a2a3"
        return _orig_backend_for(platform, fallback)

    _compile._backend_type_for_platform = _backend_for

    # Same for the kernel compiler's platform -> arch-directory mapping. Deriving
    # it by prefix covers every simulated-device variant rather than one spelling.
    from pypto.runtime.kernel_compiler import KernelCompiler as _KC

    def _arch(self):
        for a in ("a2a3", "a5"):
            if str(self.platform).startswith(a):
                return a
        raise ValueError(f"Unknown platform: {self.platform}")

    _KC._arch = _arch

# group_queue is not in pypto's RuntimeKind (a C++ enum: HOST_BUILD_GRAPH,
# TENSORMAP_AND_RINGBUFFER). The kind only ever becomes a string at
# runtime_kind_to_name, and that string is what selects kernel includes and the
# runtime binary — so renaming it there is what lets a pypto program target a
# runtime pypto does not know about.
RUNTIME_NAME = os.environ.get("PYPTO_RUNTIME_NAME")
if RUNTIME_NAME:
    # Several modules bind the name with `from ... import`, so the binding has to
    # be replaced in each holder; the source module covers the qualified callers.
    import importlib

    def _renamed(kind):
        return RUNTIME_NAME

    for _name in (
        "pypto.pypto_core.passes",
        "pypto.jit.decorator",
        "pypto.jit.cache",
        "pypto.runtime.worker",
        "pypto.runtime.device_runner",
        "pypto.backend.pto_backend",
    ):
        try:
            _mod = importlib.import_module(_name)
        except Exception:  # noqa: BLE001 - a module that will not import cannot hold the name
            continue
        if hasattr(_mod, "runtime_kind_to_name"):
            _mod.runtime_kind_to_name = _renamed

# pypto-lib's harness asks for DFX as `runner._DfxOpts(enable_l2_swimlane=...)`;
# this pypto calls the type `DfxOptions` and the field `enable_chip_swimlane`. The
# harness catches only the ImportError, so the rename does not fail the run -- it
# drops every DFX flag silently and the case completes with no records written.
# Binding the old name to a translating factory is what lets the harness's own
# --enable-l2-swimlane reach the runtime.
try:
    import pypto.runtime.runner as _runner_mod

    if not hasattr(_runner_mod, "_DfxOpts") and hasattr(_runner_mod, "DfxOptions"):

        def _dfx_opts_compat(**kw):
            level = kw.pop("enable_l2_swimlane", None)
            if level is not None:
                kw["enable_chip_swimlane"] = level
            return _runner_mod.DfxOptions(**kw)

        _runner_mod._DfxOpts = _dfx_opts_compat
except Exception:  # noqa: BLE001 - the shim is best-effort; a pypto without it runs unchanged
    pass

# pypto-lib's harness passes its own argparse platform straight to the executor,
# and a case dispatches exactly once. Both are handled here: the platform the
# simulated build needs, and a repeat so the measured invocation is a steady-state
# one. A single dispatch is a cold run, and on a simulated platform that charges it
# the one-time simulated-device bring-up real silicon pays at power-on -- which on a
# ~1 ms case is the whole difference. The wrapper applies on every platform so the
# two sides are compared at the same warmth.
import pypto.runtime.device_runner as _dr  # noqa: E402

_orig_exec_on_device = _dr._execute_on_device
_rounds = max(1, int(os.environ.get("PYPTO_ROUNDS") or "1"))


def _exec_on_device(chip_callable, orch_args, platform, *a, **kw):
    target = SIMPLER_PLATFORM or platform
    out = None
    for _ in range(_rounds):
        out = _orig_exec_on_device(chip_callable, orch_args, target, *a, **kw)
    return out


_dr._execute_on_device = _exec_on_device

SWIMLANE = os.environ.get("PYPTO_SWIMLANE")
TASK_WINDOW = os.environ.get("PYPTO_TASK_WINDOW")
DEPGEN = os.environ.get("PYPTO_DEPGEN")


def _forced_init(self, *args, **kwargs):
    kwargs["device_id"] = DEV
    # pypto-lib's own harness does not route the platform through
    # `_platform_string`, so overriding that alone leaves the run on the onboard
    # binaries. Naming the simulated platform on the config is what actually picks
    # the aSim build; the codegen patches above are what let pypto accept it.
    kwargs["platform"] = PLAT
    if TASK_WINDOW:
        # host_build_graph holds the whole graph at once and reclaims nothing
        # mid-run, so a graph larger than the default window needs a bigger one;
        # tensormap_and_ringbuffer reclaims and does not.
        kwargs["ring_task_window"] = int(TASK_WINDOW)
    if SWIMLANE:
        kwargs["enable_chip_swimlane"] = int(SWIMLANE)
    if DEPGEN:
        kwargs["enable_dep_gen"] = True
    _orig_init(self, *args, **kwargs)


_rt.RunConfig.__init__ = _forced_init

PYPTO_LIB = Path("/mount_home/simpler/tmp_pypto_lib_real")

if __name__ == "__main__":
    rel = sys.argv[1]
    # A pypto-lib case is a standalone script with its own argv, so anything after
    # the path goes to it. "lib:" names the pypto-lib tree; a bare name stays a
    # pypto example, which is what every earlier run was.
    extra = sys.argv[2:]
    if rel.startswith("lib:"):
        root = PYPTO_LIB
        path = PYPTO_LIB / rel[len("lib:") :]
    else:
        root = PYPTO_EXAMPLES.parent
        path = PYPTO_EXAMPLES / rel
    # Several pypto-lib cases build their inputs with unseeded torch RNG, and a
    # data-dependent trip count (deepseek's per-expert token counts) then makes the
    # task graph itself differ run to run. Comparing a simulated run against a real
    # one requires the same graph, so the seed is fixed here for every run.
    import torch as _torch

    _seed = int(os.environ.get("PYPTO_SEED", "1234"))
    _torch.manual_seed(_seed)
    print(f"[pypto-ex] {rel} platform={PLAT} device={DEV} seed={_seed} args={extra}", flush=True)
    sys.argv = [str(path)] + extra
    sys.path.insert(0, str(root))
    # A pypto-lib case imports its siblings by bare name (`from config import ...`),
    # which only resolves with its own directory on the path -- running it by path
    # gives it no package context of its own.
    sys.path.insert(0, str(path.parent))
    # The runtime is not a RunConfig field: it comes from the active PassContext,
    # and pypto defaults to tensormap_and_ringbuffer. Selecting it has to wrap the
    # whole call, because ir.compile() inherits the context that is current then.
    want = os.environ.get("PYPTO_RUNTIME", "")
    if want:
        from pypto.pypto_core import passes as _passes

        kind = getattr(_passes.RuntimeKind, want.upper())
        print(f"[pypto-ex] runtime={want}", flush=True)
        with _passes.PassContext([], runtime=kind):
            runpy.run_path(str(path), run_name="__main__")
    else:
        runpy.run_path(str(path), run_name="__main__")
