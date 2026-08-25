# `scan_and_claim` runtime (a2a3)

A third runtime alongside `host_build_graph` (hbg) and
`tensormap_and_ringbuffer` (tmr), selected with
`@scene_test(runtime="scan_and_claim")`.

## Status: M1 — copy only

This tree started as a **verbatim copy of `host_build_graph`** (61 files). The
only intentional differences are:

| File | Change |
|---|---|
| `build_config.py` | header rewritten to describe this runtime |
| `host/runtime_compile_info.cpp` | `get_orchestration_compiler()` returns `TOOLCHAIN_HOST_GXX` unconditionally |

Everything else is byte-identical, deliberately — `diff -r` against
`../host_build_graph/` is the intended way to review this tree. **The scheduler
is still hbg's, unchanged**, so behaviour is currently identical to hbg.

On `runtime_compile_info.cpp`: hbg's copy is byte-identical to tmr's, including
a `TOOLCHAIN_AARCH64_GXX` branch for a2a3 that contradicts the host toolchain
`KernelCompiler` actually selects for a host-orchestrated runtime. Nothing calls
the C++ hook (`simpler_setup/kernel_compiler.py::_orchestration_toolchain` is
the live path), so it is harmless in hbg — but it is not worth propagating.

## What is meant to diverge

Task **discovery**, and nothing else. The front end is shared wholesale: the
host runs orchestration to completion, relocates pointers, and H2Ds the image;
the device boots scheduler-only with the whole graph resident.

The plan is to replace push-based discovery — wake lists, ready queues, and the
dedicated resolution thread (3S+1P) — with a **bounded scan** over a
task-id-indexed state array (`PENDING` / `BUSY` / `READY` / `RETIRED`) plus a
single monotonic cursor marking the contiguous retired prefix. Readiness is
*derived* from completion flags on each scan rather than pushed at completion,
which makes lost wakeups impossible by construction and removes the startup
`classify_partition` seeding pass. The change is confined to
`runtime/scheduler/`.

## Do not rename these

`build_config.py`'s `"../../../common/host_build_graph"` source dir and the
`#include "host_build_graph/{graph_execution,graph_cache,graph_host_state,self_relative_ptr}.h"`
lines in `runtime/` resolve to **`src/common/host_build_graph/`** — the shared
host-state / recorded-graph layer, reached because `src/common` is an include
root. Sharing it is deliberate; renaming forks it by accident and breaks the
build.

Inherited prose comments still say `host_build_graph` in many places. That is
intentional at M1: they describe code that has not changed yet, and rewriting
them would destroy the diff-against-hbg property. They get rewritten alongside
the code they describe.
