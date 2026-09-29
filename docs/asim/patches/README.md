# Patches this work depends on, that live in another repo

The GroupQueue runtime reads a grouping the *graph* declares. For a pypto-built
graph that declaration has to be emitted by pypto's orchestration codegen, which
is not this repository — so the patch is kept here, next to the runtime that is
useless without it.

Nothing here is applied automatically. A tree without the patch applied still
builds and runs: pypto emits no `rt_group_begin` / `rt_group_end`, every graph
arrives undeclared, and the GroupQueue runs it exactly as an ungrouped one. That
is the failure mode to recognise — no error, no warning, just `groups=0` in the
`[GQ_GROUP]` line and M2 quietly measuring something else.

## `pypto-declare-group-per-parallel-loop.patch`

Emits the declaration around each `pl.parallel` iteration: a Parallel loop states
its iterations are independent, which is what a task group declares. Whether an
iteration also holds an edge of its own is decided in the orchestrator, which
drops a declaration set that holds none.

Against pypto `6beb9763`:

```bash
cd <pypto checkout>
git apply <this dir>/pypto-declare-group-per-parallel-loop.patch
cmake --build build --target pypto_core
cp build/python/bindings/pypto_core.cpython-*.so <venv>/lib/python3.*/site-packages/pypto/
```

Verify against a case whose groups are known to carry edges — deepseek v4
`expert_routed` should report `groups=50 ... deps_named=225`, and `deps_named=0`
anywhere means the declaration is not arriving.

## The revisions the qwen case was measured at

Reproducing the qwen number needs all four, and none of them is pinned by a
manifest — this table is the record.

| component | revision | note |
| --------- | -------- | ---- |
| pypto | `6beb9763` | 2026-09-14. Its `runtime` submodule is replaced by a symlink to the simpler worktree under test. |
| pypto-lib | `9931cac` | 2026-07-18. Paths are `models/qwen3/14b/`; pypto-lib flattened these to `models/qwen3_14b/` in `b607072` (2026-08-04), after this revision. |
| pto-isa | `5a4f74cb` | the repo's `pto_isa.pin`. |
| simpler | `e322b731` | tag `asim-gq-checkpoint-2026-09-25`. |

pypto-lib here is about two months older than pypto. That pairing is not
declared by either repository; it is what was installed and working. The two
shims in `pypto_run_wrapper.py` are symptoms of that gap and are expected to
move, not to hold: `enable_l2_swimlane` no longer exists at pypto-lib HEAD.

## The other three files

- [`pypto-lib-9931cac-bypass-sampling.diff`](pypto-lib-9931cac-bypass-sampling.diff)
  — `pl.arange` lowers to a two-operand `pto.tci` the pinned ptoas cannot parse,
  which is the only thing blocking `--validate-fwd`. The sampling kernel is
  bypassed so the multi-layer graph compiles. It contributes no scheduling
  structure, so the N-layer + LM-head graph is intact; the sampled ids are wrong
  by construction and the simulated device computes nothing anyway.
- [`pypto_run_wrapper.py`](pypto_run_wrapper.py) — the entry point the runs
  actually used. **Patching `_platform_string` alone is not enough**: pypto-lib
  passes its own platform to the executor, so without this wrapper the run
  silently loads the onboard binaries. It also forces device and platform onto
  `RunConfig`, renames the runtime through `PYPTO_RUNTIME_NAME`, and shims
  `_DfxOpts(enable_l2_swimlane=)` onto `DfxOptions(enable_chip_swimlane=)`.
- Its companion inner script is not kept here; the invocation it builds is in
  [../running.md](../running.md).

## Verifying the declaration actually arrived

The README above says a tree without the patch measures an ungrouped graph with
no warning. The counter that proves it — `groups=` in `[GQ_GROUP]` — is **not
readable on the pypto path**: `ChipWorker.init()` forwards a snapshot of the
`simpler` logger's level to the device, and neither `configure_logging` nor
setting that logger directly has been made to reach it, so the device log
carries no `[GQ_*]` lines at all.

Check the **generated orchestration** instead, which is decisive and needs no
device:

```bash
grep -c rt_group_begin build_output/_jit_*/orchestration/*.cpp
```

For qwen-40L at these revisions that is **6**, so the declarations are emitted.
Whether the runtime then resolved or dropped them is a separate question the
generated source cannot answer — see [../validation.md](../validation.md).
