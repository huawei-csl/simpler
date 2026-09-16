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
