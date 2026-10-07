# Running the aSim cases

The two cases the campaign tracks, how to run each arm, and the measurement
discipline the numbers depend on. Results live in
[validation.md](validation.md); the structure under test is in
[device-model.md](device-model.md).

Both cases run on **simulated-device platform variants**, which are ordinary
builds with the AICore replaced by an AICPU-hosted model:

| platform | runtime | role |
| -------- | ------- | ---- |
| `a2a3` | `host_build_graph` | **base** — real silicon, the fidelity reference |
| `a2a3asim` | `host_build_graph` | **M0** — the baseline scheduler |
| `a2a3asimgq` | `group_queue` | **M2** — the GroupQueue under evaluation |

The base arm is the same runtime as M0 on the real device, so the pair measures
the device model alone. Run all three interleaved in one lock when you want the
fidelity figure alongside the delta — the method and the numbers are in
[validation/fidelity.md](validation/fidelity.md).

They are silicon-agnostic in the sense that they compute nothing, but they run
**on a real AICPU** and therefore still take a device. Every invocation goes
through `task-submit` like any other onboard work — see
[running-onboard.md](../../.claude/rules/running-onboard.md).

## Build

```bash
source .venv/bin/activate
python -m simpler_setup.build_runtimes --platforms a2a3asim
python -m simpler_setup.build_runtimes --platforms a2a3asimgq
```

A platform variant builds its own runtime only, so the two are independent: a
change to `src/a2a3/platform/asimgq/` or `src/a2a3/runtime/group_queue/` needs
only the second.

Three things about this build that cost time to rediscover:

- **`pip install .` does not build the simulated-device variants.** They are not
  auto-detected, so the two commands above are the only way they appear under
  `build/lib/`. A stale variant is silent: the run loads whatever `.so` is there.
- **Build in one environment and stay in it.** If the repo is shared between a
  host and a container (a bind mount), building in both leaves files owned by
  different users and the next rebuild fails on permissions. Pick the one that
  carries the CANN toolchain and do every build, install and run there.
- **Which copy of a binary runs depends on where you launch from**, not on which
  copy is newer. There are two: `build/lib/{arch}/{variant}/{runtime}/` and the
  installed one under `.venv/.../simpler_setup/_assets/build/lib/...`. A run
  launched from the worktree root (`python -m pytest`, or a script there) imports
  `simpler_setup` from the source tree, whose `PROJECT_ROOT` is the worktree
  because it has no `_assets/src` -- so it loads `build/lib/`, which is what
  `build_runtimes` writes. The installed copy is read only when `simpler_setup`
  comes from site-packages, and `build_runtimes` does **not** refresh it, so it
  goes stale silently. Check with
  `python -c "import simpler_setup.environment as e; print(e.PROJECT_ROOT)"` from
  the directory you launch in.

PTO-ISA comes from the repo's `pto_isa.pin`. Do not export `PTO_ISA_ROOT` — it is
resolved from the pin, and an ambient path that has drifted off it is now
rejected rather than silently used.

## Case 1 — paged_attention Case1

65,792 tasks in 256 disjoint components. Both arms run the **same graph**; the
only difference is which runtime schedules it.

```bash
export SIMPLER_ASIM_CALIB=$PWD/docs/asim/calib/pa_case1.calib
export SIMPLER_DFX=1 SIMPLER_HOST_STRACE=1 GQ_THREADS=4

# M0
python -m pytest tests/st/a2a3/host_build_graph/paged_attention \
    --platform a2a3asim   --device $TASK_DEVICE \
    --case Case1 --manual include --skip-golden --rounds 14 --log-level INFO

# M2
python -m pytest tests/st/a2a3/group_queue/paged_attention \
    --platform a2a3asimgq --device $TASK_DEVICE \
    --case Case1 --manual include --skip-golden --rounds 14 --log-level INFO

# base — real silicon, same test as M0, no calibration (it computes for real)
python -m pytest tests/st/a2a3/host_build_graph/paged_attention \
    --platform a2a3     --device $TASK_DEVICE \
    --case Case1 --manual include --skip-golden --rounds 14 --log-level INFO
```

`--skip-golden` is required: the simulated device computes nothing, so a golden
comparison can only fail. It also means **a passing run proves nothing about
correctness** — work equivalence is established from the counters below, not
from the test result.

## Case 2 — qwen3-14B `decode_fwd`, 40 layers

This one is a **pypto-lib** case, not an in-repo example, so it needs the
cross-repo setup in
[multi-repo-setup](../../.claude/skills/multi-repo-setup/SKILL.md) — pypto and
pypto-lib cloned, and this worktree's simpler installed against them.

It also needs three things that live outside this repository: pypto's codegen
patch, pypto-lib's sampling bypass, and the wrapper that actually redirects the
run onto the simulated platform. **The measured revisions, all four patches, and
why `_platform_string` alone is not enough are in
[patches/README.md](patches/README.md).** The invocation below is the one the
wrapper builds; running `decode_fwd.py` directly reaches the onboard binaries
instead, with no error.

```bash
export SIMPLER_ASIM_CALIB=<simpler>/docs/asim/calib/qwen_decode_fwd.calib
export PYPTO_TASK_WINDOW=65536 PYPTO_ROUNDS=10 PTO2_MANUAL_MAX_SEQ=3338
export PYPTO_RUNTIME=host_build_graph

# M0
PYPTO_SIMPLER_PLATFORM=a2a3asim \
  python pypto_run_wrapper.py lib:models/qwen3/14b/decode_fwd.py \
      --validate-fwd --fwd-layers 40 --max-seq

# M2
PYPTO_SIMPLER_PLATFORM=a2a3asimgq PYPTO_RUNTIME_NAME=group_queue \
  python pypto_run_wrapper.py lib:models/qwen3/14b/decode_fwd.py \
      --validate-fwd --fwd-layers 40 --max-seq
```

`GQ_THREADS` is unset and defaults to 4. The case's own argparse never sees the
simulated platform — `PA_SUPPORTED_PLATFORMS` would reject it — so the wrapper
carries it in `PYPTO_SIMPLER_PLATFORM` and forces it onto the `RunConfig`.

Two things the pypto path does not do for itself:

- **It passes its own platform** to the executor, so patching `_platform_string`
  is not enough to reach the simulated build — the executor call has to be
  wrapped. Without that the run silently uses the onboard binaries, with no
  error. This is what `pypto_run_wrapper.py` exists for.
- **It does not raise the device-side log level**, so `[GQ_WORK]`, `[GQ_GROUP]`
  and the other device counters never appear. `ChipWorker.init()` forwards a
  snapshot of the `simpler` logger's level, and neither `configure_logging` nor
  setting that logger directly has been made to reach it. **Consequence: the
  work-equivalence and grouping counters below are unavailable on this path** —
  only the scene-test path (case 1) produces them.

The in-repo `examples/a2a3/tensormap_and_ringbuffer/qwen3_14b_decode` with
`decode_fwd_layers_expanded.cpp` is a second, independent path to the same model
and measures within 0.6 points of the pypto one. Use it when the cross-repo
setup is not worth standing up. Its shipped (non-expanded) orchestration submits
each layer as a Graph, which `group_queue` cannot expand — pass the expanded
source.

**The base arm has to take that in-repo path.** The pypto runner picks its own
platform and would need the wrapper rebuilt around a real device, so the
three-arm fidelity run drives the expanded orchestration directly on all three
platforms instead. That also keeps the arms honest: the same 11,087-task
orchestration reaches each of them.

## Reading a run

`device_wall` is the measurement. One marker per invocation, in nanoseconds:

```bash
grep -aoE 'device_wall ts=[0-9]+ dur=[0-9]+' <log> | sed -E 's/.*dur=//'
```

**Drop the first two rounds.** A first dispatch is cold, and the second still
carries warm-up; every figure in validation.md is the median of the rest.

The device log carries the counters that say the two arms did the same work.
Point `ASCEND_PROCESS_LOG_PATH` at the run's own output directory first, or the
log lands in a directory shared with every other user on the box.

| marker | arm | what it settles |
| ------ | --- | --------------- |
| `[ASIM_WORK]` | M0 | compute issued, dispatches, poll and push overrun |
| `[GQ_WORK]` | M2 | compute issued by shape, positions issued, poll and push overrun, look-ahead depth, cube idle by cause, held entries, steals |
| `[M0_DONE]` / `[M2_DONE]` | both | tasks completed against tasks total; M2's also gives how many durations came from the calibration rather than a default, and why the grouping path passed tasks over |
| `[M0_HIST]` / `[M2_HIST]` | both | compute samples per `func_id` |

`[GQ_WORK]`'s `aic_idle_us` is cube-core time spent free, split by what was
waiting at the time: `drain_us`, a sync-start drain holding dispatch;
`intake_us`, cube work in the software ready queue; `stranded_us`, cube work
committed to a peer queue; `none_us`, nothing. A core holding an unreleased cohort
member is not free and is not counted.

**Both arms print the same amount**: one work line per thread and two per run.
These lines are written inside `device_wall`, so an arm that printed more would
carry the difference into its measurement -- M2 once printed three times as many
and paid ~0.36 ms a run for it.

Compare compute issued across the arms before trusting any delta. On the two
cases it matches to 0.08 % and 0.005 % — though for qwen those figures come from
the in-repo expanded orchestration, since the pypto path emits no counters.

## Measurement discipline

These are not style preferences; each one was learned from a wrong number.

- **Run both arms back to back inside one `task-submit`**, and **repeat the
  pair** -- a single pair resolves much less than it looks like it does. The two
  cases are not alike here, and the difference is large (2026-10-07, model
  clock):

  | case | arm | repeatability |
  | ---- | --- | ------------- |
  | qwen-40L | silicon, M0 | 0.6-0.9 % over three interleaved reps |
  | qwen-40L | M2 (`a2a3asimgq`) | 1.5 % over three reps |
  | PA Case1 | silicon, M0 | 0.7-1.7 % over four reps |
  | PA Case1 | **M2 (`a2a3asimgq`)** | **7.6 % over four reps** |

  The spread is a property of the arm and the case together, not of the arm.
  Before the model clock PA's M2 arm spanned 13.7-17.6 %: the simulator's own
  work, which varies run to run, sat inside its window. Taking it out halved the
  spread but did not remove it, so **quote PA from a pooled median of several
  interleaved reps, never from one pair**, and treat a change worth less than
  ~3 % on PA's M2 arm as unmeasured. qwen's arms hold to under 2 %.
- **To A/B a code change, build both binaries first and swap the `.so` between
  runs** inside a single submission, rather than rebuilding between submissions.
  Copy it to the location your launch directory resolves to (above) -- and to
  both, if anything in the submission might run from elsewhere.
- **Instrumentation is not free.** Adding the overrun and poll-cost counters to
  `[GQ_WORK]` cost ~1.3 % of `device_wall`, so an instrumented build is its own
  baseline and must not be compared against an uninstrumented one.
- **`device_wall` excludes the simulator's own work; the raw scheduler window
  does not.** Each simulated device runs its cores on a per-thread model clock --
  real time less the thread's ledger of work beyond the latencies it models --
  and the phase recorder removes the same ledger from `device_wall`, so
  `device_wall` describes the modelled design ([DESIGN.md](DESIGN.md) §3).
  `poll_overrun_us` plus `push_overrun_us` -- in `[GQ_WORK]` for M2, with
  `overrun_us` in place of the first in `[ASIM_WORK]` for M0 -- is each thread's
  ledger for the run. The
  `sched_start..sched_end` lines are real time and include it, so compare
  `device_wall` across arms, never raw windows. Two checks before quoting a
  simulated number: it must not be below the busiest core's summed compute,
  which no schedule can beat; and the threads' ledgers must not diverge by more
  than a few percent of the window, because coupling through that divergence is
  the one effect a per-thread clock does not remove (DESIGN.md §3 has the check
  that showed it small).

## Calibration

A calibration file supplies the injected latencies and the per-`func_id` compute
distribution:

```text
LAT <push> <read> <ack> <notice>     # ns
<func_id> <mean_ns> <sigma_ns>
...
```

The two in-scope cases have theirs checked in under [calib/](calib/). Generate a
new one from a chip-swimlane capture with
`docs/asim/analyzers/asim_gen_calib.py`; the method is in
[calibration.md](calibration.md).
