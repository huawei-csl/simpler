# Running the aSim cases

The two cases the campaign tracks, how to run each arm, and the measurement
discipline the numbers depend on. Results live in
[validation.md](validation.md); the structure under test is in
[device-model.md](device-model.md).

Both cases run on **simulated-device platform variants**, which are ordinary
builds with the AICore replaced by an AICPU-hosted model:

| platform | runtime | role |
| -------- | ------- | ---- |
| `a2a3asim` | `host_build_graph` | **M0** — the baseline scheduler |
| `a2a3asimgq` | `group_queue` | **M2** — the GroupQueue under evaluation |

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

```bash
export SIMPLER_ASIM_CALIB=$PWD/docs/asim/calib/qwen_decode_fwd.calib
export PTO2_MANUAL_MAX_SEQ=3338 PYPTO_TASK_WINDOW=65536
python models/qwen3/14b/decode_fwd.py -p a2a3 -d $TASK_DEVICE \
    --validate-fwd --fwd-layers 40 --max-seq
```

Two things the pypto path does not do for itself:

- **It passes its own platform** to the executor, so patching `_platform_string`
  is not enough to reach the simulated build — the executor call has to be
  wrapped. Without that the run silently uses the onboard binaries.
- **It does not raise the device-side log level**, so `[GQ_WORK]` and the other
  device counters never appear. Host-side `configure_logging` does not reach
  them.

The in-repo `examples/a2a3/tensormap_and_ringbuffer/qwen3_14b_decode` with
`decode_fwd_layers_expanded.cpp` is a second, independent path to the same model
and measures within 0.6 points of the pypto one. Use it when the cross-repo
setup is not worth standing up. Its shipped (non-expanded) orchestration submits
each layer as a Graph, which `group_queue` cannot expand — pass the expanded
source.

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
| `[ASIM_WORK]` | M0 | compute issued, dispatches, poll overrun |
| `[GQ_WORK]` | M2 | compute issued by shape, look-ahead depth, poll and push overrun, steal counters |
| `[M0_DONE]` / `[M2_DONE]` | both | tasks completed against tasks total |
| `[M0_HIST]` / `[M2_HIST]` | both | compute samples per `func_id` |
| `[GQ_CALIB]` | M2 | share of tasks whose duration came from the calibration rather than a default |

Compare compute issued across the arms before trusting any delta. On the two
cases it matches to 0.08 % and 0.005 %.

## Measurement discipline

These are not style preferences; each one was learned from a wrong number.

- **Run both arms back to back inside one `task-submit`.** Three consecutive
  runs of one binary agree to 0.55 %, but the same build an hour later differs by
  ~3 %. A cross-submission A/B cannot resolve anything below that.
- **To A/B a code change, build both binaries first and swap the `.so` between
  runs** inside a single submission, rather than rebuilding between submissions.
  Copy it to **both** load locations — `build/lib/...` and the installed
  `.venv/.../simpler_setup/_assets/build/lib/...`.
- **Instrumentation is not free.** Adding the overrun and poll-cost counters to
  `[GQ_WORK]` cost ~1.3 % of `device_wall`, so an instrumented build is its own
  baseline and must not be compared against an uninstrumented one.
- **Read `[GQ_WORK] poll_overrun_us` before quoting an M2 number.** The
  simulator overruns the latency it models whenever its own work costs more than
  the access it stands for, and that excess lands inside the measured window.
  See the audit in [validation.md](validation.md).

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
