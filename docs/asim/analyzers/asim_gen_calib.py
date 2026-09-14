"""Generate an aSim per-test compute-calibration table from a level>=2
chip_swimlane_records.json + the test's deps.json.

Emits one `func_id mean_ns sigma_ns` line per kernel, the format Runtime loads
via SIMPLER_ASIM_CALIB. The sigma makes aSim draw a per-task duration so cores
finish scattered like real ones rather than in lockstep batches.

  usage: asim_gen_calib.py <swimlane.json> <deps.json|-> [out.calib]
`deps.json` is only consulted when the swimlane carries no recorded
`aicpu_task_func_ids`; pass `-` when there is no dep-gen round.
A task's func_id is the first non-negative entry of its deps kernel_ids (the slot
the scheduler dispatches). Tasks with multiple kernels (MIX) are not yet split.
"""
import json
import sys
import statistics as st
from collections import defaultdict

swim = json.load(open(sys.argv[1]))
try:
    deps = json.load(open(sys.argv[2]))
except (OSError, ValueError):
    deps = {"tasks": []}
out = sys.argv[3] if len(sys.argv) > 3 else None

ns = 1e9 / swim["metadata"]["clock_freq_hz"]
# token -> compute ns (end - start). The receive->start setup is modeled by the
# ack latency, not folded into compute, so it is not double-counted in the
# 2-deep pipeline (it overlaps the previous task's compute).
tok_occ = {}
for core, tok, rtid, s, e, r in swim["aicore_tasks"]:
    tok_occ[int(tok)] = (e - s) * ns

# task_id -> func_id (first non-negative kernel_id)
def func_of(t):
    for k in t.get("kernel_ids", []):
        if k >= 0:
            return k
    return -1

by_func = defaultdict(list)
# Preferred source: the func_id the scheduler recorded per completed task
# (`aicpu_task_func_ids`, parallel to `aicpu_tasks`). deps.json cannot supply it
# for a graph-execution case — those tasks are expanded on device and have no
# deps node — so fall back to the deps join only when the array is absent.
rec_funcs = swim.get("aicpu_task_func_ids") or []
# Newer traces carry the func_id inside each scheduler record (5th field) rather
# than as a parallel array. That is the only source that works for a
# graph-execution case, whose tasks have no deps node to join against.
if not rec_funcs:
    _rows = (swim.get("scheduler_tasks") or {}).get("records") or []
    if _rows and len(_rows[0]) >= 5:
        rec_funcs = [int(r[4]) for r in _rows]
# The AICPU-side per-task rows moved under a versioned wrapper when the chip
# swimlane grew a schema version: `scheduler_tasks.records` carries the same
# [core_id, reg_task_id, dispatch_ts, finish_ts] shape the flat `aicpu_tasks`
# list used to. Read whichever the trace carries, so one generator serves both.
def _aicpu_rows(sw):
    flat = sw.get("aicpu_tasks")
    if flat:
        return flat
    return (sw.get("scheduler_tasks") or {}).get("records") or []

aicpu_rows = _aicpu_rows(swim)
aicore_rows = swim.get("aicore_tasks") or []
if rec_funcs and len(rec_funcs) == len(aicpu_rows):
    # Join AICore timing to the AICPU row by (core_id, reg_task_id).
    occ_by_key = {}
    for core_id, tok, rtid, s_, e_, r_ in aicore_rows:
        occ_by_key[(int(core_id), int(rtid))] = (e_ - s_) * ns
    for row, fid in zip(aicpu_rows, rec_funcs):
        key = (int(row[0]), int(row[1]))
        if int(fid) >= 0 and key in occ_by_key:
            by_func[int(fid)].append(occ_by_key[key])
else:
    if not deps.get("tasks"):
        sys.exit("no recorded aicpu_task_func_ids and no deps.json: cannot resolve func_id")
    for t in deps["tasks"]:
        tid = int(t["task_id"])
        if tid in tok_occ:
            by_func[func_of(t)].append(tok_occ[tid])

lines = []
for fid in sorted(by_func):
    if fid < 0:
        continue
    mean = st.mean(by_func[fid])
    # Robust (MAD-based) spread, not the raw stdev: the observed per-task
    # duration is right-skewed because its tail is scheduling interference
    # rather than compute variance, and feeding that tail in as compute sigma
    # over-fragments the simulated completions.
    _v = by_func[fid]
    _med = st.median(_v)
    _mad = st.median([abs(x - _med) for x in _v])
    sigma = 1.4826 * _mad if len(_v) > 1 else 0.0
    lines.append((fid, round(mean), round(sigma)))
    print(f"func_id={fid} n={len(by_func[fid])} mean_ns={mean:.0f} sigma_ns={sigma:.0f} "
          f"median={st.median(by_func[fid]):.0f} CV={100*sigma/mean:.1f}%")

# Injected MMIO latencies (ns): push, read, ack. `push` is the posted doorbell
# write and `read` the in-situ COND poll (DESIGN.md 5a). `ack` spans the whole
# push -> kernel-start path and is the sum of two independently measured,
# sequential costs: AICPU->core propagation (the 633 ns latch less the 5 ns
# posted write) plus this case's own AICore-side dcci+ack, carried per task as
# receive_to_start_cycles. Compute is end-start and excludes that setup, so
# leaving it out of `ack` drops it from the model entirely.
# Override via env ASIM_LAT="p r a".
import os  # noqa: E402
_r2s = [r[5] * ns for r in swim.get("aicore_tasks", [])]
_setup = round(sum(_r2s) / len(_r2s)) if _r2s else 0
lat = os.environ.get("ASIM_LAT", f"5 195 {628 + _setup}")
print(f"receive_to_start mean={_setup} ns -> ack={628 + _setup} ns")

# FIN-write -> AICPU-observable floor, measured per case as a low percentile of
# (aicpu finish_time - aicore end_time). The bulk of that spread is the wait for
# the AICPU's next poll, which the simulated scheduler reproduces on its own, so
# only the floor is a hardware property; using the median would double-count.
_notice = 0
_end_by = {}
for _c, _t, _rt, _s, _e, _r in swim.get("aicore_tasks", []):
    _end_by[(int(_c), int(_rt))] = _e
_d = []
for _row in _aicpu_rows(swim):
    _k = (int(_row[0]), int(_row[1]))
    if _k in _end_by:
        _v = (_row[3] - _end_by[_k]) * ns
        if 0 <= _v < 1e8:
            _d.append(_v)
if _d:
    _d.sort()
    _notice = round(_d[int(0.01 * (len(_d) - 1))])   # p1 = hardware floor
    print(f"notice_floor_ns={_notice} (p1 of finish-end, n={len(_d)})")

if out:
    with open(out, "w") as f:
        f.write(f"LAT {lat} {_notice}\n")
        for fid, ns_, sd_ in lines:
            f.write(f"{fid} {ns_} {sd_}\n")
    print(f"wrote {len(lines)} entries (LAT {lat}) -> {out}")
