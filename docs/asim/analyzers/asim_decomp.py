"""Decompose the sched makespan into per-task lifecycle components from a
level-3 chip_swimlane_records.json.
  aicore_tasks: [core_id, task_token, reg_task_id, start_c, end_c, recv_c]
  aicpu_tasks:  [core_id, reg_task_id, dispatch_c, finish_c]
Lifecycle: dispatch -> receive(=start-recv) -> start -> end -> finish(FIN observed)
  submit    = receive - dispatch
  recv2start= recv_c
  compute   = end - start
  notice    = finish - end
"""
import json
import sys
import statistics as st
from collections import defaultdict

d = json.load(open(sys.argv[1]))
ns = 1e9 / d["metadata"]["clock_freq_hz"]
ctypes = d["metadata"]["core_types"]

aic = {}  # (core,rtid) -> (start,end,recv, core_type)
for core, tok, rtid, s, e, r in d["aicore_tasks"]:
    aic[(core, rtid)] = (s, e, r, ctypes[core])
acp = {}  # (core,rtid) -> (dispatch, finish)
for core, rtid, disp, fin in d["aicpu_tasks"]:
    acp[(core, rtid)] = (disp, fin)

comp = defaultdict(lambda: defaultdict(list))  # core_type -> component -> [ns]
per_core = defaultdict(list)  # core_id -> [(dispatch,start,end,finish)]
for key, (s, e, r, ct) in aic.items():
    if key not in acp:
        continue
    disp, fin = acp[key]
    receive = s - r
    comp[ct]["submit"].append((receive - disp) * ns)
    comp[ct]["recv2start"].append(r * ns)
    comp[ct]["compute"].append((e - s) * ns)
    comp[ct]["notice"].append((fin - e) * ns)
    comp[ct]["aicpu_seen(disp->fin)"].append((fin - disp) * ns)
    per_core[key[0]].append((disp, s, e, fin))

def line(xs):
    xs = sorted(xs)
    return f"n={len(xs)} mean={st.mean(xs):.0f} median={st.median(xs):.0f} min={xs[0]:.0f} max={xs[-1]:.0f}"

print("=== per-task lifecycle components (ns), by core_type ===")
for ct in sorted(comp):
    print(f"[{ct}]")
    for name in ["submit", "recv2start", "compute", "notice", "aicpu_seen(disp->fin)"]:
        print(f"   {name:22s} {line(comp[ct][name])}")

# Critical path: busiest core (max span). Decompose span into compute vs gaps.
print("\n=== critical-path decomposition (busiest AIC core) ===")
best = None
for core, evs in per_core.items():
    if ctypes[core] != "aic":
        continue
    evs.sort(key=lambda x: x[1])  # by start
    span = (evs[-1][2] - evs[0][1]) * ns  # last end - first start
    if best is None or span > best[1]:
        best = (core, span, evs)
core, span, evs = best
busy = sum((e - s) for _, s, e, _ in evs) * ns
gaps = sum((evs[i + 1][1] - evs[i][2]) for i in range(len(evs) - 1)) * ns  # start_{n+1}-end_n
ntasks = len(evs)
print(f"core={core} tasks={ntasks} span={span/1000:.1f}us  compute_busy={busy/1000:.1f}us "
      f"gaps={gaps/1000:.1f}us  gap/task={gaps/max(1,ntasks-1):.0f}ns")

print("\n=== scheduler phases (level 3) ===")
ph = defaultdict(float); phn = defaultdict(int)
for rec in d.get("aicpu_scheduler_phases", []):
    ph[rec["kind"]] += (rec["end_cycles"] - rec["start_cycles"]) * ns
    phn[rec["kind"]] += 1
for k in sorted(ph):
    print(f"   {k:12s} total={ph[k]/1000:.1f}us records={phn[k]}")
