"""Real parallelism + per-type makespan from chip_swimlane_records.json."""
import json
import sys
from collections import defaultdict

d = json.load(open(sys.argv[1]))
ns = 1e9 / d["metadata"]["clock_freq_hz"]
core_types = d["metadata"]["core_types"]

# aicore_tasks: [core_id, token, reg_task_id, start_c, end_c, recv_c]
by_type_cores = defaultdict(set)
by_type_tasks = defaultdict(int)
by_type_start = defaultdict(lambda: float("inf"))
by_type_end = defaultdict(lambda: 0)
busy_by_core = defaultdict(float)
tasks_by_core = defaultdict(int)
ctype_of_core = {}
for core_id, tok, rtid, s, e, r in d["aicore_tasks"]:
    ct = core_types[core_id]
    by_type_cores[ct].add(core_id)
    by_type_tasks[ct] += 1
    by_type_start[ct] = min(by_type_start[ct], s)
    by_type_end[ct] = max(by_type_end[ct], e)
    busy_by_core[core_id] += (e - s) * ns
    tasks_by_core[core_id] += 1
    ctype_of_core[core_id] = ct

for ct in sorted(by_type_cores):
    ncores = len(by_type_cores[ct])
    span = (by_type_end[ct] - by_type_start[ct]) * ns
    ntasks = by_type_tasks[ct]
    print(f"{ct}: cores_used={ncores} tasks={ntasks} tasks/core~{ntasks/ncores:.1f} "
          f"span(min_start->max_end)={span/1000:.1f}us")
    # per-core busy occupancy fraction
    busies = [busy_by_core[c] for c in by_type_cores[ct]]
    print(f"    per-core busy(sum end-start): mean={sum(busies)/len(busies)/1000:.1f}us "
          f"max={max(busies)/1000:.1f}us (of span {span/1000:.1f}us)")
