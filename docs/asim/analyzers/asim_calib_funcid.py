"""Per-func_id AICore compute durations: join deps.json (task_id -> func_id) with
chip_swimlane_records.json (task_token_raw -> occupancy duration)."""
import json
import sys
import statistics as st
from collections import defaultdict

swim = json.load(open(sys.argv[1]))
deps = json.load(open(sys.argv[2]))

freq = swim["metadata"]["clock_freq_hz"]
ns = 1e9 / freq
core_types = swim["metadata"]["core_types"]

# task_token_raw -> (occupancy_ns, core_type)
# aicore_tasks cols: [core_id, task_token_raw, reg_task_id, start_c, end_c, recv_c]
tok_dur = {}
for core_id, token, rtid, start_c, end_c, recv_c in swim["aicore_tasks"]:
    tok_dur[int(token)] = ((end_c - start_c + recv_c) * ns, core_types[core_id])

# deps task_id -> func_id (kernel_ids[0])
task_func = {}
for t in deps["tasks"]:
    tid = int(t["task_id"])
    kids = t.get("kernel_ids", [])
    task_func[tid] = kids[0] if kids else -1

# join
by_func = defaultdict(list)
func_ctype = {}
matched = 0
for tok, (dur, ct) in tok_dur.items():
    if tok in task_func:
        fid = task_func[tok]
        by_func[fid].append(dur)
        func_ctype[fid] = ct
        matched += 1
print(f"swimlane tasks={len(tok_dur)} deps tasks={len(task_func)} matched={matched}")

def stats(ds):
    ds = sorted(ds)
    return (f"n={len(ds)} mean={st.mean(ds):.1f} median={st.median(ds):.1f} "
            f"min={ds[0]:.1f} max={ds[-1]:.1f} CV={100*st.pstdev(ds)/st.mean(ds):.1f}%")

print("\n=== per func_id occupancy (ns) ===")
for fid in sorted(by_func):
    print(f"func_id={fid} ({func_ctype.get(fid,'?')}): {stats(by_func[fid])}")
