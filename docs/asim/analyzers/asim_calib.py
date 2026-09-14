"""Per-task AICore compute durations by core_type from chip_swimlane_records.json.
Columns: [core_id, task_token_raw, reg_task_id, start_cycles, end_cycles, receive_to_start_cycles]."""
import json
import sys
import statistics as st
from collections import defaultdict

d = json.load(open(sys.argv[1]))
tasks = d["aicore_tasks"]
core_types = d["metadata"]["core_types"]
freq = d["metadata"]["clock_freq_hz"]
ns_per_cycle = 1e9 / freq
print(f"tasks={len(tasks)} clock={freq}Hz ns/cycle={ns_per_cycle:g}")

comp = defaultdict(list)   # end-start (pure compute)
occ = defaultdict(list)    # end-start + receive_to_start (core occupancy from receive)
r2s = defaultdict(list)    # receive_to_start
for core_id, token, rtid, start_c, end_c, recv_c in tasks:
    ct = core_types[core_id]
    comp[ct].append((end_c - start_c) * ns_per_cycle)
    occ[ct].append((end_c - start_c + recv_c) * ns_per_cycle)
    r2s[ct].append(recv_c * ns_per_cycle)

def stats(ds):
    ds = sorted(ds)
    return (f"n={len(ds)} mean={st.mean(ds):.1f} median={st.median(ds):.1f} "
            f"min={ds[0]:.1f} max={ds[-1]:.1f} stdev={st.pstdev(ds):.1f} "
            f"CV={100*st.pstdev(ds)/st.mean(ds):.1f}%")

for ct in sorted(comp):
    print(f"\ncore_type={ct}")
    print(f"  compute (end-start), ns:        {stats(comp[ct])}")
    print(f"  receive_to_start,   ns:         {stats(r2s[ct])}")
    print(f"  occupancy (end-receive), ns:    {stats(occ[ct])}")
