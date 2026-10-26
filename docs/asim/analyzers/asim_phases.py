# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-round device-phase attribution for a host-orchestrated aSim A/B.

Reads the `[STRACE] ... name=chip.run.runner_run.device_wall[.<phase>]` markers
each run writes, groups them by invocation, drops the first two rounds and reports
the median duration of each phase an arm spends per round.

Only durations are read. A span's `ts` is a raw sys-counter offset while its `dur`
has the simulated device's own overrun removed, so on an arm whose model overruns
(every group_queue arm) the two are on different clocks and any ts-derived
interval is wrong by the overrun accrued before it. `other` is therefore taken as
the residual of `graph_build` after its children — the per-thread device_log
writes between the phases and the last thread's finalize, less whatever the
children's cross-thread union spans double-count where they overlap, so it is a
remainder and not comparable across arms — and `total` is the sum of the three
top-level phases rather than the uncorrected `device_wall`.
"""

import glob
import os
import re
import statistics as st

SPAN = re.compile(r"inv=(\d+) [^\n]*?name=chip\.run\.runner_run\.device_wall(\.[a-z_0-9]+)? ts=(\d+) dur=(\d+)")

OUT = os.environ.get("OUT_ROOT", "tmp_sac_out")
PFX = os.environ.get("PFX", "q8p")
ARMS = os.environ.get("ARMS_ORDER", "m0 m2 part").split()
DROP = int(os.environ.get("DROP", "2"))
PHASES = ("preamble", "graph_build", "arena_wire", "sm_reset", "group_seed", "classify", "sched", "retire", "post_orch")
COLS = (
    "total",
    "preamble",
    "arena_wire",
    "sm_reset",
    "group_seed",
    "classify",
    "prologue",
    "sched",
    "retire",
    "post_orch",
    "other",
    "device_wall",
)


def rounds(path):
    """Yield one {phase: (ts, dur)} dict a round, in invocation order."""
    by_inv = {}
    for inv, phase, ts, dur in SPAN.findall(open(path, errors="ignore").read()):
        by_inv.setdefault(int(inv), {})[(phase or ".device_wall")[1:]] = (int(ts), int(dur))
    return [by_inv[i] for i in sorted(by_inv)][DROP:]


def metrics(r):
    m = {p: r.get(p, (0, 0))[1] / 1e3 for p in PHASES}
    m["device_wall"] = r.get("device_wall", (0, 0))[1] / 1e3
    m["prologue"] = m["arena_wire"] + m["sm_reset"] + m["group_seed"] + m["classify"]
    m["other"] = m["graph_build"] - m["prologue"] - m["sched"] - m["retire"]
    m["total"] = m["preamble"] + m["graph_build"] + m["post_orch"]
    return m


res = {}
for arm in ARMS:
    acc = {c: [] for c in COLS}
    n = 0
    for f in sorted(glob.glob(f"{OUT}/{PFX}_{arm}_*/pytest_full.log")):
        for r in rounds(f):
            if "device_wall" not in r:
                continue
            m = metrics(r)
            for c in COLS:
                acc[c].append(m[c])
            n += 1
    res[arm] = (acc, n)

hdr = f"{'arm':5}" + "".join(f"{c:>12}" for c in COLS) + "   n"
print(hdr)
print("-" * len(hdr))
for arm in ARMS:
    acc, n = res[arm]
    if not n:
        print(f"{arm:5} (no data)")
        continue
    print(f"{arm:5}" + "".join(f"{st.median(acc[c]):12.1f}" for c in COLS) + f" {n:>4}")
print(f"\n(us a round, median over rounds; first {DROP} rounds dropped)")

base = ARMS[0]
if res[base][1]:
    print(f"\ndelta vs {base} (us a round):")
    print(f"{'arm':5}" + "".join(f"{c:>12}" for c in COLS))
    for arm in ARMS[1:]:
        if not res[arm][1]:
            continue
        print(f"{arm:5}" + "".join(f"{st.median(res[arm][0][c]) - st.median(res[base][0][c]):+12.1f}" for c in COLS))
