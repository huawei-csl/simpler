# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dataflow oracle: what a round would take if no manager stood between a
producer finishing and its consumer starting.

A list schedule of one round's task graph on the simulator's own core model -- 24
clusters of one cube and two vector cores, a mix block taking a whole cluster --
with the calibration's per-kernel compute, drawn as aSim draws it. A task is
released when its last producer finishes plus a hop cost, and its blocks take the
first free cores of their type. Tasks enter in submission order at RATE per us,
which is what the manager has to sustain to hand the fabric every task once.

The GroupQueue structure is modelled too: four controllers of six clusters
(cluster c belongs to controller c % 4, as in M2). A task runs on the controller
its last producer finished on unless SPREAD lets its blocks take any cluster; a
dependency costs H_IN inside a controller and H_X across one, and a task whose
blocks ran on more than one controller finishes in none, so every consumer pays
H_X for it.

  usage: asim_oracle.py <deps.json> <swimlane.json> <calib>
The swimlane names the tasks the simulator actually ran: dispatch predicates
retire the rest inline, and those are zero-length nodes here. Env: DRAW=0 uses
the plain means, RATE (default 6) the hand-over rate, SEEDS the draws averaged,
VARIANTS a JSON list of [name, whole_chip, h_in_us, h_x_us, spread] to run
instead of the default ladder.
"""

import heapq
import json
import os
import random
import sys
from collections import defaultdict

L = (1 << 32) - 1
deps = json.load(open(sys.argv[1]))
swim = json.load(open(sys.argv[2]))
calib, sigma = {}, {}
for line in open(sys.argv[3]):
    f = line.split()
    if f and f[0] != "LAT":
        calib[int(f[0])] = int(f[1]) / 1e3
        sigma[int(f[0])] = int(f[2]) / 1e3 if len(f) > 2 else 0.0
DRAW = os.environ.get("DRAW", "1") == "1"
RATE = float(os.environ.get("RATE", "6"))
SEEDS = int(os.environ.get("SEEDS", "3"))

ran = {int(r[1]) & L for r in swim["aicore_tasks"]}
info = {int(t["task_id"]) & L: (t["kernel_ids"], t.get("block_num", 1)) for t in deps["tasks"]}
preds, succs = defaultdict(set), defaultdict(set)
for e in deps["edges"]:
    p, s = int(e["pred"]) & L, int(e["succ"]) & L
    if p != s:
        preds[s].add(p)
        succs[p].add(s)
ids = sorted(set(info) | set(preds) | set(succs))
order = {i: n for n, i in enumerate(ids)}
N_CL, N_Q = 24, 4


def shape(k):
    if k[0] >= 0 and (k[1] >= 0 or k[2] >= 0):
        return "mix"
    if k[0] >= 0:
        return "aic"
    return "aiv"


class Cores:
    """Each core's free time: cube c, vector 2c and 2c+1 of cluster c."""

    def __init__(self, rng):
        self.aic = [0.0] * N_CL
        self.aiv = [0.0] * (2 * N_CL)
        self.rng = rng

    def dur(self, fid):
        m = calib.get(fid, 5.0)
        if not DRAW or sigma.get(fid, 0.0) == 0.0:
            return m
        return max(0.001, m + sigma[fid] * sum(self.rng.uniform(-1.0, 1.0) for _ in range(3)))

    def place(self, kernels, sh, release, clusters):
        """Start one block on the first free cores of its type; return (end, cluster)."""
        if sh == "aic":
            c = min(clusters, key=lambda x: self.aic[x])
            st = max(release, self.aic[c])
            self.aic[c] = st + self.dur(kernels[0])
            return self.aic[c], c
        if sh == "aiv":
            v = min((2 * c + j for c in clusters for j in (0, 1)), key=lambda x: self.aiv[x])
            st = max(release, self.aiv[v])
            self.aiv[v] = st + self.dur(kernels[1] if kernels[1] >= 0 else kernels[2])
            return self.aiv[v], v // 2
        c = min(clusters, key=lambda x: max(self.aic[x], self.aiv[2 * x], self.aiv[2 * x + 1]))
        st = max(release, self.aic[c], self.aiv[2 * c], self.aiv[2 * c + 1])
        end = st
        for slot, core in ((0, None), (1, 2 * c), (2, 2 * c + 1)):
            if kernels[slot] < 0:
                continue
            d = self.dur(kernels[slot])
            if core is None:
                self.aic[c] = st + d
            else:
                self.aiv[core] = st + d
            end = max(end, st + d)
        return end, c


def run_task(cores, task, release, q, chip, spread):
    """Run every block of one task; return (end, the controller it finished in or -1)."""
    kernels, nb = info[task]
    clusters = range(N_CL) if (chip or spread) else [c for c in range(N_CL) if c % N_Q == q]
    end, last_q, used = release, q, set()
    for _ in range(max(1, nb)):
        e, c = cores.place(kernels, shape(kernels), release, clusters)
        used.add(c % N_Q)
        if e >= end:
            end, last_q = e, c % N_Q
    # A task whose blocks ran on more than one controller finishes in no single
    # controller, so none can release its consumers locally.
    return end, (last_q if len(used) == 1 else -1)


def release_of(s, done, where, chip, h_in, h_x):
    """When consumer s may start, and the controller its last producer finished in."""
    last = max(preds[s], key=lambda p: done[p])
    qs = where[last]
    if s not in ran:
        return max(done[p] for p in preds[s]), qs
    return max(done[p] + (h_in if (chip or (where[p] == qs and qs >= 0)) else h_x) for p in preds[s]), qs


def run(chip, h_in, h_x, spread, seed):
    cores = Cores(random.Random(seed))
    left = {i: len(preds[i]) for i in ids}
    done, where = {}, {}

    def avail(i):
        return order[i] / RATE if RATE > 0 else 0.0

    ready = []
    for n, i in enumerate(i for i in ids if left[i] == 0):
        heapq.heappush(ready, (avail(i), i, n % N_Q))
    while ready:
        r, i, q = heapq.heappop(ready)
        if i in ran and i in info:
            done[i], where[i] = run_task(cores, i, r, q, chip, spread)
        else:
            done[i], where[i] = r, q
        for s in succs[i]:
            left[s] -= 1
            if left[s] == 0:
                rel, qs = release_of(s, done, where, chip, h_in, h_x)
                heapq.heappush(ready, (max(avail(s), rel), s, qs if qs >= 0 else order[s] % N_Q))
    return max(done.values())


VARIANTS = [
    ("today's hop on the whole chip (~ M0)", True, 3.7, 3.7, False),
    ("ideal fabric: whole chip, every hop 0.45 us", True, 0.45, 0.45, False),
    ("4 controllers, wide tasks confined, cross hops via manager", False, 0.45, 3.7, False),
    ("4 controllers, wide tasks confined, cross hops in hardware (change 1 only)", False, 0.45, 0.8, False),
    ("4 controllers, wide tasks span, cross hops via manager (change 2 only)", False, 0.45, 3.7, True),
    ("4 controllers, wide tasks span, cross hops in hardware (changes 1+2)", False, 0.45, 0.8, True),
]
if os.environ.get("VARIANTS"):
    VARIANTS = [tuple(v) for v in json.loads(os.environ["VARIANTS"])]
for name, chip, h_in, h_x, spread in VARIANTS:
    vals = [run(chip, h_in, h_x, spread, s) for s in range(SEEDS)]
    print(f"{sum(vals) / len(vals) / 1e3:7.3f} ms  {name}")
