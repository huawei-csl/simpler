# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Silo oracle: one round's task graph on silos that forward completions to each other.

Every task runs in one silo -- four controllers of six clusters, cluster c in silo
c % 4. A supertask of at least SPLIT_MIN blocks is cut by block range into one piece
per silo; a narrower one runs whole in the silo of its latest producer that has one,
or the next silo in turn. A piece waits for every producer piece: one in its own
silo costs H_IN after it finishes, one in another H_N. EDGE=tile lets piece j of a
split consumer wait only on piece j of a split producer; EDGE=all (default) on all.

Each silo takes its pieces in task order through a window of W positions, one per
block: a piece enters once its blocks fit, it is at the head of its silo's order
(ORDER=fifo) or the oldest whose producers have entered (ORDER=ready), and every
producer piece has entered -- a consumer can name only a producer some silo has fed.
Positions free RETIRE_US after a piece finishes -- the manager learns of it and
retires it -- and a piece entering is legible to its controller ENTER_US after the
manager decides. C bounds the producer pieces a piece can name:
past it, the manager resolves the rest and the piece cannot start before MGR_US after
the (n-C)-th of them finished. A task its dispatch predicate retires is a zero-length
node; PRED adds a hop through it, for a predicate the manager evaluates.

  usage: asim_silo_oracle.py <deps.json> <swimlane.json> <calib>
Env: W (default 32), ORDER, H_IN (0.45), H_N (0.8), PRED (0), SPLIT_MIN (4), EDGE,
C (unset = any), RETIRE_US (0), ENTER_US (0), RATE (6 tasks/us), SEEDS (3).
"""

import heapq
import itertools
import json
import os
import random
import sys
from collections import defaultdict

L = (1 << 32) - 1
N_CL, N_Q = 24, 4
MGR_US = 3.7
ENV = os.environ
W = float(ENV.get("W", "32"))
ORDER = ENV.get("ORDER", "fifo")
H_IN = float(ENV.get("H_IN", "0.45"))
H_N = float(ENV.get("H_N", "0.8"))
PRED = float(ENV.get("PRED", "0"))
SPLIT_MIN = float(ENV.get("SPLIT_MIN", "4"))
EDGE = ENV.get("EDGE", "all")
C = int(ENV["C"]) if ENV.get("C") else None
RATE = float(ENV.get("RATE", "6"))
RETIRE_US = float(ENV.get("RETIRE_US", "0"))
ENTER_US = float(ENV.get("ENTER_US", "0"))
SEEDS = int(ENV.get("SEEDS", "3"))

deps = json.load(open(sys.argv[1]))
swim = json.load(open(sys.argv[2]))
calib, sigma = {}, {}
for line in open(sys.argv[3]):
    f = line.split()
    if f and f[0] != "LAT":
        calib[int(f[0])] = int(f[1]) / 1e3
        sigma[int(f[0])] = int(f[2]) / 1e3 if len(f) > 2 else 0.0

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
real = {i for i in ids if i in ran and i in info}
nb = {t: max(1, info[t][1]) for t in real}
split = {t: nb[t] >= SPLIT_MIN for t in real}


def piece_blocks(t, j):
    return nb[t] if j < 0 else nb[t] // N_Q + (1 if j < nb[t] % N_Q else 0)


def pieces(t):
    if split[t]:
        return [("p", t, j) for j in range(N_Q) if piece_blocks(t, j) > 0]
    return [("p", t, -1)]


silo, turn = {}, 0
for t in ids:
    if t in real and not split[t]:
        stamped = [p for p in preds[t] if p in silo]
        if stamped:
            silo[t] = silo[max(stamped, key=lambda q: order[q])]
        else:
            silo[t], turn = turn % N_Q, turn + 1


def gof(n):
    return n[2] if n[2] >= 0 else silo[n[1]]


def blocks(n):
    return piece_blocks(n[1], n[2])


nodes = [n for t in ids for n in (pieces(t) if t in real else [("r", t, -1)])]
real_nodes = [n for n in nodes if n[0] == "p"]
npreds = defaultdict(list)
for s in ids:
    for p in preds[s]:
        src = pieces(p) if p in real else [("r", p, -1)]
        for n in pieces(s) if s in real else [("r", s, -1)]:
            tile = n[0] == "p" and p in real and EDGE == "tile" and split[p] and split[s] and ("p", p, n[2]) in src
            npreds[n].extend([("p", p, n[2])] if tile else src)
nsuccs = defaultdict(list)
for n, ps in npreds.items():
    for d in ps:
        nsuccs[d].append(n)
named = {}
for n in nodes:
    named[n] = set().union(*[named[d] if d[0] == "r" else {d} for d in npreds[n]]) if npreds[n] else set()
named_by = defaultdict(list)
for n in real_nodes:
    for d in named[n]:
        named_by[d].append(n)


def shape(k):
    if k[0] >= 0 and (k[1] >= 0 or k[2] >= 0):
        return "mix"
    return "aic" if k[0] >= 0 else "aiv"


class Cores:
    """Each core's free time: cube c, vector 2c and 2c+1 of cluster c."""

    def __init__(self, rng):
        self.aic = [0.0] * N_CL
        self.aiv = [0.0] * (2 * N_CL)
        self.rng = rng

    def dur(self, fid):
        m = calib.get(fid, 5.0)
        if sigma.get(fid, 0.0) == 0.0:
            return m
        return max(0.001, m + sigma[fid] * sum(self.rng.uniform(-1.0, 1.0) for _ in range(3)))

    def place(self, k, rel, cl):
        sh = shape(k)
        if sh == "aic":
            c = min(cl, key=lambda x: self.aic[x])
            st = max(rel, self.aic[c])
            self.aic[c] = st + self.dur(k[0])
            return self.aic[c]
        if sh == "aiv":
            v = min((2 * c + j for c in cl for j in (0, 1)), key=lambda x: self.aiv[x])
            st = max(rel, self.aiv[v])
            self.aiv[v] = st + self.dur(k[1] if k[1] >= 0 else k[2])
            return self.aiv[v]
        c = min(cl, key=lambda x: max(self.aic[x], self.aiv[2 * x], self.aiv[2 * x + 1]))
        st = max(rel, self.aic[c], self.aiv[2 * c], self.aiv[2 * c + 1])
        end = st
        for slot, core in ((0, None), (1, 2 * c), (2, 2 * c + 1)):
            if k[slot] >= 0:
                d = self.dur(k[slot])
                if core is None:
                    self.aic[c] = st + d
                else:
                    self.aiv[core] = st + d
                end = max(end, st + d)
        return end


class Run:
    """One draw: an event schedule of finishes, entries and releases."""

    FIN, FREE, ENT, REL = 0, 1, 2, 3

    def __init__(self, seed):
        self.cores = Cores(random.Random(seed))
        self.left = {n: len(npreds[n]) for n in nodes}
        self.unentered = {n: len(named[n]) for n in real_nodes}
        self.R, self.done, self.entered, self.relmet = {}, {}, {}, {}
        self.pos = [0] * N_Q
        self.fifo = [[n for n in real_nodes if gof(n) == g] for g in range(N_Q)]
        self.ptr = [0] * N_Q
        self.ready = [[] for _ in range(N_Q)]
        for n in real_nodes:
            if self.unentered[n] == 0:
                heapq.heappush(self.ready[gof(n)], (order[n[1]], n[2], n))
        self.ev, self.seq = [], itertools.count()

    def push(self, t, kind, x):
        heapq.heappush(self.ev, (t, kind, next(self.seq), x))

    def release_vec(self, n):
        v = [order[n[1]] / RATE] * N_Q
        for d in npreds[n]:
            if d[0] == "r":
                v = [max(a, b) for a, b in zip(v, self.R[d])]
            else:
                g = gof(d)
                v = [max(x, self.done[d] + (H_IN if q == g else H_N)) for q, x in enumerate(v)]
        if C is not None and n[0] == "p" and len(named[n]) > C:
            times = sorted(self.done[d] for d in named[n])
            v = [max(x, times[len(times) - C - 1] + MGR_US) for x in v]
        return v

    def deps_done(self, n0):
        stack = [n0]
        while stack:
            n = stack.pop()
            if n[0] == "r":
                self.R[n] = [x + (PRED if npreds[n] else 0.0) for x in self.release_vec(n)]
                for s in nsuccs[n]:
                    self.left[s] -= 1
                    if self.left[s] == 0:
                        stack.append(s)
            else:
                self.relmet[n] = self.release_vec(n)[gof(n)]
                if n in self.entered:
                    self.push(max(self.relmet[n], self.entered[n]), self.REL, n)

    def head(self, g):
        if ORDER == "fifo":
            if self.ptr[g] >= len(self.fifo[g]):
                return None
            n = self.fifo[g][self.ptr[g]]
            return n if self.unentered[n] == 0 else None
        return self.ready[g][0][2] if self.ready[g] else None

    def try_enter(self, g, now):
        while (n := self.head(g)) is not None:
            if self.pos[g] > 0 and self.pos[g] + blocks(n) > W:
                return
            if order[n[1]] / RATE > now:
                self.push(order[n[1]] / RATE, self.ENT, g)
                return
            if ORDER == "fifo":
                self.ptr[g] += 1
            else:
                heapq.heappop(self.ready[g])
            self.entered[n] = now + ENTER_US
            self.pos[g] += blocks(n)
            if n in self.relmet:
                self.push(max(self.relmet[n], self.entered[n]), self.REL, n)
            for c in named_by[n]:
                self.unentered[c] -= 1
                if self.unentered[c] == 0:
                    if ORDER == "ready":
                        heapq.heappush(self.ready[gof(c)], (order[c[1]], c[2], c))
                    if gof(c) != g:
                        self.push(now, self.ENT, gof(c))

    def run(self):
        for n in [n for n in nodes if self.left[n] == 0]:
            self.deps_done(n)
        for g in range(N_Q):
            self.push(0.0, self.ENT, g)
        while self.ev:
            t, kind, _, x = heapq.heappop(self.ev)
            if kind == self.ENT:
                self.try_enter(x, t)
            elif kind == self.REL:
                g = gof(x)
                cl = [c for c in range(N_CL) if c % N_Q == g]
                self.done[x] = max(self.cores.place(info[x[1]][0], t, cl) for _ in range(blocks(x)))
                self.push(self.done[x], self.FIN, x)
                self.push(self.done[x] + RETIRE_US, self.FREE, x)
            elif kind == self.FIN:
                for s in nsuccs[x]:
                    self.left[s] -= 1
                    if self.left[s] == 0:
                        self.deps_done(s)
            else:
                self.pos[gof(x)] -= blocks(x)
                self.try_enter(gof(x), t)
        return max(self.done.values())


spans = [Run(seed).run() for seed in range(SEEDS)]
cross = sum(1 for n in real_nodes for d in named[n] if gof(d) != gof(n))
print(
    f"{sum(spans) / len(spans) / 1e3:.3f} ms  W={W:g} ORDER={ORDER} H_IN={H_IN} H_N={H_N} PRED={PRED} "
    f"SPLIT_MIN={SPLIT_MIN:g} EDGE={EDGE} C={C} RETIRE_US={RETIRE_US} ENTER_US={ENTER_US}  "
    f"({cross} of {sum(len(named[n]) for n in real_nodes)} named producer pieces cross silos)"
)
