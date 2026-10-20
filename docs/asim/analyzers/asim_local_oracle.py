# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Local-id oracle: one round's task graph on partitions whose groups number their share locally.

Every group numbers its share of each partition -- its subpartition -- in one local id
stream, in dependency-level order, and keeps its own watermark over it. A producer run
in another group stands as one local id, its proxy, just below its first dependent in
the group (PROXY_AT=producer puts the proxy for a later partition's dependent at the
end of the producer's own subpartition instead); the proxy lands RING us after the
last piece of the producer finishes. A record names NAMED producers, the latest-ending,
and its threshold -- where the next one ends -- must be reached by the group's
watermark; an entry starts only once the watermark is within LOOK ids of it.
BARRIER=local holds an entry until its group's earlier subpartitions have finished.
Subpartition p of every group is resident once every group has finished p - GATE
(GATE=0: all at once), ENTER_US later. A task its dispatch predicate retires takes one
id, SKIP_IDX, in its home group; EXPERTS=static or ring makes half of them run, at home
or dealt to the groups in turn, into a region of DYN ids per subpartition (DYN_AT=level:
at the level conditional work starts; end: after the rest), naming its producers
(DYN_MGR=0) or posted MGR_US after they finish (DYN_MGR=1).

SHARED=1 models the shared-watermark design in the same machinery: one partition-wide
numbering, every finish known in the other groups RING later, and each group entering
a partition once the whole partition before has finished (with BARRIER=local).

  usage: asim_local_oracle.py <deps.json> <swimlane.json> <calib>
Env: H_IN (1.02) RING (0.015) SPLIT_MIN (8) PART (372) NAMED (4) LOOK (inf) BARRIER
(none|local) GATE (2) ENTER_US (1.32) RATE (6) SEEDS (3) PRED (0) EXPERTS (0) DYN (0)
DYN_AT (level) DYN_MGR (0) MGR_US (3.7) SKIP_IDX (1) PROXY_AT (consumer) STATS (0)
SHARED (0). The line reports the mean makespan, each seed's, the look-ahead each group
used, and the waits on the watermark: for producers past NAMED, for the look-ahead,
and for the barrier.
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
ENV = os.environ
H_IN = float(ENV.get("H_IN", "1.02"))
RING = float(ENV.get("RING", "0.015"))
SPLIT_MIN = int(ENV.get("SPLIT_MIN", "8"))
PART = int(ENV.get("PART", "372"))
NAMED = int(ENV.get("NAMED", "4"))
LOOK = float(ENV.get("LOOK", "inf"))
BARRIER = ENV.get("BARRIER", "none")
GATE = int(ENV.get("GATE", "2"))
ENTER_US = float(ENV.get("ENTER_US", "1.32"))
RATE = float(ENV.get("RATE", "6"))
SEEDS = int(ENV.get("SEEDS", "3"))
PRED = float(ENV.get("PRED", "0"))
EXPERTS = ENV.get("EXPERTS", "0")
DYN = int(ENV.get("DYN", "0"))
SKIP_IDX = int(ENV.get("SKIP_IDX", "1"))
STATS = int(ENV.get("STATS", "0"))
SHARED = int(ENV.get("SHARED", "0"))
DYN_AT = ENV.get(
    "DYN_AT", "level"
)  # level: the dynamic region sits at the conditional tasks' level; end: after the subpartition
DYN_MGR = int(ENV.get("DYN_MGR", "0"))  # routed work posted once its producers are complete as the managers see them
MGR_US = float(ENV.get("MGR_US", "3.7"))
# producer: a proxy a later partition's dependent needs goes at the end of its producer's own subpartition
PROXY_AT = ENV.get("PROXY_AT", "consumer")
INF = float("inf")

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
kern = {t for t in ids if t in info and any(k >= 0 for k in info[t][0]) and info[t][1] > 0}
subs = {t: sum(1 for k in info[t][0] if k >= 0) for t in kern}
nb = {t: max(1, info[t][1]) for t in kern}
real = {t for t in kern if t in ran}
skipped = kern - real
split = {t: nb[t] >= SPLIT_MIN for t in kern}

# --- partitions, as the host cuts them
total = len(ids)
crossing = [0] * (total + 2)
for c in ids:
    for p in preds[c]:
        if order[p] < order[c]:
            crossing[order[p] + 1] += 1
            crossing[order[c] + 1] -= 1
for i in range(1, len(crossing)):
    crossing[i] += crossing[i - 1]
starts = [0]
for k in range(PART, total, PART):
    lo, hi = max(starts[-1] + 1, k - PART // 4), min(total - 1, k + PART // 4)
    if lo > hi:
        continue
    best = lo
    for i in range(lo, hi + 1):
        if crossing[i] < crossing[best] or (crossing[i] == crossing[best] and abs(i - k) < abs(best - k)):
            best = i
    starts.append(best)
starts.append(total)
part_of = {}
for p in range(len(starts) - 1):
    for n in range(starts[p], starts[p + 1]):
        part_of[ids[n]] = p
NP = len(starts) - 1

# --- experts: half the skipped tasks run (even expert, six supertasks each)
cond_runs = set()
if EXPERTS != "0":
    by_part = defaultdict(list)
    for t in sorted(skipped, key=lambda t: order[t]):
        by_part[part_of[t]].append(t)
    for p, ts in by_part.items():
        for r, t in enumerate(ts):
            if (r // 6) % 2 == 0:
                cond_runs.add(t)
cond = skipped  # every conditional task; the ones in cond_runs run

# --- placement, as the host stamps it
silo, turn = {}, 0
for t in ids:
    if t not in kern:
        continue
    if split[t]:
        silo[t] = "spread"
        continue
    latest, inh = -1, None
    for p in preds[t]:
        if p in silo and silo[p] != "spread" and order[p] > latest:
            latest, inh = order[p], silo[p]
    silo[t] = inh if inh is not None else turn % N_Q
    if inh is None:
        turn += 1
ticket_group = {}
if EXPERTS == "ring":
    for k, t in enumerate(sorted(cond_runs, key=lambda t: order[t])):
        ticket_group[t] = k % N_Q


def piece_blocks(t, g):
    if split[t]:
        return nb[t] * (g + 1) // N_Q - nb[t] * g // N_Q
    return nb[t]


def home(t):
    return silo[t] if silo[t] != "spread" else None


def run_groups(t):
    """Groups holding a piece of t that runs on cores."""
    if t in cond and t not in cond_runs:
        return []
    if split[t]:
        return [g for g in range(N_Q) if piece_blocks(t, g) > 0]
    if t in cond_runs and EXPERTS == "ring":
        return [ticket_group[t]]
    return [silo[t]]


# --- levels, as the host numbers them
level = {}
for t in ids:
    first = starts[part_of[t]]
    lv = 0
    for p in preds[t]:
        if order[p] < first or order[p] >= order[t]:
            continue
        lv = max(lv, level[p] + (1 if p in kern else 0))
    level[t] = lv


_succs_k = {}


def succs_k(t):
    """Kernel consumers of t, seen through dummies."""
    if t in _succs_k:
        return _succs_k[t]
    out, stack, seen = [], list(succs[t]), set()
    while stack:
        s_ = stack.pop()
        if s_ in seen:
            continue
        seen.add(s_)
        if s_ in kern:
            out.append(s_)
        else:
            stack.extend(succs[s_])
    _succs_k[t] = out
    return out


def producers(t):
    out, stack, seen = [], list(preds[t]), set()
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        if p in kern:
            out.append(p)
        else:
            stack.extend(preds[p])
    return out


class Piece:
    __slots__ = (
        "task",
        "g",
        "p",
        "lo",
        "hi",
        "blocks",
        "kind",
        "named",
        "thr",
        "deps",
        "unmet",
        "known",
        "rel",
        "done",
        "cond_at",
        "resident",
        "announce",
        "conds_at",
        "expect",
    )

    def __init__(self, task, g, p, kind):
        self.task, self.g, self.p, self.kind = task, g, p, kind  # run | slot | proxy
        self.named, self.thr, self.deps, self.announce = [], 0, [], []
        self.lo = self.hi = -1
        self.blocks = self.expect = 0


members = defaultdict(list)  # (p, g) -> pieces with ids, in id order
piece_of = {}  # (task, g) -> run piece
slot_of = {}  # task -> its completion slot, in its home group (conditional tasks)
proxy_of = {}  # (task, g) -> proxy in g (new design) or mirror (SHARED)
base = {}
stream_len = [0] * N_Q
dyn_ids = {}
stats = defaultdict(int)


def new_proxy(P, g, p, expect, cur):
    pr = Piece(P, g, p, "proxy")
    pr.lo, pr.hi = cur, cur + 1
    pr.expect = expect
    proxy_of[(P, g)] = pr
    members[(p, g)].append(pr)
    stats["proxies"] += 1
    return pr


def remote_count(P, g):
    """Finishes group g waits for from elsewhere before producer P is done for it."""
    if P in cond and P not in cond_runs:
        return 0 if home(P) == g else 1
    if P in cond_runs and EXPERTS == "ring":
        return 1
    remote = sum(1 for h in run_groups(P) if h != g)
    return remote + (1 if remote and g in run_groups(P) else 0)


def place_dyn(cur, p, g, ordered):
    """Place group g's dynamic region of partition p at local id `cur`: with DYN_MGR=0,
    proxies for every conditional task's producers first. Returns the next free id."""
    if not DYN_MGR:
        for t in ordered:
            if t not in cond:
                continue
            for P in producers(t):
                if (P, g) in proxy_of:
                    continue
                n_remote = remote_count(P, g)
                if n_remote > 0:
                    new_proxy(P, g, p, n_remote, cur)
                    cur += 1
    dyn_ids[(p, g)] = [cur, cur + DYN]
    return cur + DYN


for p in range(NP):
    tasks_p = [t for t in ids[starts[p] : starts[p + 1]] if t in kern]
    ordered = sorted(tasks_p, key=lambda t: (level[t], order[t]))
    if SHARED:
        # one numbering for the partition; every group's stream carries it
        cur = stream_len[0]
        for g in range(N_Q):
            base[(p, g)] = cur
        for t in ordered:
            n = nb[t] * subs[t]
            if t in cond and t not in cond_runs:
                pc = Piece(t, home(t), p, "slot")
                pc.lo, pc.hi = cur, cur + n
                slot_of[t] = pc
                members[(p, home(t))].append(pc)
                stats["slots"] += n
            else:
                lo = cur
                for g in run_groups(t):
                    pc = Piece(t, g, p, "run")
                    pc.blocks = piece_blocks(t, g)
                    k = pc.blocks * subs[t]
                    pc.lo, pc.hi = lo, lo + k
                    lo += k
                    piece_of[(t, g)] = pc
                    members[(p, g)].append(pc)
                    stats["entries"] += k
            cur += n
        for g in range(N_Q):
            stream_len[g] = cur
        continue
    for g in range(N_Q):
        base[(p, g)] = stream_len[g]
        mem = []
        for t in ordered:
            if t in cond:
                if home(t) == g:
                    mem.append((t, "slot"))
                if t in cond_runs and EXPERTS != "ring" and home(t) == g:
                    mem.append((t, "run"))
                continue
            if g in run_groups(t):
                mem.append((t, "run"))
        cur = stream_len[g]
        dyn_placed = False

        for t, kind in mem:
            if EXPERTS == "ring" and DYN_AT == "level" and not dyn_placed and t in cond:
                cur = place_dyn(cur, p, g, ordered)
                dyn_placed = True
            if kind == "run" or t not in cond_runs:
                prods = producers(t)
                if BARRIER == "global":
                    prods = [P for P in prods if part_of[P] == p]
                if PROXY_AT == "producer":
                    prods = [P for P in prods if part_of[P] == p]
                for P in prods:
                    if (P, g) in proxy_of:
                        continue
                    n_remote = remote_count(P, g)
                    if n_remote > 0:
                        new_proxy(P, g, p, n_remote, cur)
                        cur += 1
            pc = Piece(t, g, p, kind)
            if kind == "run":
                pc.blocks = piece_blocks(t, g)
                n = pc.blocks * subs[t]
                piece_of[(t, g)] = pc
                stats["entries"] += n
            else:
                n = SKIP_IDX
                slot_of[t] = pc
                stats["slots"] += n
            pc.lo, pc.hi = cur, cur + n
            cur += n
            members[(p, g)].append(pc)
        if EXPERTS == "ring" and not dyn_placed:
            cur = place_dyn(cur, p, g, ordered)
        if PROXY_AT == "producer":
            # remote producers of this partition that later partitions depend on
            later = set()
            for t in tasks_p:
                for S in succs_k(t):
                    if part_of[S] > p and (S in cond or g in run_groups(S) or (S in cond and home(S) == g)):
                        later.add(t)
                        break
            for P in sorted(later, key=lambda t: (level[t], order[t])):
                if (P, g) in proxy_of:
                    continue
                n_remote = remote_count(P, g)
                if n_remote > 0:
                    new_proxy(P, g, p, n_remote, cur)
                    cur += 1
        stream_len[g] = cur

dyn_next = {k: v[0] for k, v in dyn_ids.items()}
if EXPERTS == "ring" and not SHARED:
    for t in sorted(cond_runs, key=lambda t: order[t]):
        g, p = ticket_group[t], part_of[t]
        n = nb[t] * subs[t]
        if dyn_next[(p, g)] + n > dyn_ids[(p, g)][1]:
            stats["dyn_overflow"] += 1
            continue
        pc = Piece(t, g, p, "run")
        pc.blocks = nb[t]
        pc.lo, pc.hi = dyn_next[(p, g)], dyn_next[(p, g)] + n
        dyn_next[(p, g)] += n
        piece_of[(t, g)] = pc
        members[(p, g)].append(pc)
        stats["dyn_entries"] += n

# --- records
if SHARED:
    # a producer is done for group g once every piece of it is known there: a mirror
    # per (producer, group), with no id of its own
    for (p, g), pcs in list(members.items()):
        for pc in pcs:
            if pc.kind == "proxy":
                continue
            cands = []
            for P in producers(pc.task):
                if part_of[P] != p:
                    continue  # the shared switch finished it
                if (P, g) not in proxy_of:
                    m = Piece(P, g, p, "proxy")
                    srcs = (
                        [slot_of[P]] if P in cond and P not in cond_runs else [piece_of[(P, h)] for h in run_groups(P)]
                    )
                    m.hi = max(s.hi for s in srcs)
                    m.expect = len(srcs)
                    for s in srcs:
                        s.announce.append((m, 0.0 if s.g == g else RING))
                    proxy_of[(P, g)] = m
                cands.append(proxy_of[(P, g)])
            cands = sorted({id(c): c for c in cands}.values(), key=lambda c: -c.hi)
            pc.named = cands[:NAMED]
            pc.thr = cands[NAMED].hi if len(cands) > NAMED else 0
            if len(cands) > NAMED:
                stats["over_named"] += 1
            for c in pc.named:
                c.deps.append(pc)
else:
    mirrors = {}
    for (p, g), pcs in list(members.items()):
        for pc in pcs:
            if pc.kind == "proxy" or (pc.kind == "slot" and pc.task in cond_runs):
                continue
            cands = []
            dynamic = (p, g) in dyn_ids and pc.lo >= dyn_ids[(p, g)][0] and pc.hi <= dyn_ids[(p, g)][1]
            if dynamic and DYN_MGR:
                for P in producers(pc.task):
                    if (P, g) not in mirrors:
                        m = Piece(P, g, p, "proxy")
                        srcs = (
                            [slot_of[P]]
                            if P in cond and P not in cond_runs
                            else [piece_of[(P, h)] for h in run_groups(P)]
                        )
                        m.hi = -1
                        m.expect = len(srcs)
                        for sr in srcs:
                            sr.announce.append((m, MGR_US))
                        mirrors[(P, g)] = m
                    cands.append(mirrors[(P, g)])
                pc.named = cands
                for c in pc.named:
                    c.deps.append(pc)
                continue
            for P in producers(pc.task):
                if BARRIER == "global" and part_of[P] != p:
                    continue
                if PROXY_AT == "producer" and part_of[P] < p:
                    continue
                if (P, g) in proxy_of:
                    cands.append(proxy_of[(P, g)])
                elif P in cond and P not in cond_runs:
                    cands.append(slot_of[P])
                elif (P, g) in piece_of:
                    if BARRIER == "local" and part_of[P] < p:
                        continue
                    cands.append(piece_of[(P, g)])
            cands = sorted({id(c): c for c in cands}.values(), key=lambda c: -c.hi)
            pc.named = cands[:NAMED]
            pc.thr = cands[NAMED].hi if len(cands) > NAMED else 0
            if len(cands) > NAMED:
                stats["over_named"] += 1
            for c in pc.named:
                c.deps.append(pc)
    for (P, g), pr in proxy_of.items():
        if P in cond and P not in cond_runs:
            slot_of[P].announce.append((pr, RING))
        else:
            for h in run_groups(P):
                if h != g or pr.expect > 1 or (P in cond_runs and EXPERTS == "ring"):
                    piece_of[(P, h)].announce.append((pr, RING if h != g else 0.0))
    for t in cond_runs:
        sl = slot_of[t]
        sl.kind = "proxy"
        sl.expect = 1
        piece_of[(t, run_groups(t)[0])].announce.append((sl, RING if EXPERTS == "ring" else 0.0))

MIRRORS = mirrors if not SHARED else {}
if STATS and not SHARED:
    per = defaultdict(lambda: defaultdict(int))
    for (p, g), pcs in members.items():
        for pc in pcs:
            per[(p, g)][pc.kind] += pc.hi - pc.lo
            per[(p, g)]["n_" + pc.kind] += 1
    sizes = [v["run"] + v["slot"] + v["proxy"] for v in per.values()]
    prox = [v["n_proxy"] for v in per.values()]
    runs_ = [v["run"] for v in per.values()]
    two = [per[(p, g)]["n_proxy"] + per[(p + 1, g)]["n_proxy"] for p in range(NP - 1) for g in range(N_Q)]
    glob = {P for (P, g) in proxy_of}
    ann = [0] * N_Q
    for P in glob:
        for h in run_groups(P):
            ann[h] += 1
        hp = home(P)
        if P in cond and P not in cond_runs and hp is not None:
            ann[hp] += 1
    all_named = sum(len(pc.named) for pcs in members.values() for pc in pcs)
    cross_named = sum(1 for pcs in members.values() for pc in pcs for c in pc.named if c.kind == "proxy")
    cur_sizes = [sum(nb[t] * subs[t] for t in ids[starts[p] : starts[p + 1]] if t in kern) for p in range(NP)]
    print(
        f"partitions={NP} subpartition ids: max {max(sizes)} mean {sum(sizes) / len(sizes):.0f} "
        f"(entries max {max(runs_)} mean {sum(runs_) / len(runs_):.0f}; proxies max {max(prox)} "
        f"mean {sum(prox) / len(prox):.0f}; two resident max {max(two)}) | current partition-wide ids: "
        f"max {max(cur_sizes)} mean {sum(cur_sizes) / len(cur_sizes):.0f}"
    )
    print(
        f"global supertasks={len(glob)} of {len(kern)} ({sum(1 for P in glob if split[P])} spread, "
        f"{sum(1 for P in glob if P in cond)} conditional); announcements per group per round={ann}; "
        f"named producers={all_named}, proxies among them={cross_named}; records over {NAMED} producers="
        f"{stats['over_named']}; dyn_overflow={stats['dyn_overflow']} dyn_entries={stats['dyn_entries']}"
    )


def shape(k):
    if k[0] >= 0 and (k[1] >= 0 or k[2] >= 0):
        return "mix"
    return "aic" if k[0] >= 0 else "aiv"


class Cores:
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
    FIN, KNOWN, RES, DEC, MARK = 0, 1, 2, 3, 4

    def __init__(self, seed):
        self.cores = Cores(random.Random(seed))
        self.ev, self.seq = [], itertools.count()
        self.known = [[INF] * stream_len[g] for g in range(N_Q)]
        self.passed = [[INF] * stream_len[g] for g in range(N_Q)]
        self.wm = [0] * N_Q
        self.wm_at = [0.0] * N_Q
        self.wait = [[] for _ in range(N_Q)]
        self.resident_at = {}
        self.part_left = defaultdict(int)
        self.part_done = {}
        self.makespan = 0.0
        self.la_need = [0] * N_Q
        self.waits, self.wait_us = [0, 0, 0], [0.0, 0.0, 0.0]
        self.proxy_hold_us = [0.0] * N_Q
        self.cond_left = defaultdict(int)
        self.all = (
            [pc for pcs in members.values() for pc in pcs]
            + [m for m in proxy_of.values() if m.lo < 0]
            + list(MIRRORS.values())
        )
        for pc in self.all:
            pc.known = pc.done = INF
            pc.rel = None
            pc.resident = False
            pc.conds_at = pc.cond_at = 0.0
            if pc.kind == "proxy":
                pc.unmet = pc.expect
            else:
                pc.unmet = len(pc.named)
                self.part_left[(pc.p, pc.g)] += 1
                if pc.kind == "slot":
                    self.cond_left[pc.p] += 1

    def push(self, t, kind, x):
        heapq.heappush(self.ev, (t, kind, next(self.seq), x))

    def wm_cond(self, pc):
        """The id the group's watermark must reach for pc, which condition sets it
        (0 threshold, 1 look-ahead, 2 barrier), and when it did (or None)."""
        need, why = pc.thr, 0
        if LOOK != INF and int(pc.hi - LOOK) > need:
            need, why = int(pc.hi - LOOK), 1
        if BARRIER == "local" and base[(pc.p, pc.g)] > need:
            need, why = base[(pc.p, pc.g)], 2
        if need <= 0:
            return 0, why, 0.0
        if self.wm[pc.g] >= need:
            return need, why, self.passed[pc.g][need - 1]
        return need, why, None

    def try_release(self, pc, now):
        if pc.rel is not None or not pc.resident or pc.unmet > 0:
            return
        need, why, at = self.wm_cond(pc)
        if at is None:
            heapq.heappush(self.wait[pc.g], (need, next(self.seq), pc))
            return
        if at > pc.conds_at and at > self.resident_at[(pc.p, pc.g)]:
            self.waits[why] += 1
            self.wait_us[why] += at - max(pc.conds_at, self.resident_at[(pc.p, pc.g)])
        t = max(at, pc.conds_at)
        pc.rel = max(t + (H_IN if pc.kind == "run" else PRED), self.resident_at[(pc.p, pc.g)])
        self.la_need[pc.g] = max(self.la_need[pc.g], pc.hi - self.wm[pc.g])
        if pc.kind == "run":
            cl = [c for c in range(N_CL) if c % N_Q == pc.g]
            pc.done = max(self.cores.place(info[pc.task][0], pc.rel, cl) for _ in range(pc.blocks))
        else:
            pc.done = pc.rel
        self.push(pc.done, self.FIN, pc)

    def mark(self, g, lo, hi, at):
        for i in range(lo, hi):
            self.known[g][i] = min(self.known[g][i], at)
        w = self.wm[g]
        while w < stream_len[g] and self.known[g][w] != INF:
            self.wm_at[g] = max(self.wm_at[g], self.known[g][w])
            self.passed[g][w] = self.wm_at[g]
            w += 1
        self.wm[g] = w
        while self.wait[g] and self.wait[g][0][0] <= w:
            _, _, pc = heapq.heappop(self.wait[g])
            self.try_release(pc, at)

    def notify(self, pc, t):
        for d in pc.deps:
            d.unmet -= 1
            d.conds_at = max(d.conds_at, t)
            if d.unmet == 0:
                self.try_release(d, t)
        for pr, lat in pc.announce:
            pr.unmet -= 1
            pr.cond_at = max(pr.cond_at, t + lat)
            if pr.unmet == 0:
                self.push(pr.cond_at, self.KNOWN, pr)

    def finish(self, pc, t):
        self.makespan = max(self.makespan, t)
        self.mark(pc.g, pc.lo, pc.hi, t)
        if SHARED:
            for g in range(N_Q):
                if g != pc.g:
                    self.push(t + RING, self.MARK, (g, pc.lo, pc.hi))
        self.notify(pc, t)
        key = (pc.p, pc.g)
        self.part_left[key] -= 1
        if self.part_left[key] == 0:
            self.part_done[key] = t
            self.gate(pc.p + GATE, t)
        if pc.kind == "slot":
            self.cond_left[pc.p] -= 1
            if self.cond_left[pc.p] == 0:
                self.push(t, self.DEC, pc.p)

    def gate(self, p, t):
        if GATE == 0 or p >= NP:
            return
        if all((p - GATE, h) in self.part_done for h in range(N_Q)):
            at = max(self.part_done[(p - GATE, h)] for h in range(N_Q)) + ENTER_US
            for g in range(N_Q):
                self.push(at, self.RES, (p, g))

    def make_resident(self, key, t):
        if key in self.resident_at:
            return
        self.resident_at[key] = t
        for pc in members[key]:
            if pc.kind == "proxy":
                continue
            at = max(t, order[pc.task] / RATE)
            if at > t:
                self.push(at, self.RES, pc)
            else:
                pc.resident = True
                self.try_release(pc, t)

    def land(self, x, t):
        """A proxy, or a mirror of a producer's pieces, is known here from `t`."""
        x.known = x.done = t
        if x.lo >= 0:
            held = self.passed[x.g][x.lo - 1] if x.lo > 0 else 0.0
            if self.wm[x.g] == x.lo and t > held:
                self.proxy_hold_us[x.g] += t - held
            self.mark(x.g, x.lo, x.hi, t)
        self.notify(x, t)

    def run(self):
        for p in range(NP if GATE == 0 else min(GATE, NP)):
            for g in range(N_Q):
                self.push(0.0, self.RES, (p, g))
        for pc in self.all:
            if pc.kind == "proxy" and pc.unmet == 0:
                self.push(0.0, self.KNOWN, pc)
        while self.ev:
            t, kind, _, x = heapq.heappop(self.ev)
            if kind == self.RES:
                if isinstance(x, tuple):
                    self.make_resident(x, t)
                else:
                    x.resident = True
                    self.try_release(x, t)
            elif kind == self.FIN:
                self.finish(x, t)
            elif kind == self.KNOWN:
                self.land(x, t)
            elif kind == self.MARK:
                self.mark(x[0], x[1], x[2], t)
            elif kind == self.DEC:
                for g in range(N_Q):
                    if (x, g) in dyn_ids:
                        self.mark(g, dyn_next[(x, g)], dyn_ids[(x, g)][1], t)
        unfinished = sum(1 for pc in self.all if pc.kind != "proxy" and pc.done == INF)
        return self.makespan, unfinished


res = []
for s in range(SEEDS):
    r = Run(s)
    res.append((r.run(), r))
span = sum(x[0][0] for x in res) / len(res)
seeds = [round(x[0][0] / 1e3, 3) for x in res]
r0 = res[0][1]
unfinished = res[0][0][1]
print(
    f"{span / 1e3:.3f} ms  SHARED={SHARED} H_IN={H_IN} RING={RING} NAMED={NAMED} LOOK={LOOK:g} BARRIER={BARRIER} "
    f"GATE={GATE} ENTER_US={ENTER_US} EXPERTS={EXPERTS} DYN={DYN} DYN_AT={DYN_AT} DYN_MGR={DYN_MGR} "
    f"SKIP_IDX={SKIP_IDX} "
    f"unfinished={unfinished} seeds={seeds} "
    f"la_need={r0.la_need} waits(thr,la,bar)={r0.waits} wait_us={[round(x) for x in r0.wait_us]} "
    f"proxy_hold_us={[round(x) for x in r0.proxy_hold_us]}"
)
