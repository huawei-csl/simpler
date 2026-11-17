# Running the AICPU ping-pong matrix on another system

Hand-off notes for reproducing this measurement on a different box / different
Ascend part. Assumes you already know how to build and run `simpler`.

**What you are producing:** a symmetric matrix of atomic ping-pong round-trip
latency (ns) between every pair of user-schedulable AICPU `cpu_id`s on your
device, plus the `OCCUPY` mask and `cpu_id` list the pool actually resolved to.

The matrix is the point, but the *reason* it is worth running is that the
latency bands recover the AICPU topology — SMT siblings, cluster-mates, die
boundary — on parts whose `CPU_TOPO` the driver refuses to report. Newer a5
drivers return `DRV_ERROR_NOT_SUPPORT` for both
`halGetDeviceInfoByBuff(SYSTEM, CPU_TOPO)` and
`dsmi_get_device_info(SOC_INFO, CPU_TOPO)`, so timing is the only route left.

Read [`README.md`](README.md) first — it covers what is measured, why the
protocol is sound, the build, and the status flags. This file only covers what
is different when you are not on the box it was written on.

## 0. Get the tool — it is not committed

`tools/cann-examples/aicpu-pingpong/` is untracked. Copy the whole directory to
the target repo at the same path:

```bash
# from the source box
tar czf pingpong.tgz -C <repo>/tools/cann-examples aicpu-pingpong
# on the target box
tar xzf pingpong.tgz -C <repo>/tools/cann-examples
```

Nothing in it is arch-specific. The dispatcher SO comes from
`SIMPLER_DISPATCHER_SO` at run time, and `aicpu-thread-spread` (whose bootstrap
path this shares verbatim) documents that **a5 and a2a3 both accept the same
inner SO body**. It has only ever been *run* on a5, so treat a2a3 as expected to
work rather than known to.

`tools/build_cann_examples.sh` picks it up automatically — the `device/` +
`host/` layout is what that script discovers, so no workflow needs to name it.

## 1. Discovery first — never assume the cpu_id list

This is the step most likely to surprise you, and everything downstream depends
on it.

Run with `launch_count = 0`. The tool does a discovery launch, has every thread
report `sched_getcpu()`, and sweeps only the `cpu_id`s that actually came back.

Two things make a hardcoded list wrong:

- **The user pool is a per-SKU mask, not a core count.** Four different shapes
  are already on record for a5 alone: `OCCUPY = 0x3e` (5 user cpus), `0xfe`
  (7), `0x1f8` (6, at cpu_id 3..8), `0x7ffe` (14). Same 8 physical cores
  underneath each time.
- **CANN over-subscribes rather than widening the pool.** Launching more threads
  than the pool holds does not reach more cpus; the surplus piles onto one sink
  cpu. On the box this was written on, 16 launched threads produced the spread
  `1 1 1 1 2 2 3 3 4 4 5 5 6 6 7 7` — seven distinct cpus, cpu1 four times.

Report the raw spread line, not just the unique set. The duplicate pattern tells
the next reader how the dispatcher behaved.

## 2. The sweep

```bash
REPO=<repo>
export SIMPLER_DISPATCHER_SO=$REPO/build/lib/<arch>/dispatcher/libsimpler_aicpu_dispatcher.so
export SIMPLER_AICPU_PINGPONG_SO=$REPO/tools/cann-examples/aicpu-pingpong/device/build/libaicpu_pingpong.so

$REPO/tools/cann-examples/aicpu-pingpong/host/build/aicpu_pingpong <device_id> 0 1000 15 200
```

Arguments: `<device_id> [launch_count] [iters] [reps] [warmup]`.
`launch_count = 0` means "one thread per discovered cpu_id".

If your box arbitrates devices with a queue (`task-submit` or equivalent), wrap
it — this holds an AICPU core pair for the whole sweep.

## 3. Repeat whole sweeps, not just reps

**Within** one launch the measurement is extremely stable: min and median across
reps agreed to 0.01 ns on every pair. **Between** launches it is not. Two sweeps
on the same box, same binary, minutes apart:

| pair | sweep A | sweep B |
| ---- | ------- | ------- |
| (1,2) | 1114 ns | 1560 ns |
| (4,6) | 1418 ns | 1114 ns |

So raising `reps` buys you nothing you do not already have. Run the **whole
sweep** 5–6 times and report the spread across sweeps. A single sweep's matrix
looks far more precise than it is.

The intra-cluster (fast-tier) numbers were stable across everything; it is the
cross-cluster band that wanders.

## 4. Reading the bands

Expect tiers, fastest first: SMT siblings → same cluster → same die → cross die.
Where the steps fall tells you the topology. a5's runtime derives
`cluster_id = phy_cpu_id / 2` and `die_id = phy_cpu_id / 4`
(`src/a5/platform/onboard/host/aicpu_topology_probe.h`); your part may differ.

Two cautions learned the hard way:

- **A cpu that shares a cluster with the AICPU OS core reads slow in every
  pairing.** On the reference box cpu0 is the OS and cpu1 is its cluster-mate;
  every cpu1 pair is inflated, and cpu1↔cluster-mate-of-another-cluster came out
  *slower than crossing the die boundary*. Do not fit a topology model before
  identifying which cpu_id the OS owns (`AICPU + OS_SCHED`, device-side).
- **Fast tier ≈ 96 ns one-way is a cluster/L3 number, not an L1/SMT number.** If
  your part exposes real SMT siblings, expect a tier well below that. If your
  fastest tier is ~96 ns and you assumed SMT, you have probably found
  cluster-mates instead. `MPIDR_EL1` settles it directly — Aff0 = thread,
  Aff1 = core, Aff2 = cluster — and it is a register read, so it works even when
  `CPU_TOPO` is unsupported. That probe is **not** in the tool yet.

## 5. Traps

- **Check the timer frequency the tool prints** (`Timer: cntvct_el0 @ <hz>`).
  It was 1 GHz on the reference part. The bracket covers all `iters` round trips
  at once so a coarser timer is survivable, but if one batch lands under ~1000
  ticks, raise `iters`.
- **The shared counter must stay in the inner SO's BSS.** GM-side atomics from
  device glibc trap, and GM is the wrong thing to time anyway. If you port the
  counter to a GM slot to "simplify", the tool stops measuring the coherence
  fabric and starts measuring a device-memory path — or dies.
- **"Only two threads at a time" is enforced by early return, not by a
  barrier.** Non-participants read `sched_getcpu()` and return before touching
  any shared line, so there is no spectator traffic on the fabric. A role CAS
  additionally guards against CANN landing two threads on the same pair cpu_id.
  If you are asked whether the constraint held, that is the answer.
- **Absolute numbers are LL/SC numbers.** The stock AICPU cross toolchain is
  gcc 7.3 at `-march=armv8-a`, so `fetch_add` becomes `ldxr`/`stxr`, not `ldadd`.
  The protocol serializes the two threads so there is no RMW contention, but if
  your toolchain has LSE the comparison is not apples to apples — record the
  compiler and `-march`.
- **Trust the status column.** `partner-absent`, `spin-budget`, `migrated` and
  `[role-missing]` each mean the number next to them is not a clean pair
  measurement. The reference run was `ok/ok` on all 21 pairs; if yours is not,
  say so rather than averaging it away.

## 6. What to send back

So the two sets are comparable:

1. SoC name (CANN ini `Short_SoC_version` / the ACL device name).
2. `OCCUPY` (host-side **and** device-side if you can get both — on a5 they
   differ) and `OS_SCHED`.
3. The discovery spread line, verbatim.
4. The `CSV,a,b,min,median` block from each sweep — all of them, not a merged
   matrix.
5. The timer frequency line.
6. Compiler version and `-march` used for the inner SO.
7. `CPU_TOPO` if your driver still serves it. That makes your run the ground
   truth that calibrates everyone else's timing-only inference.

## 7. Reference result to compare against

`Ascend950DT_9592`, device 0, `OCCUPY = 0xfe`, 7 user cpus at cpu_id 1..7,
`iters=1000 reps=5 warmup=200`, median ns per **round trip** (halve for one-way):

```text
   cpu       1        2        3        4        5        6        7
     1       -   1561.5   1560.0   1392.5   1392.9   1377.7   1379.8
     2  1561.5       -     193.0   1300.0   1300.1   1300.1   1300.1
     3  1560.0   193.0       -    1300.0   1300.0   1300.1   1300.0
     4  1392.5  1300.0   1300.0       -     192.3   1114.3   1114.3
     5  1392.9  1300.1   1300.0   192.3       -    1114.3   1114.3
     6  1377.7  1300.1   1300.1  1114.3   1114.3       -     192.3
     7  1379.8  1300.1   1300.0  1114.3   1114.3   192.3        -
```

Read as: 8 physical AICPU cores in 4 clusters of 2 across 2 dies, cpu0 owned by
the AICPU OS. ~192 ns intra-cluster, ~1114 ns same-die cross-cluster, ~1300 ns
cross-die, and the cpu1 row inflated by sharing its cluster's L3 with the OS
core. This matches the published a5 spec — 4 AI CPU clusters × 2 dual-threaded
Linx816 with 4 MB L3 each, and a 128 MB chip-wide unified L2 — which the matrix
reproduced without any topology API.
