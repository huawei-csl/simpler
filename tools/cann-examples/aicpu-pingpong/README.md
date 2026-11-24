# aicpu-pingpong

Measures the **atomic ping-pong round-trip latency between every pair of AICPU
cpu_ids** and prints the result as a matrix. Answers "how expensive is it for
two AICPU cores to hand a cache line back and forth", which is the floor on any
cross-core AICPU handshake — scheduler-to-scheduler notification, work stealing,
a ticket lock, a shared ready-queue index.

Shares the launch path with
[`aicpu-thread-spread`](../aicpu-thread-spread/README.md) (dispatcher bootstrap
+ inner SO via `rtsBinaryLoadFromFile` + `rtsLaunchCpuKernel`); the difference
is the inner SO runs the measurement and the host sweeps pairs.

## What it measures

Each participating thread runs the classic two-thread parity hand-off:

```cpp
for (uint64_t i = 0; i < iters; ++i) {
    while ((shared.load(std::memory_order_relaxed) & 1) != role) { /* spin */ }
    shared.fetch_add(1, std::memory_order_relaxed);
}
```

`role` is 0 for the thread on `cpu_a` and 1 for the thread on `cpu_b`, so the
counter can only advance by alternating between the two cores. One **round
trip** is both threads incrementing once (2 `fetch_add`s, one full there-and-back
line transfer). The reported number is `elapsed / iters`; halve it for a one-way
hand-off.

Timing is `cntvct_el0` read by role 0, bracketing exactly `iters` complete round
trips: role 0 latches the counter and the clock while it is its own turn (so the
partner is parked and the value cannot move), and stops only once the partner's
final increment lands.

## Why the numbers mean something

Four design points, each of which would otherwise corrupt the measurement:

- **The shared counter lives in the inner SO's BSS, not in GM.** GM-side atomics
  from device glibc trap (noted in `aicpu-thread-spread`), and GM is the wrong
  thing to measure anyway — BSS is ordinary cacheable memory, so what is timed
  is the AICPU coherence fabric rather than a device-memory access path.
- **Only the measured pair runs.** Every thread reads `sched_getcpu()` and
  returns immediately unless it landed on `cpu_a` or `cpu_b`, before touching any
  shared line. The rest of the pool is idle for the whole measurement rather than
  spinning on a barrier, so no spectator traffic reaches the fabric.
- **Each hot variable is 128-byte aligned.** False sharing between the ping-pong
  counter and the rendezvous flags would show up as pair latency.
- **The spin is bounded.** A vanished partner ends the op with a status bit
  instead of burning an AICPU core until the 45 s op-execute timeout kills
  `aicpu-sd`. The guard is a register increment and compare, identical on both
  roles, against a loop whose cost is a cross-core coherence miss.

Threads re-read `sched_getcpu()` at the end; a mismatch sets a `migrated` status
bit, so a sample whose thread moved cores mid-measurement is visible rather than
silently mislabelled.

## Protocol

```text
1. discovery launch (mode=0)   every thread records sched_getcpu() -> unique cpu_id set,
                               plus the AICPU OS's own /sys cpu inventory
2. for each unordered pair:
     launch simpler_aicpu_init with 1 thread   (resets the BSS counters)
     launch simpler_aicpu_pingpong with N      (only the pair participates)
     read back reps x round-trip tick counts
3. for each unordered pair (mode=2):
     one timed interval on each of 64 counters 128 B apart, in a single launch
```

A `reps`-long run inside one launch amortizes launch overhead; the host reports
both the minimum and the median across reps.

**Discovery also prints what the OS believes about its own cores** — the node
list with each node's cpus and memory, the package and SMT sibling lists, and
which cores share each cache level. On both measured a5 parts that view is
flat: one package, no SMT, one CPU NUMA node holding every core, and a
last-level cache shared by all of them, beside 8-10 cpu-less memory nodes that
sum to the chip's HBM. The 2-core grouping the latency finds appears nowhere in
it, so the matrix is the only way to see it.

**The discovered set is a lower bound on the pool.** It is where CANN chose to
put threads, so a part could have cores it never dispatches to. The inventory
from step 1 is the AICPU OS's own count and is the upper bound; when the two
agree, as they do on `_9592` (`present` = 0-7, 7 user-schedulable) and on
`_957b` (`present` = 0-3, 3 user-schedulable), the pool is the hardware. Raising
`launch_count` past the pool makes CANN over-subscribe, and the duplicates in
the printed spread are the direct evidence that it is exhausted.

**Step 3 is what separates a per-pair cost from a per-address one.** Step 2
measures one counter, so a cost that depends on where that counter sits is
indistinguishable from a cost that depends on the two cores. Walking 64 lines
inside one launch holds the pair, the clock and the process fixed and varies
only the address. On both measured a5 parts the cross-cluster round trip varies
by 400-540 ns across those lines while the same-cluster one varies by under
5 ns, so step 2 alone is not a pair measurement — see
[`../a5-latency-findings/`](../a5-latency-findings/README.md).

**Each role also times a dependent-ALU chain inside the measured launch**
(`probe_a` / `probe_b`, microseconds). It reads 800.1 µs ± 0.3 on both parts
across every launch taken so far, which is what rules out DVFS as the source of
a hand-off number that moves. A probe that moves with the hand-off would mean
the opposite, so read the two columns together. `cpufreq` is absent on the AICPU
OS, so the probe is the only clock witness; the per-cpu `scaling_cur_freq` read
prints only when the file exists.

## Build

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

(cd tools/cann-examples/aicpu-pingpong/device && \
   cmake -B build -S . \
     -DCMAKE_C_COMPILER=${ASCEND_HOME_PATH}/tools/hcc/bin/aarch64-target-linux-gnu-gcc \
     -DCMAKE_CXX_COMPILER=${ASCEND_HOME_PATH}/tools/hcc/bin/aarch64-target-linux-gnu-g++ && \
   cmake --build build)

(cd tools/cann-examples/aicpu-pingpong/host && cmake -B build -S . && cmake --build build)
```

`tools/build_cann_examples.sh` builds it with everything else — the layout is
the discovered `device/` + `host/` shape, so no workflow needs to name it.

## Run

```bash
REPO=$PWD
export SIMPLER_DISPATCHER_SO=$REPO/build/lib/a5/dispatcher/libsimpler_aicpu_dispatcher.so
export SIMPLER_AICPU_PINGPONG_SO=$REPO/tools/cann-examples/aicpu-pingpong/device/build/libaicpu_pingpong.so

# Always through task-submit on a shared box (.claude/rules/running-onboard.md).
task-submit --device 0 --max-time 1800 \
    --run "$REPO/tools/cann-examples/aicpu-pingpong/host/build/aicpu_pingpong 0 0 1000 5 200"
```

Arguments are `<device_id> [launch_count] [iters] [reps] [warmup]`;
`launch_count=0` means "launch one thread per discovered cpu_id".

## Reading the output

The matrix is symmetric (a round trip has no direction) with an empty diagonal —
one thread per cpu_id means a core has no partner on itself.

Latency clusters by topology, but only the diagonal blocks are reproducible: a
single sweep's off-diagonal cells are one draw of a quantity that spans hundreds
of ns, so compare matrices only as floors over many sweeps, or use the line
sweep. a5 derives `cluster_id = phy_cpu_id / 2` and `die_id = phy_cpu_id / 4` (see
[`src/a5/docs/hardware.md`](../../../src/a5/docs/hardware.md)), so expect bands:
SMT siblings fastest, then same-cluster, then same-die, then cross-die. Reading
the bands off the matrix is a way to **recover the AICPU topology from timing**
on a SKU whose `CPU_TOPO` the driver refuses to report — which is the case on
newer a5 drivers, where `halGetDeviceInfoByBuff(SYSTEM, CPU_TOPO)` returns
`DRV_ERROR_NOT_SUPPORT`.

The line sweep prints its 64 samples per pair plus `min / median / max /
spread`, and `LINE,<cpu_a>,<cpu_b>,<line>,<ns>` rows for analysis. `lines N/64`
says how many completed: a pair whose partner stalls past the spin budget stops
the walk there rather than reporting zeros.

Per-pair status is printed next to each sample:

| status | meaning |
| ------ | ------- |
| `ok` | both roles ran, no migration |
| `partner-absent` | no thread landed on the other cpu_id this launch |
| `spin-budget` | the partner stopped advancing the counter mid-run |
| `migrated` | the AICPU OS moved a thread to a different cpu mid-measurement |
| `[role-missing]` | one of the two cpu_ids got no thread at all |

## Scope and limits

- The stock AICPU cross toolchain (gcc 7.3, `-march=armv8-a`) has no LSE, so
  `fetch_add` compiles to an `ldxr`/`stxr` pair rather than `ldadd`. The protocol
  serializes the two threads, so there is no RMW contention and the cost is
  dominated by the line transfer — but the number is for LL/SC code, which is
  what AICPU C++ actually compiles to here.
- Only the AICPU user pool is probed. AICore-to-AICore and AICPU-to-AICore
  signalling are different paths with different costs.
- One thread per cpu_id, so there is no same-core (SMT-sibling-free) diagonal
  sample.
- The counter's physical page is not observable: `/proc/self/pagemap` returns
  the present bit and a zero frame to this process, and the SO's load address is
  the same in every process, so the line sweep shows *that* the address decides
  the cost without naming which address bits do it.
