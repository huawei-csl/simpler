/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
//
// aicpu_pingpong — inner SO measuring the atomic ping-pong round-trip latency
// between a named pair of AICPU cpu_ids.
//
// Three modes, selected by DeviceArgs::mode:
//   0 (discover)  every launched thread records sched_getcpu() and returns,
//                 alongside the AICPU OS's own cpu inventory.
//   1 (pingpong)  only the threads that landed on cpu_a / cpu_b take part; every
//                 other thread returns before touching any shared state, so the
//                 measured pair runs with the rest of the pool idle.
//   2 (lines)     the same pair, one timed interval per line of a 64-line bank,
//                 so a cost that tracks the counter's address is separable from
//                 one that tracks the two cores.
//
// The shared counter lives in this SO's BSS, not in GM: GM-side atomics from
// device glibc trap (see aicpu-thread-spread), and BSS is ordinary cacheable
// memory, so the measurement reflects the AICPU coherence fabric rather than a
// device-memory access path. Each hot variable is 128-byte aligned so no two
// of them share a cache line.

#include <fcntl.h>
#include <sched.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr uint32_t kMaxSlots = 32;
constexpr uint32_t kMaxReps = 16;
// 3 dependent ops per iteration, so ~600k core cycles -- hundreds of
// microseconds against a 1 ns counter tick.
constexpr uint64_t kProbeOps = 200000;
// 128 B apart and 8 KiB in total, so the bank covers two pages at a stride wide
// enough that no two counters share a line.
constexpr uint32_t kBankLines = 64;

struct DeviceArgs {
    uint64_t reserved_pre[12];  // 0..95   — bootstrap slots, unused on this path
    uint64_t output_addr;       // 96
    uint64_t mode;              // 104
    uint64_t cpu_a;             // 112
    uint64_t cpu_b;             // 120
    uint64_t iters;             // 128
    uint64_t warmup;            // 136
    uint64_t reps;              // 144
    uint64_t spin_budget;       // 152
};
static_assert(sizeof(DeviceArgs) == 160, "DeviceArgs must match the host's 160-byte block");

struct KernelArgs {
    uint64_t _pad[5];
    void *device_args;
};

#pragma pack(push, 4)
struct PingPongOutput {
    uint32_t claim_counter;
    int32_t status_a;  // bit0 partner absent, bit1 spin budget hit, bit2 migrated
    int32_t status_b;
    int32_t role_a_seen;
    int32_t role_b_seen;
    int32_t cpu_a_start;
    int32_t cpu_a_end;
    int32_t cpu_b_start;
    int32_t cpu_b_end;
    uint32_t reps_done;
    uint64_t freq_hz;
    uint64_t final_counter;
    uint64_t ticks[kMaxReps];
    int32_t thread_cpu[kMaxSlots];
    uint32_t reps_done_relax;
    uint64_t ticks_relax[kMaxReps];
    // Discovery mode only. The AICPU OS's sysfs view of its own topology, which
    // is the only route to cpu_id -> cluster/die left once the driver answers
    // DRV_ERROR_NOT_SUPPORT for CPU_TOPO. -1 means the file was absent or
    // unreadable, which is itself the finding.
    int32_t thread_core_id[kMaxSlots];
    int32_t thread_package_id[kMaxSlots];
    int32_t thread_cluster_id[kMaxSlots];
    int32_t thread_die_id[kMaxSlots];
    // The AICPU OS's own cpu inventory, read once by the first claiming thread.
    // The dispatched set is a CANN scheduling decision and is a lower bound on
    // the pool; `present` and `online` are the OS's count, so the two together
    // separate "this part has fewer AICPU cores" from "CANN dispatches threads
    // to fewer of them". An empty string means the file was unreadable.
    char cpu_present[64];
    char cpu_online[64];
    char cpu_possible[64];
    // What the OS believes about its own cores, as opposed to what latency
    // implies: the sibling lists give the package and SMT grouping, the cache
    // lines give which cores share which level, and the node lists say whether
    // the OS models any of it as NUMA at all.
    char node_info[512];
    char slot_topo[kMaxSlots][112];
    // Measurement mode. A dependent-ALU chain each role times against the fixed
    // 1 GHz counter inside the measured launch: a hand-off cost that moves while
    // this stays put did not move because the core clock moved. 0 means the
    // probe did not run. The cpufreq samples bracket the phases; -1 is an absent
    // file, which on a device OS without cpufreq is the normal answer.
    uint64_t probe_ops;
    uint64_t probe_ticks_a;
    uint64_t probe_ticks_b;
    // Where the contended line sits. The inner SO is loaded per process, so its
    // BSS lands on a fresh physical page each run; a cost that is constant
    // within a process and redrawn between them is a property of the line's
    // address rather than of the cores. pfn 0 means pagemap withheld it.
    uint64_t shared_va;
    uint64_t shared_pfn;
    uint64_t page_bytes;
    int32_t khz_a_start;
    int32_t khz_a_end;
    int32_t khz_b_start;
    int32_t khz_b_end;
    // Mode 2. One round-trip time per line of the bank, same pair and same
    // launch, so the only variable between entries is the line's address.
    uint64_t bank_va;
    uint32_t lines_done;
    uint64_t line_ticks[kBankLines];
};
#pragma pack(pop)

constexpr int32_t kStatusPartnerAbsent = 1;
constexpr int32_t kStatusSpinBudget = 2;
constexpr int32_t kStatusMigrated = 4;

extern "C" void DlogRecord(int moduleId, int level, const char *fmt, ...);
constexpr int kDlogModuleCcecpu = 3;
constexpr int kDlogLevelError = 3;
void DiagLog(const char *msg) { DlogRecord(kDlogModuleCcecpu, kDlogLevelError, "[aicpu-pingpong] %s", msg); }

// Each on its own cache line: false sharing between the ping-pong counter and
// the rendezvous flags would be measured as pair latency.
alignas(128) std::atomic<uint64_t> s_shared{0};

struct alignas(128) BankLine {
    std::atomic<uint64_t> v;
    char pad[128 - sizeof(std::atomic<uint64_t>)];
};
BankLine s_bank[kBankLines];
static_assert(sizeof(BankLine) == 128, "one counter per cache line");

alignas(128) std::atomic<int32_t> s_arrived{0};
alignas(128) std::atomic<uint32_t> s_claim{0};
alignas(128) std::atomic<int32_t> s_role_taken[2] = {};

inline uint64_t TimerTicks() {
    uint64_t v = 0;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}

inline uint64_t TimerFrequencyHz() {
    uint64_t v = 0;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v));
    return v;
}

// MPIDR_EL1 is EL1-only and traps from a user process, so the AICPU OS's sysfs
// view is the safe route to its own topology. A missing file reads as -1 or as
// an empty string rather than failing the op: on a device whose driver has
// already withdrawn CPU_TOPO, "absent" is a result worth reporting.
inline bool ReadSysfsText(const char *path, char *out, size_t cap) {
    out[0] = '\0';
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return false;
    const ssize_t n = read(fd, out, cap - 1);
    close(fd);
    if (n <= 0) return false;
    out[n] = '\0';
    for (ssize_t i = n - 1; i >= 0 && (out[i] == '\n' || out[i] == ' '); --i)
        out[i] = '\0';
    return true;
}

inline int32_t ReadSysfsCpuInt(int32_t cpu, const char *rel) {
    char path[160];
    std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/%s", cpu, rel);
    char buf[32];
    if (!ReadSysfsText(path, buf, sizeof(buf))) return -1;
    return static_cast<int32_t>(strtol(buf, nullptr, 10));
}

// cpufreq exposes the governor's view under scaling_cur_freq and the hardware's
// under cpuinfo_cur_freq; a kernel built without cpufreq has neither.
inline int32_t ReadCpuFreqKhz(int32_t cpu) {
    const int32_t v = ReadSysfsCpuInt(cpu, "cpufreq/scaling_cur_freq");
    if (v >= 0) return v;
    return ReadSysfsCpuInt(cpu, "cpufreq/cpuinfo_cur_freq");
}

// pagemap reports the frame only to a caller with CAP_SYS_ADMIN; anyone else
// reads the present bit and zeros.
inline uint64_t ReadPfn(const void *va, uint64_t page_bytes) {
    const int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) return 0;
    uint64_t ent = 0;
    const off_t off = (off_t)(((uint64_t)(uintptr_t)va / page_bytes) * 8ULL);
    const ssize_t n = pread(fd, &ent, sizeof(ent), off);
    close(fd);
    if (n != (ssize_t)sizeof(ent)) return 0;
    if ((ent & (1ULL << 63)) == 0) return 0;
    return ent & ((1ULL << 55) - 1);
}

// "online=<list> n<id>[<cpus>]=<MiB> ..." for every node the OS publishes. A
// node with no cpus is memory the OS reaches but schedules nothing on.
inline void DescribeNodes(char *out, size_t cap) {
    out[0] = '\0';
    size_t n = 0;
    char buf[64];
    char path[160];
    if (ReadSysfsText("/sys/devices/system/node/online", buf, sizeof(buf)))
        n += (size_t)std::snprintf(out + n, cap - n, "online=%s", buf);
    for (int id = 0; id < 64 && n + 1 < cap; ++id) {
        std::snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", id);
        char cpus[32];
        if (!ReadSysfsText(path, cpus, sizeof(cpus))) continue;
        unsigned long long mib = 0;
        std::snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/meminfo", id);
        char mem[256];
        if (ReadSysfsText(path, mem, sizeof(mem))) {
            const char *p = std::strstr(mem, "MemTotal:");
            if (p != nullptr) mib = strtoull(p + 9, nullptr, 10) / 1024ULL;
        }
        const int w = std::snprintf(out + n, cap - n, " n%d[%s]=%lluM", id, cpus[0] ? cpus : "-", mib);
        if (w > 0) n += (size_t)w;
        if (n >= cap) n = cap - 1;
    }
}

// "sib=<package> thr=<smt> L<n><d|i|u>:<size>@<sharers> ..." for one cpu. A
// level whose files are absent is left out, so an empty string means the OS
// publishes no cache or sibling topology for this core.
inline void DescribeCpu(int32_t cpu, char *out, size_t cap) {
    out[0] = '\0';
    size_t n = 0;
    char buf[48];
    char path[160];
    auto append = [&](const char *fmt, const char *a, const char *b) {
        if (n + 1 >= cap) return;
        const int w = std::snprintf(out + n, cap - n, fmt, a, b);
        if (w > 0) n += (size_t)w;
        if (n >= cap) n = cap - 1;
    };
    std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/core_siblings_list", cpu);
    if (ReadSysfsText(path, buf, sizeof(buf))) append("sib=%s%s", buf, "");
    std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
    if (ReadSysfsText(path, buf, sizeof(buf))) append(" thr=%s%s", buf, "");
    for (int idx = 0; idx < 4; ++idx) {
        char lvl[8];
        char type[24];
        char size[16];
        char shared[32];
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index%d/level", cpu, idx);
        if (!ReadSysfsText(path, lvl, sizeof(lvl))) continue;
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index%d/type", cpu, idx);
        if (!ReadSysfsText(path, type, sizeof(type))) type[0] = '\0';
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index%d/size", cpu, idx);
        if (!ReadSysfsText(path, size, sizeof(size))) size[0] = '\0';
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index%d/shared_cpu_list", cpu, idx);
        if (!ReadSysfsText(path, shared, sizeof(shared))) shared[0] = '\0';
        char tag[40];
        std::snprintf(tag, sizeof(tag), " L%s%c:%s", lvl, type[0] ? (char)(type[0] | 0x20) : 'u', size[0] ? size : "?");
        append("%s@%s", tag, shared[0] ? shared : "?");
    }
}

// A latency-bound chain: every EOR consumes the previous one's result, so the
// loop retires at one dependent op per core cycle however wide the core is.
inline uint64_t ProbeAluTicks(uint64_t ops) {
#if defined(__aarch64__)
    uint64_t x = 0x9E3779B97F4A7C15ULL;
    const uint64_t t0 = TimerTicks();
    for (uint64_t i = 0; i < ops; ++i) {
        __asm__ volatile("eor %0, %0, %0, lsl #13\n\t"
                         "eor %0, %0, %0, lsr #7\n\t"
                         "eor %0, %0, %0, lsl #17\n\t"
                         : "+r"(x));
    }
    return TimerTicks() - t0;
#else
    (void)ops;
    return 0;
#endif
}

// ARMv8 YIELD is architecturally a hint and a NOP on most implementations, so
// ISB is the instruction that actually costs the spinning thread issue slots --
// which is the whole point of a relax hint on an SMT-capable core.
inline void CpuRelax() {
#if defined(__aarch64__)
    __asm__ __volatile__("isb" ::: "memory");
#else
    std::atomic_signal_fence(std::memory_order_acquire);
#endif
}

// kRelax selects the spin body. Both instantiations are measured in the SAME
// launch: the fabric clock DVFS's between launches by up to 40%, so a hint
// whose effect is a few hundred cycles is only resolvable against a partner
// measurement taken while the clock is held.
// The measured loop. Identical in shape to the reference algorithm -- relaxed
// load, relaxed fetch_add -- with a bounded spin so a vanished partner ends the
// op instead of burning an AICPU core until the 45 s op-execute timeout kills
// aicpu-sd. The guard is a register increment and compare, the same on both
// roles, against a loop whose cost is a cross-core coherence miss.
template <bool kRelax>
inline bool RunHalfOn(std::atomic<uint64_t> &word, uint64_t want_parity, uint64_t n, uint64_t budget) {
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t spins = 0;
        while ((word.load(std::memory_order_relaxed) & 1ULL) != want_parity) {
            if (kRelax) CpuRelax();
            if (++spins > budget) return false;
        }
        word.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

// One timed interval on one counter: `warmup` untimed round trips, then exactly
// `iters` complete ones. Role 0 latches the counter and the clock while it is
// its own turn -- the partner is parked in its spin, so the value cannot move --
// and stops once the partner's final increment lands. Only role 0 returns a
// time; role 1 returns 0 and exists to answer.
template <bool kRelax>
inline uint64_t MeasureRoundTrips(
    std::atomic<uint64_t> &word, int role, uint64_t want, uint64_t warmup, uint64_t iters, uint64_t budget,
    int32_t *status
) {
    if (!RunHalfOn<kRelax>(word, want, warmup, budget)) {
        *status |= kStatusSpinBudget;
        return 0;
    }
    uint64_t base = 0;
    uint64_t t0 = 0;
    if (role == 0) {
        base = word.load(std::memory_order_relaxed);
        t0 = TimerTicks();
    }
    if (!RunHalfOn<kRelax>(word, want, iters, budget)) {
        *status |= kStatusSpinBudget;
        return 0;
    }
    if (role != 0) return 0;
    // Close the final half-trip so the interval spans exactly `iters` complete
    // round trips. This one wait stays bare in both phases; it is 1 transfer in
    // `iters`.
    const uint64_t target = base + 2 * iters;
    uint64_t spins = 0;
    while (word.load(std::memory_order_relaxed) != target) {
        if (++spins > budget) {
            *status |= kStatusSpinBudget;
            return 0;
        }
    }
    return TimerTicks() - t0;
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) int simpler_aicpu_init(void *args) {
    (void)args;
    s_shared.store(0, std::memory_order_relaxed);
    s_arrived.store(0, std::memory_order_relaxed);
    s_claim.store(0, std::memory_order_relaxed);
    s_role_taken[0].store(0, std::memory_order_relaxed);
    for (uint32_t i = 0; i < kBankLines; ++i)
        s_bank[i].v.store(0, std::memory_order_relaxed);
    s_role_taken[1].store(0, std::memory_order_release);
    return 0;
}

__attribute__((visibility("default"))) int simpler_aicpu_pingpong(void *args) {
    if (args == nullptr) {
        DiagLog("args==nullptr");
        return 1;
    }
    auto *k = reinterpret_cast<KernelArgs *>(args);
    auto *d = reinterpret_cast<DeviceArgs *>(k->device_args);
    if (d == nullptr || d->output_addr == 0) {
        DiagLog("missing GM I/O");
        return 1;
    }
    auto *out = reinterpret_cast<PingPongOutput *>(d->output_addr);
    const int32_t cpu = static_cast<int32_t>(sched_getcpu());

    if (d->mode == 0) {
        const uint32_t idx = s_claim.fetch_add(1u, std::memory_order_acq_rel);
        if (idx < kMaxSlots) {
            out->thread_cpu[idx] = cpu;
            out->thread_core_id[idx] = ReadSysfsCpuInt(cpu, "topology/core_id");
            out->thread_package_id[idx] = ReadSysfsCpuInt(cpu, "topology/physical_package_id");
            out->thread_cluster_id[idx] = ReadSysfsCpuInt(cpu, "topology/cluster_id");
            out->thread_die_id[idx] = ReadSysfsCpuInt(cpu, "topology/die_id");
            if (idx == 0) {
                ReadSysfsText("/sys/devices/system/cpu/present", out->cpu_present, sizeof(out->cpu_present));
                ReadSysfsText("/sys/devices/system/cpu/online", out->cpu_online, sizeof(out->cpu_online));
                ReadSysfsText("/sys/devices/system/cpu/possible", out->cpu_possible, sizeof(out->cpu_possible));
                DescribeNodes(out->node_info, sizeof(out->node_info));
            }
            DescribeCpu(cpu, out->slot_topo[idx], sizeof(out->slot_topo[idx]));
            out->claim_counter = idx + 1;
        }
        return 0;
    }

    const int32_t cpu_a = static_cast<int32_t>(d->cpu_a);
    const int32_t cpu_b = static_cast<int32_t>(d->cpu_b);
    int role;
    if (cpu == cpu_a) {
        role = 0;
    } else if (cpu == cpu_b) {
        role = 1;
    } else {
        return 0;  // not in this pair: leave before touching any shared line
    }

    // CANN over-subscribes a sink cpu when launch_count exceeds the user pool.
    // A second thread landing on a pair cpu_id must not become a second role.
    int32_t expected = 0;
    if (!s_role_taken[role].compare_exchange_strong(
            expected, 1, std::memory_order_acq_rel, std::memory_order_acquire
        )) {
        return 0;
    }

    const uint64_t iters = d->iters;
    const uint64_t warmup = d->warmup;
    const uint64_t budget = d->spin_budget;
    uint32_t reps = static_cast<uint32_t>(d->reps);
    if (reps > kMaxReps) reps = kMaxReps;

    const uint64_t probe_ticks = ProbeAluTicks(kProbeOps);
    const int32_t khz_start = ReadCpuFreqKhz(cpu);
    if (role == 0) {
        out->cpu_a_start = cpu;
        out->role_a_seen = 1;
        out->freq_hz = TimerFrequencyHz();
        out->probe_ops = kProbeOps;
        out->probe_ticks_a = probe_ticks;
        out->khz_a_start = khz_start;
        out->page_bytes = (uint64_t)sysconf(_SC_PAGESIZE);
        out->shared_va = (uint64_t)(uintptr_t)&s_shared;
        out->shared_pfn = ReadPfn(&s_shared, out->page_bytes);
    } else {
        out->cpu_b_start = cpu;
        out->role_b_seen = 1;
        out->probe_ticks_b = probe_ticks;
        out->khz_b_start = khz_start;
    }

    int32_t status = 0;
    const uint64_t want = static_cast<uint64_t>(role);

    s_arrived.fetch_add(1, std::memory_order_acq_rel);
    uint64_t guard = 0;
    while (s_arrived.load(std::memory_order_acquire) < 2) {
        if (++guard > budget) {
            status |= kStatusPartnerAbsent;
            break;
        }
    }

    if (d->mode == 2) {
        // Each line is a fresh counter at 0, so the roles re-synchronise on it
        // without a rendezvous: role 0 owns parity 0 and goes first.
        if (role == 0) out->bank_va = (uint64_t)(uintptr_t)&s_bank[0];
        for (uint32_t l = 0; l < kBankLines && (status & (kStatusPartnerAbsent | kStatusSpinBudget)) == 0; ++l) {
            const uint64_t t = MeasureRoundTrips<false>(s_bank[l].v, role, want, warmup, iters, budget, &status);
            if (role == 0 && (status & kStatusSpinBudget) == 0) {
                out->line_ticks[l] = t;
                out->lines_done = l + 1;
            }
        }
    } else {
        // Each phase performs 2*(warmup+iters) increments per rep, an even
        // number, so the counter parity on exit matches entry and phase 1 chains
        // onto phase 0 with no extra rendezvous. Both roles walk the phases in
        // the same order.
        for (uint32_t phase = 0; phase < 2 && (status & (kStatusPartnerAbsent | kStatusSpinBudget)) == 0; ++phase) {
            const bool relax = (phase == 1);
            uint64_t *dst = relax ? out->ticks_relax : out->ticks;
            uint32_t *done = relax ? &out->reps_done_relax : &out->reps_done;
            for (uint32_t r = 0; r < reps; ++r) {
                const uint64_t t = relax ?
                                       MeasureRoundTrips<true>(s_shared, role, want, warmup, iters, budget, &status) :
                                       MeasureRoundTrips<false>(s_shared, role, want, warmup, iters, budget, &status);
                if ((status & kStatusSpinBudget) != 0) break;
                if (role == 0) {
                    dst[r] = t;
                    *done = r + 1;
                }
            }
        }
    }

    const int32_t cpu_end = static_cast<int32_t>(sched_getcpu());
    const int32_t khz_end = ReadCpuFreqKhz(cpu_end);
    if (cpu_end != cpu) status |= kStatusMigrated;

    if (role == 0) {
        out->khz_a_end = khz_end;
        out->cpu_a_end = cpu_end;
        out->status_a = status;
        out->final_counter = s_shared.load(std::memory_order_relaxed);
    } else {
        out->khz_b_end = khz_end;
        out->cpu_b_end = cpu_end;
        out->status_b = status;
    }
    return 0;
}

}  // extern "C"
