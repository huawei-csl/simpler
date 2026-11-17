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
// Two modes, selected by DeviceArgs::mode:
//   0 (discover)  every launched thread records sched_getcpu() and returns.
//   1 (pingpong)  only the threads that landed on cpu_a / cpu_b take part; every
//                 other thread returns before touching any shared state, so the
//                 measured pair runs with the rest of the pool idle.
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

namespace {

constexpr uint32_t kMaxSlots = 32;
constexpr uint32_t kMaxReps = 16;

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

// The measured loop. Identical in shape to the reference algorithm --
// relaxed load, relaxed fetch_add -- with a bounded spin so a vanished partner
// ends the op instead of burning an AICPU core until the 45 s op-execute
// timeout kills aicpu-sd. The guard is a register increment and compare, the
// same on both roles, against a loop whose cost is a cross-core coherence miss.
// MPIDR_EL1 is EL1-only and traps from a user process, so the AICPU OS's sysfs
// topology view is the safe route. A missing file returns -1 rather than
// failing the op: on a device whose driver has already withdrawn CPU_TOPO,
// "absent" is a result worth reporting, not an error.
inline int32_t ReadSysfsTopology(int32_t cpu, const char *leaf) {
    char path[128];
    std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, leaf);
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    char buf[32] = {0};
    const ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return -1;
    return static_cast<int32_t>(strtol(buf, nullptr, 10));
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
template <bool kRelax>
inline bool RunHalf(uint64_t want_parity, uint64_t n, uint64_t budget) {
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t spins = 0;
        while ((s_shared.load(std::memory_order_relaxed) & 1ULL) != want_parity) {
            if (kRelax) CpuRelax();
            if (++spins > budget) return false;
        }
        s_shared.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) int simpler_aicpu_init(void *args) {
    (void)args;
    s_shared.store(0, std::memory_order_relaxed);
    s_arrived.store(0, std::memory_order_relaxed);
    s_claim.store(0, std::memory_order_relaxed);
    s_role_taken[0].store(0, std::memory_order_relaxed);
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
            out->thread_core_id[idx] = ReadSysfsTopology(cpu, "core_id");
            out->thread_package_id[idx] = ReadSysfsTopology(cpu, "physical_package_id");
            out->thread_cluster_id[idx] = ReadSysfsTopology(cpu, "cluster_id");
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

    if (role == 0) {
        out->cpu_a_start = cpu;
        out->role_a_seen = 1;
        out->freq_hz = TimerFrequencyHz();
    } else {
        out->cpu_b_start = cpu;
        out->role_b_seen = 1;
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

    // Each phase performs 2*(warmup+iters) increments, an even number, so the
    // counter parity on exit matches entry and phase 1 chains onto phase 0 with
    // no extra rendezvous. Both roles walk the phases in the same order.
    for (uint32_t phase = 0;
         phase < 2 && (status & (kStatusPartnerAbsent | kStatusSpinBudget)) == 0;
         ++phase) {
        const bool relax = (phase == 1);
        uint64_t *dst = relax ? out->ticks_relax : out->ticks;
        uint32_t *done = relax ? &out->reps_done_relax : &out->reps_done;
        for (uint32_t r = 0; r < reps; ++r) {
            if (!(relax ? RunHalf<true>(want, warmup, budget)
                        : RunHalf<false>(want, warmup, budget))) {
                status |= kStatusSpinBudget;
                break;
            }
            uint64_t base = 0;
            uint64_t t0 = 0;
            if (role == 0) {
                // It is role 0's turn, so the partner is parked in its spin and
                // this value cannot move until role 0 increments it.
                base = s_shared.load(std::memory_order_relaxed);
                t0 = TimerTicks();
            }
            if (!(relax ? RunHalf<true>(want, iters, budget)
                        : RunHalf<false>(want, iters, budget))) {
                status |= kStatusSpinBudget;
                break;
            }
            if (role == 0) {
                // Close the final half-trip so the interval spans exactly
                // `iters` complete round trips. This one wait stays bare in
                // both phases; it is 1 transfer in `iters`.
                const uint64_t target = base + 2 * iters;
                uint64_t spins = 0;
                while (s_shared.load(std::memory_order_relaxed) != target) {
                    if (++spins > budget) {
                        status |= kStatusSpinBudget;
                        break;
                    }
                }
                dst[r] = TimerTicks() - t0;
                *done = r + 1;
                if ((status & kStatusSpinBudget) != 0) break;
            }
        }
    }

    const int32_t cpu_end = static_cast<int32_t>(sched_getcpu());
    if (cpu_end != cpu) status |= kStatusMigrated;

    if (role == 0) {
        out->cpu_a_end = cpu_end;
        out->status_a = status;
        out->final_counter = s_shared.load(std::memory_order_relaxed);
    } else {
        out->cpu_b_end = cpu_end;
        out->status_b = status;
    }
    return 0;
}

}  // extern "C"
