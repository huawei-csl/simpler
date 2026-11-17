// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//
// The AICPU half of the AICPU <-> AICore ping-pong. Uploaded through the
// dispatcher bootstrap path and launched with rtsLaunchCpuKernel, exactly as
// in aicpu-pingpong.
//
// Ordering, and why there is no cache maintenance here: on a5
// cache_invalidate_range / cache_flush_range are empty no-ops
// (src/a5/platform/onboard/aicpu/cache_ops.cpp:15-23) because GM is coherent;
// the comment there says "callers provide the ordering required by their
// protocols". So this file provides it with rmb()/wmb() and nothing else,
// which is the same thing the production PMU collector does when it reads an
// AICore-written GM slot (src/a5/platform/shared/aicpu/pmu_collector_aicpu.cpp).
//
// The counter is a plain volatile word, never a std::atomic: GM-side atomics
// from device glibc trap, and the parity gate already guarantees that only one
// side is eligible to write at any moment, so no read-modify-write is needed.
// The BSS atomics below are AICPU-local role bookkeeping, not GM.

#include <sched.h>

#include <atomic>
#include <cstdint>

#include "../shared/pingpong.h"

namespace {

// dsb ld / dsb st, matching src/common/platform/include/common/memory_barrier.h.
#if defined(__aarch64__)
inline void Rmb() { __asm__ __volatile__("dsb ld" ::: "memory"); }
inline void Wmb() { __asm__ __volatile__("dsb st" ::: "memory"); }
#else
inline void Rmb() { std::atomic_thread_fence(std::memory_order_acquire); }
inline void Wmb() { std::atomic_thread_fence(std::memory_order_release); }
#endif

inline uint64_t TimerTicks() {
    uint64_t v = 0;
#if defined(__aarch64__)
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
#endif
    return v;
}

inline uint64_t TimerFrequencyHz() {
    uint64_t v = 0;
#if defined(__aarch64__)
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v));
#endif
    return v;
}

// AICPU-local, not GM. Guards against CANN over-subscribing a cpu_id: two
// threads can land on the same cpu, and only one may take a role.
alignas(128) std::atomic<int32_t> s_role_taken[2] = {};
alignas(128) std::atomic<uint32_t> s_claim{0};

struct KernelArgs {
    uint64_t _pad[5];
    void *device_args;
};

}  // namespace

extern "C" {

__attribute__((visibility("default"))) int simpler_aicpu_init(void *args) {
    (void)args;
    s_claim.store(0, std::memory_order_relaxed);
    s_role_taken[0].store(0, std::memory_order_relaxed);
    s_role_taken[1].store(0, std::memory_order_release);
    return 0;
}

__attribute__((visibility("default"))) int simpler_aicpu_run(void *args) {
    if (args == nullptr) return 1;
    KernelArgs *ka = static_cast<KernelArgs *>(args);
    PpDeviceArgs *d = static_cast<PpDeviceArgs *>(ka->device_args);
    if (d == nullptr || d->arena_addr == 0) return 1;
    PpArena *arena = reinterpret_cast<PpArena *>(d->arena_addr);

    const int32_t cpu = static_cast<int32_t>(sched_getcpu());

    // Full-matrix sweep. Round r names its pair in pair_list[r]. A thread
    // serves only the rounds naming its own cpu_id, so exactly one AICPU
    // thread and one AICore block are in the hot loop at any instant; the
    // rest are parked.
    const uint32_t n_cpus = arena->ctl.n_cpus;
    const uint32_t n_rounds = arena->ctl.n_rounds;
    int32_t my_idx = -1;
    for (uint32_t i = 0; i < n_cpus && i < PP_MAX_CPUS; ++i) {
        if (arena->ctl.cpu_list[i] == cpu) {
            my_idx = static_cast<int32_t>(i);
            break;
        }
    }
    if (my_idx < 0) return 0;

    // CANN can put two threads on one cpu_id; only one may take the role.
    int32_t expected = 0;
    if (my_idx < 2) {
        if (!s_role_taken[my_idx].compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) return 0;
    } else {
        const uint32_t bit = s_claim.fetch_or(1u << my_idx, std::memory_order_acq_rel);
        if ((bit & (1u << my_idx)) != 0u) return 0;
    }

    PpCpuState *probe = &arena->cpu_state[my_idx];
    {
        const uint32_t n = arena->ctl.clock_iters;
        uint64_t acc = TimerTicks() | 1ull;
        const uint64_t c0 = TimerTicks();
        for (uint32_t i = 0; i < n; ++i) {
            acc ^= acc << 13;
            acc ^= acc >> 7;
            acc ^= acc << 17;
        }
        const uint64_t c1 = TimerTicks();
        probe->clock_ticks = c1 - c0;
        probe->sink = acc;
    }

    volatile uint64_t *mine = &arena->cpu_ball[0].token;
    volatile uint64_t *peer = &arena->core_ball[0].token;
    const uint64_t budget = arena->ctl.spin_budget;
    const uint64_t iters = arena->ctl.iters;
    const uint64_t warmup = arena->ctl.warmup;
    const uint64_t role_tag = static_cast<uint64_t>(my_idx) << 56;
    uint64_t counter = 0;
    uint64_t served = 0;
    uint32_t stop_reason = 0;  // 0 = schedule completed, 1 = deadline
    const uint64_t deadline_ticks = arena->ctl.aicpu_deadline_ticks;
    const uint64_t t_start = TimerTicks();

    auto trip = [&](uint64_t extra) -> bool {
        ++counter;
        const uint64_t token = role_tag | counter | extra;
        *mine = token;
        Wmb();
        uint64_t spins = 0;
        while (*peer != token) {
            if (++spins > budget) return false;
        }
        Rmb();
        return true;
    };

    while (true) {
        if (deadline_ticks != 0 && TimerTicks() - t_start > deadline_ticks) {
            stop_reason = 1;
            break;
        }
        Rmb();
        const uint64_t r = arena->sched.round;
        if (r >= static_cast<uint64_t>(n_rounds)) break;
        const uint32_t entry = arena->pair_list[r];
        if ((entry >> 16) != static_cast<uint32_t>(my_idx)) {
            // Parked: another cpu owns this round. Yielding is forbidden on
            // AICPU, so spin, but only on a counter -- no GM traffic.
            for (volatile uint64_t k = 0; k < 2000; ++k) {
            }
            continue;
        }

        if (served == 0) {
            const uint32_t steps = arena->ctl.chase_steps;
            uint64_t idx = 0;
            const uint64_t h0 = TimerTicks();
            for (uint32_t i = 0; i < steps; ++i) {
                idx = arena->chase[idx * 8ull];
            }
            const uint64_t h1 = TimerTicks();
            probe->chase_ticks = h1 - h0;
            probe->sink ^= idx;
        }

        uint32_t rc = 0;
        for (uint64_t i = 0; i < warmup; ++i) {
            if (!trip(0)) { rc = PP_RC_RUN_BUDGET; break; }
        }
        uint64_t elapsed = 0;
        if (rc == 0) {
            const uint64_t t0 = TimerTicks();
            for (uint64_t i = 0; i < iters; ++i) {
                if (!trip(0)) { rc = PP_RC_RUN_BUDGET; break; }
            }
            elapsed = TimerTicks() - t0;
        }
        // End-bit token releases the AICore's mirror loop for this round.
        (void)trip(PP_TOKEN_END_BIT);
        // Retire the token before the round advances. A partner enters a round
        // with a published value of 0, so a live token left in this word is
        // taken for that round's own traffic -- and an end-bit one ends the
        // round before its first trip. Consecutive rounds may name the same
        // slot, whose block is not throttled, so it reaches the new round
        // ahead of the AICPU that owns it.
        *mine = 0;
        Wmb();

        if (r < PP_MAX_ROUNDS) {
            arena->result_ticks[r] = elapsed;
            arena->result_rc[r] = rc;
            arena->result_slot[r] = arena->slot_list[entry & 0xFFFFu];
            arena->result_cpu[r] = cpu;
        }
        ++served;
        Wmb();
        arena->sched.round = r + 1;  // single writer at this instant: only this thread is active
        Wmb();
    }

    PpCpuState *out = probe;
    out->cpu_id = cpu;
    out->elapsed_ticks = served;
    out->trips = counter;
    out->timer_hz = TimerFrequencyHz();
    out->rc = stop_reason;
    out->attended = PP_ATTEND_MAGIC;
    Wmb();
    return 0;
}

}  // extern "C"
