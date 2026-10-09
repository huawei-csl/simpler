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
#include "aicpu/asimgq_core.h"
#include "aicpu/platform_aicpu_affinity.h"

#include <vector>

#include "aicpu/device_time.h"
#include "common/platform_config.h"

namespace asimgq {

namespace {

// Injected latencies and the compute duration, in the sys-cnt tick domain.
uint64_t g_pickup_ticks = 0;       // core receives a task -> its kernel begins
uint64_t g_link_ticks = 0;         // controller <-> one of its cores, one way
uint64_t g_core_poll_ticks = 0;    // task at a core -> the core's loop reads it
uint64_t g_arrive_ticks = 0;       // submitted entry -> legible to the controller
uint64_t g_push_ticks = 0;         // manager's posted write into a queue
uint64_t g_fin_ticks = 0;          // core finish -> its FIN signal is raised
uint64_t g_report_ticks = 0;       // that fact -> legible in the manager's register
uint64_t g_queue_poll_ticks = 0;   // one read of that register
uint64_t g_mix_push_ticks = 0;     // controller's per-entry push of a mix group
uint64_t g_mix_arrival_ticks = 0;  // controller -> core, per mix entry
uint64_t g_cancel_ticks = 0;       // controller -> core, one way, for a cancellation

std::vector<uint64_t> g_func_compute_ticks;        // index = func_id, value = mean compute ticks
std::vector<uint64_t> g_func_compute_sigma_ticks;  // index = func_id, value = compute stddev ticks
uint64_t g_default_compute_ticks = 0;

// What modelling has cost, per queue, in counter ticks, and the per-task tallies
// the end-of-run lines report. Deliberately outside SimQueue: a run resets the
// device being modelled, but the cost of running the model is the simulator's own
// ledger and must accumulate monotonically. Resetting it inside a measured phase
// makes the phase's delta go negative, and the window then silently keeps the
// overhead it was supposed to shed.
// `overrun_ticks` is also the queue's clock offset: the model reads real time less
// it, so it must only ever grow while a run is in flight. Each queue's ledger is
// written by its own scheduler thread on every access, so each has a cache line of
// its own.
struct alignas(64) SimLedger {
    uint64_t overrun_ticks = 0;
    uint64_t calib_hits = 0;
    uint64_t calib_misses = 0;
    uint64_t func_hist[16] = {};
    uint64_t func_other = 0;
    int32_t func_max = -1;
};
SimLedger g_ledger[SIM_MAX_QUEUES];

// Register backing, written only at bring-up: the handshake reads each core's
// COND to see an initialised, idle fabric. Nothing reads it afterwards, because
// under the GroupQueue a manager never asks a core anything.
uint64_t g_reg_base = 0;
uint32_t g_num_cores = 0;

uint64_t ns_to_ticks(uint64_t ns) {
    const uint64_t freq = get_sys_cnt_aicpu_frequency_hz();
    return static_cast<uint64_t>(static_cast<__uint128_t>(ns) * freq / 1'000'000'000ULL);
}

// Active-wait until the sys-cnt clock reaches `deadline`. Spin (never sleep/yield
// to the OS — this is the dispatch path); a bare architecture hint is the only
// pause. The caller is held here so the modeled latency is real elapsed
// wall-clock, exactly as the AICPU is held waiting on a real nGnRE access.
void wait_until(uint64_t deadline) {
    while (get_sys_cnt_aicpu() < deadline) {
#if defined(__aarch64__)
        __asm__ volatile("yield");
#endif
    }
}

// One simulated compute core. It is a sequential server with a two-deep intake:
// it can hold a running task plus one pipelined behind it, and the pipelined one
// starts the instant the running one ends. Its ACK and FIN are what move it
// between the controller's sets; nothing else advances it, because its committed
// ends are known at the moment the work is committed.
struct SimCore {
    uint64_t ready_at = 0;                         // when the last committed task ends
    uint64_t finish[2] = {0, 0};                   // committed ends, oldest first
    uint64_t index[2] = {UINT64_MAX, UINT64_MAX};  // their positions in the manager's queue
    // A gated slot holds a cohort member: its end is UINT64_MAX until the cohort
    // is released, so it never comes due, and its duration is kept here because
    // it was drawn when the member was placed rather than when it starts.
    uint64_t gated_dur[2] = {0, 0};
    // Earliest a gated member could start: when its core comes free. Co-residency
    // is about starting together, not about the core being idle when the member
    // was placed, so a member may sit behind running work and still be co-resident
    // once the cohort's release passes that point.
    uint64_t gated_floor[2] = {0, 0};
    // How many of this core's slots hold a member that has not started. A core
    // holding one takes nothing further, and `ready_at` stays the real end of its
    // running work rather than being poisoned with a sentinel: the sentinel is a
    // marker in `finish`, never a time to compute with.
    uint32_t gated_count = 0;
    uint32_t outstanding = 0;  // 0..2
    // The task sitting in the pipeline slot, for as long as it is still
    // cancellable. `pipe_at` is when the controller pushed it, which orders
    // victim selection; `pipe_dur` is the compute already drawn for it, so a
    // stolen task keeps the duration it would have had; `pipe_task_id` is what a
    // cancellation names, so a core can refuse one that raced a promotion and
    // now refers to a different task. A mix part is never stealable: its halves
    // share one package's local memory.
    uint64_t pipe_at = 0;
    uint64_t pipe_dur = 0;
    int32_t pipe_task_id = -1;
    bool pipe_stealable = false;
};

constexpr uint64_t SIM_GATED = UINT64_MAX;

// Rings a queue keeps, indexed by SimTaskType. Empty is unused, so the array
// carries one slot per real shape plus that hole.
constexpr uint32_t SIM_RING_COUNT = 4;

uint32_t ring_of(SimTaskType t) { return static_cast<uint32_t>(t); }

// A set of the group's cores. The controller keeps its core states as these
// sets rather than as a field it walks, because picking a target is then a
// priority encode over a mask instead of a sweep of every package.
struct CoreMask {
    uint64_t w[2] = {0, 0};

    void set(uint32_t i) { w[i >> 6] |= (1ULL << (i & 63)); }
    void clear(uint32_t i) { w[i >> 6] &= ~(1ULL << (i & 63)); }
    bool test(uint32_t i) const { return ((w[i >> 6] >> (i & 63)) & 1ULL) != 0; }
    bool any() const { return (w[0] | w[1]) != 0; }
};

// Visit the cores in a set, lowest first, touching only the ones that are in it.
// A queue's event set is sparse whenever its cores are mostly idle -- which is
// exactly the workload where a poll retires many positions and the sweep is
// repeated for each -- so stepping the set bits rather than testing every slot is
// what keeps the simulator's own cost inside the latency it models.
template <typename F>
void for_each_core(const CoreMask &m, F f) {
    for (uint32_t w = 0; w < 2; ++w) {
        uint64_t bits = m.w[w];
        while (bits != 0) {
            const uint32_t b = static_cast<uint32_t>(__builtin_ctzll(bits));
            bits &= bits - 1;
            f(w * 64 + b);
        }
    }
}

CoreMask mask_and(const CoreMask &a, const CoreMask &b) {
    CoreMask r;
    r.w[0] = a.w[0] & b.w[0];
    r.w[1] = a.w[1] & b.w[1];
    return r;
}

// Lowest set bit at or after `from`, wrapping at `n`. A rotating priority
// encoder: the controller spreads work over its cores instead of always
// loading the lowest-numbered one. Returns UINT32_MAX when the set is empty.
uint32_t select_rotating(const CoreMask &m, uint32_t from, uint32_t n) {
    if (!m.any() || n == 0) {
        return UINT32_MAX;
    }
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t i = (from + k) % n;
        if (m.test(i)) {
            return i;
        }
    }
    return UINT32_MAX;
}

// One AI package: three cores that share local memory, which is why a mix group
// has to land wholly inside one. It is no longer an arbitration unit — the
// controller places single tasks on cores directly. Its cores are named by their
// slot in the owning queue, because that is the only place they exist.
struct SimPackage {
    uint32_t aic = UINT32_MAX;
    uint32_t aiv0 = UINT32_MAX;
    uint32_t aiv1 = UINT32_MAX;
};

// One GroupQueue: the controller for the packages and cores it serves.
//
// It is owned outright by a single AICPU scheduler: no other thread reads or
// writes any of it. That is the point of the design rather than an incidental
// property — a structure shared between schedulers would need a lock to arbitrate,
// and a lock between AICPUs costs more than the work it protects.
// One task waiting on producers in its own group, in a fixed slot so the waiter
// table can name it.
struct SimHeld {
    SimQueueEntryStore e;
    uint64_t seq;        // admission order, which is the order held entries are placed in
    uint32_t remaining;  // producers that have not finished
};

// Held entries are tracked in 32-bit masks, one bit per slot.
static_assert(SIM_HELD_CAP <= 32, "held-slot masks are 32 bits wide");

// Waiter-table keys. A producer a held entry can still be waiting on lies within
// the look-ahead horizon of the watermark, so positions twice that far apart
// never share a key.
constexpr uint32_t SIM_HELD_WAIT_SLOTS = 2 * SIM_AHEAD_HORIZON;

struct alignas(64) SimQueue {
    // Completion accounting. A position's bit is set when the core running it
    // finishes; the watermark is the contiguous prefix over those bits, which is
    // the single value the manager reads instead of polling every core.
    uint64_t watermark = 0;

    // One ready ring per shape. Only the head of a ring is pushed, so a head that
    // has nowhere to go stalls that ring behind it — but a cube that cannot be
    // placed no longer holds up queued vector work, which it did when all shapes
    // shared one FIFO. Rings are indexed by SimTaskType.
    SimQueueEntryStore ring[SIM_RING_COUNT][SIM_QUEUE_DEPTH];
    uint32_t push[SIM_RING_COUNT] = {};
    uint32_t pop[SIM_RING_COUNT] = {};
    // The rotation advances on a successful push, not on the clock, so placement is
    // reproducible and independent of when a read happens to land.
    uint32_t rr = 0;
    // Which of a package's two vector cores a vector task is offered first. `rr`
    // alone cannot decide this: it counts packages, so the slot it names is always
    // a multiple of three and therefore always the package's cube core. The scan
    // below takes the first core at or after that, which for a vector task is the
    // package's aiv0 -- so aiv1 is reached only when aiv0 is already busy, and a
    // workload that never saturates one vector core per package never reaches the
    // second at all.
    uint32_t vrr = 0;

    // How deep each ring may go: the cores of that shape this queue serves, and
    // its packages for mix. A queue that cannot hold more than its cores can run
    // stops its manager from taking work it has nowhere to put, which leaves that
    // work in the shared ready queues for a manager that does.
    uint32_t ring_cap[SIM_RING_COUNT] = {};

    SimPackage packages[SIM_QUEUE_MAX_PACKAGES];
    uint32_t package_count = 0;
    SimCore cores[SIM_QUEUE_MAX_PACKAGES * 3];
    uint32_t core_count = 0;

    // Every core's state, as the controller holds it. There is no frontend
    // controller between the queue and the cores: each core signals its own ACK
    // and FIN here, and these sets are what those signals move. A core is in
    // exactly one of three states, and two of them are sets the controller can
    // draw a target from:
    //
    //   free  — nothing committed; takes a task now
    //   pipe  — one task committed, pipeline slot open; takes a task to run next
    //   (neither) — two committed, or holding an unstarted cohort member
    //
    // Keeping them as sets rather than deriving them is what makes placement a
    // priority encode instead of a walk over every package on every task.
    CoreMask free_m;
    CoreMask pipe_m;
    // When each core's oldest committed end becomes legible to the manager, or
    // UINT64_MAX for a core with none -- an idle core, or one holding a member that
    // has not started. One contiguous array, so finding what came due reads a few
    // cache lines rather than one per core. Valid for slots below `core_count`.
    uint64_t evt_at[SIM_QUEUE_MAX_PACKAGES * 3] = {};
    // Cores running one task with another already behind it. Only these can have a
    // task cancelled, so a queue with none can skip the victim search outright --
    // on a workload that never fills a pipeline slot that search would otherwise
    // sweep every core on every retirement and never find anything.
    CoreMask full_m;
    // Which cores are cube and which are vector, fixed at bring-up. A task is
    // only ever offered cores of its own type.
    CoreMask aic_m;
    CoreMask aiv_m;

    // The soonest committed finish becoming legible. A poll landing before it can
    // return the watermark it already holds: replaying an interval in which
    // nothing happens yields what is already there. This is what makes the common
    // poll a single compare instead of a sweep of the group.
    uint64_t next_event = UINT64_MAX;

    // Tasks handed over before their in-group producers finished, one slot each.
    // A retiring producer finds the entries waiting on it in `held_waiters`, keyed
    // by its position, so placing them costs what retired rather than a pass over
    // everything held.
    SimHeld held[SIM_HELD_CAP] = {};
    uint32_t held_used = 0;   // slots occupied
    uint32_t held_ready = 0;  // slots whose producers have all finished, not yet in a ring
    uint32_t held_n = 0;
    uint64_t held_seq = 0;

    // Cohort members placed on this queue's cores and not yet started, and how many
    // members over the run had to sit behind running work rather than on an idle
    // core -- a manager offers a cohort only cores it holds idle, so that count
    // says how often the two views of a core disagreed.
    uint32_t cohort_staged = 0;
    uint64_t cohort_piped = 0;
    uint32_t held_waiters[SIM_HELD_WAIT_SLOTS] = {};
    uint32_t held_high = 0;
    uint64_t held_admitted = 0;
    uint64_t held_promoted = 0;
    uint64_t held_refused = 0;

    uint64_t rng = 0;

    // What the simulator itself spends inside a poll, against the latency meant
    // to hide it. Per manager, like everything else here, so reading it costs no
    // synchronisation and reports one thread's cost rather than four threads' sum.
    // Where a poll's own work goes, so the cost can be attributed rather than
    // guessed at: retiring ends that came due, and assembling the status register.
    uint64_t poll_retire_ticks = 0;
    uint64_t poll_status_ticks = 0;

    // Work that did not fit inside the latency the call models. Only this inflates
    // a measurement: when the work fits, the spin to the deadline hides it and the
    // call costs exactly what it should. Comparing work against the window instead
    // over-states the problem, and against nothing at all misses it.
    uint64_t poll_overrun_total = 0;
    uint64_t poll_overrun_calls = 0;

    // The same accounting for the push path. A submit models a 5 ns posted write,
    // but the controller work it triggers -- settling the sub-ready array and
    // placing whatever that frees -- is a loop here and a single clock in hardware.
    // Unmeasured, it would inflate every window this arm reports.
    uint64_t push_overrun_total = 0;
    uint64_t push_overrun_calls = 0;
    uint64_t push_work_total = 0;
    uint64_t push_calls = 0;

    uint64_t poll_work_total = 0;
    uint64_t poll_work_max = 0;
    uint64_t poll_work_calls = 0;

    // Core time committed to work, by type. See queue_busy_ticks.
    uint64_t busy_aic = 0;
    uint64_t busy_aiv = 0;

    // When the core running each position actually ended, by position. Read by the
    // manager after it retires one, so it can tell how much of the gap between a
    // task ending and its consumers becoming ready was the trip to the manager.
    // Indexed modulo SIM_FINISH_TS_SLOTS rather than by the full index space: a
    // manager's live window is a few dozen positions, so this stays a page the
    // retire path keeps hot, where an array spanning the whole index space took a
    // cache miss on every retirement -- paid by the simulator, inside the latency
    // it is supposed to be modelling.
    uint64_t finish_at[SIM_FINISH_TS_SLOTS] = {};

    // How long an entry sits between reaching the queue and starting on a core.
    // The manager's own ready -> submit link cannot see this: it ends when the
    // task is handed to the queue, and the wait for a core happens after that.
    // Task stealing: a cancellation the controller sent to a core holding an
    // unacknowledged pipelined task, so that task can run now on a core that has
    // gone idle instead of waiting for the one it was bound to. `won` is a
    // cancellation that beat the core's own promotion; `lost` is one that did
    // not, which leaves the task running where it was. Only one may be in flight
    // at a time, which `cancel_busy_until` enforces.
    uint64_t cancel_busy_until = 0;
    uint64_t steals_tried = 0;
    uint64_t steals_won = 0;
    int64_t steal_gain_ticks = 0;

    // Split of the wait between reaching the queue and running. `ring` is time
    // held unassigned because no core of the type could take it; `bound` is time
    // spent in a core's pipeline slot after the controller committed it. They
    // answer different questions: ring wait says the group was saturated, bound
    // wait says the task was committed to a core earlier than it needed to be.
    uint64_t ring_wait_total = 0;
    uint64_t bound_wait_total = 0;
    uint64_t residency_max = 0;
    uint64_t residency_n = 0;

    // Inter-queue load imbalance, sampled every SIM_IMBALANCE_SAMPLE_POLLS status
    // reads. Each manager samples independently and keeps its own tally, so
    // reading it costs no synchronisation; the four should broadly agree.
    //
    // `spread` is the difference between the deepest and shallowest queue's total
    // ring occupancy at a sample -- how unevenly submitted work is spread over the
    // groups. `starved` counts samples where this queue had an idle core of a type
    // some other queue had queued work for: the part of the imbalance that is
    // actually costing core time, since that work could have run here.
    uint64_t imb_samples = 0;
    uint64_t imb_spread_sum = 0;
    uint32_t imb_spread_max = 0;
    uint64_t imb_own_depth_sum = 0;
    uint64_t imb_deepest = 0;
    uint64_t imb_starved = 0;
    uint32_t imb_tick = 0;

    // How long finished work waits before a poll makes it legible to the manager,
    // summed and worst-case. A core cannot be given its next task until the
    // manager reacts to the one that ended, so this is the slack between a core
    // going idle and the manager being able to do anything about it.
    uint64_t retire_lag_total = 0;
    uint64_t retire_lag_max = 0;
    uint64_t retire_n = 0;

    // Finished positions the prefix has not yet reached, one bit each over a
    // window `SIM_AHEAD_HORIZON` wide; a position beyond it waits for the
    // watermark, as everything did before.
    // Bit i is the completion of `watermark + 1 + i`, held circularly: the window
    // rotates rather than shifts, so advancing it is a moved head instead of a
    // pass over every word. Nothing shifts zeros in behind the head, so a bit the
    // head passes is cleared as it goes -- a stale one would read as a completion
    // that never happened.
    uint64_t ahead_bits[SIM_AHEAD_WORDS] = {};
    uint32_t ahead_base = 0;  // slot holding `watermark + 1`
    // Furthest above the watermark a finish has landed, and how many landed past
    // the window and had to wait for the watermark instead. Together they say
    // whether the window's width is what bounds out-of-order retirement.
    uint32_t ahead_high_water = 0;
    uint64_t ahead_overflow = 0;
};

SimQueue g_queues[SIM_MAX_QUEUES];

// Which queue this is. A controller needs its own index to tag the work it places,
// so a completion can be published to the queue that owns the position.
// xorshift64: 3 shifts + 3 xors, cheap enough to hide inside a modeled latency.
uint64_t next_rand(SimQueue &q) {
    uint64_t x = q.rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    q.rng = x;
    return x;
}

// Per-func_id compute (the kernel a task runs). Falls back to the default when
// the func_id is out of range or has no calibrated entry (0). A calibrated sigma
// draws a per-task duration instead of returning the mean: real cores vary per
// task, and identical durations would finish cores in lockstep batches,
// collapsing the scheduler's dispatch/complete phase count well below the real one.
// A func_id absent from the calibration table is charged the default duration
// rather than a measured one, which is silent in the result. Counted so a run can
// say how much of its compute was actually calibrated.
uint64_t sample_compute_ticks(SimQueue &q, int32_t func_id) {
    SimLedger &l = g_ledger[&q - &g_queues[0]];
    if (func_id >= 0 && func_id < 16) {
        ++l.func_hist[func_id];
    } else {
        ++l.func_other;
        if (func_id > l.func_max) l.func_max = func_id;
    }
    uint64_t mean = g_default_compute_ticks;
    uint64_t sigma = 0;
    bool calibrated = false;
    if (func_id >= 0 && static_cast<size_t>(func_id) < g_func_compute_ticks.size()) {
        const uint64_t t = g_func_compute_ticks[func_id];
        if (t != 0) {
            calibrated = true;
            mean = t;
            if (static_cast<size_t>(func_id) < g_func_compute_sigma_ticks.size()) {
                sigma = g_func_compute_sigma_ticks[func_id];
            }
        }
    }
    if (calibrated) {
        ++l.calib_hits;
    } else {
        ++l.calib_misses;
    }
    if (sigma == 0) {
        return mean;
    }
    // Three uniform draws over [-2^20, 2^20) sum to a mean-0 deviate whose stddev
    // is exactly 2^20 (Irwin-Hall), so scaling by sigma >> 20 gives stddev
    // `sigma`, bounded to +-3 sigma. Integer-only: no float on the path.
    int64_t sum = 0;
    for (int k = 0; k < 3; ++k) {
        sum += static_cast<int64_t>(next_rand(q) & ((1ULL << 21) - 1)) - (1LL << 20);
    }
    int64_t dur = static_cast<int64_t>(mean) + ((static_cast<int64_t>(sigma) * sum) >> 20);
    if (dur < 1) {
        dur = 1;
    }
    return static_cast<uint64_t>(dur);
}

// A producer has finished: count it off every held entry waiting on it.
void wake_held(SimQueue &q, uint64_t index) {
    uint32_t &slot = q.held_waiters[index & (SIM_HELD_WAIT_SLOTS - 1)];
    uint32_t m = slot;
    slot = 0;
    while (m != 0) {
        const uint32_t s = static_cast<uint32_t>(__builtin_ctz(m));
        m &= m - 1;
        if (--q.held[s].remaining == 0) {
            q.held_ready |= 1u << s;
        }
    }
}

// Record that `index` finished. A position below the watermark is already
// covered by the prefix; the watermark's own position advances it; anything
// above sets its bit.
//
// The advance walks the run of completions the prefix now absorbs, clearing each
// as the head passes it, and moves the head by one more than the run -- the extra
// step being the watermark's own position, which the window does not store.
void mark_queue_done(SimQueue &q, uint64_t index) {
    if (index == UINT64_MAX || index < q.watermark) {
        return;
    }
    wake_held(q, index);
    if (index > q.watermark) {
        const uint64_t off = index - q.watermark - 1;
        if (off >= SIM_AHEAD_HORIZON) {
            // Unreachable while the manager honours the horizon when it issues a
            // position. Latched rather than dropped, so a wrong guard shows up
            // instead of silently stranding the task.
            ++q.ahead_overflow;
            return;
        }
        const uint32_t slot = (q.ahead_base + static_cast<uint32_t>(off)) & (SIM_AHEAD_HORIZON - 1);
        q.ahead_bits[slot >> 6] |= (1ULL << (slot & 63));
        const uint32_t reach = static_cast<uint32_t>(off) + 1;
        if (reach > q.ahead_high_water) {
            q.ahead_high_water = reach;
        }
        return;
    }
    uint32_t run = 0;
    while (run < SIM_AHEAD_HORIZON) {
        const uint32_t slot = (q.ahead_base + run) & (SIM_AHEAD_HORIZON - 1);
        const uint64_t bit = 1ULL << (slot & 63);
        if ((q.ahead_bits[slot >> 6] & bit) == 0) {
            break;
        }
        q.ahead_bits[slot >> 6] &= ~bit;
        ++run;
    }
    q.watermark += run + 1;
    q.ahead_base = (q.ahead_base + run + 1) & (SIM_AHEAD_HORIZON - 1);
}

// Move a core into the set its state puts it in. Every path that commits work to
// a core, retires work from it, or starts a cohort member on it ends here, which
// is what keeps the sets and the cores from disagreeing.
void refresh_core(SimQueue &q, uint32_t slot) {
    const SimCore &c = q.cores[slot];
    q.free_m.clear(slot);
    q.pipe_m.clear(slot);
    q.full_m.clear(slot);
    // A core holding a member that has not started takes nothing further: there
    // is no end to queue behind.
    if (c.gated_count == 0) {
        if (c.outstanding == 0) {
            q.free_m.set(slot);
        } else if (c.outstanding == 1) {
            q.pipe_m.set(slot);
        }
    }
    q.evt_at[slot] =
        (c.outstanding > 0 && c.finish[0] != SIM_GATED) ? c.finish[0] + g_fin_ticks + g_link_ticks : UINT64_MAX;
    if (c.outstanding == 2 && c.gated_count == 0 && c.pipe_stealable) {
        q.full_m.set(slot);
    }
}

// Commit one entry to a core, starting no earlier than the core is free and no
// earlier than the entry existed. The end is computed here, once, and never
// revisited — which is what replaces advancing a state machine on every access.
void commit(
    SimQueue &q, uint32_t core_slot, const SimQueueEntryStore &e, uint32_t part, uint64_t start, uint64_t pushed_at
) {
    SimCore &c = q.cores[core_slot];
    if (c.ready_at > start) {
        start = c.ready_at;
    }
    if (e.enqueued_at > start) {
        start = e.enqueued_at;
    }
    const uint64_t dur = sample_compute_ticks(q, e.func_id[part]);
    const bool pipelined = (c.outstanding > 0);
#if ASIMGQ_SELF_PROFILE
    {
        const uint64_t ring = pushed_at > e.enqueued_at ? pushed_at - e.enqueued_at : 0;
        const uint64_t bound = start > pushed_at ? start - pushed_at : 0;
        q.ring_wait_total += ring;
        q.bound_wait_total += bound;
        if (ring + bound > q.residency_max) {
            q.residency_max = ring + bound;
        }
        ++q.residency_n;
    }
#endif
    // Cube cores are the first of each package's three.
    if ((core_slot % 3) == 0) {
        q.busy_aic += dur;
    } else {
        q.busy_aiv += dur;
    }
    if (e.gated) {
        // Placed, holding its core, but not running: it waits for the rest of its
        // cohort. Its duration is drawn now because that is when the entry was
        // read; only its start is deferred.
        c.finish[c.outstanding] = SIM_GATED;
        c.gated_dur[c.outstanding] = dur;
        c.gated_floor[c.outstanding] = start;
        c.index[c.outstanding] = e.index[part];
        ++c.outstanding;
        ++c.gated_count;
        ++q.cohort_staged;
        if (pipelined) {
            ++q.cohort_piped;
        }
        refresh_core(q, core_slot);
        return;
    }
    // Reaching an idle core costs the trip over the controller's link plus the
    // core's own startup before the kernel body -- cache invalidate and ack, which
    // `compute` excludes by construction. A task landing behind running work pays
    // neither: it is already at the core, and its startup overlaps the compute of
    // the task ahead of it, which is what the two-deep intake exists to allow.
    const uint64_t fin = start + (pipelined ? 0 : g_link_ticks + g_core_poll_ticks + g_pickup_ticks) + dur;
    if (pipelined) {
        c.pipe_at = pushed_at;
        c.pipe_dur = dur;
        c.pipe_task_id = e.task_id;
        // A mix part cannot move: its halves share one package's local memory.
        c.pipe_stealable = (e.type != SimTaskType::Mix);
    }
    c.finish[c.outstanding] = fin;
    c.index[c.outstanding] = e.index[part];
    ++c.outstanding;
    c.ready_at = fin;
    refresh_core(q, core_slot);
    const uint64_t legible_at = fin + g_fin_ticks + g_link_ticks;
    if (legible_at < q.next_event) {
        q.next_event = legible_at;
    }
}

// Whether a package can seat a mix group: every one of its three cores must have
// a slot. The group's halves then start as each core frees, one behind whatever
// is running there, so a mix never waits for a wholly idle package.
//
// A gated cohort member takes the same rule. Co-residency is about the blocks
// *starting* together, not about their cores being idle when they were placed, so
// a member may sit behind running work; the cohort's release simply cannot precede
// the moment the last of them comes free. Demanding idle packages instead would
// drain the whole device once per cohort.
bool package_has_room(const SimQueue &q, const SimPackage &p) {
    for (uint32_t core : {p.aic, p.aiv0, p.aiv1}) {
        if (!q.free_m.test(core) && !q.pipe_m.test(core)) {
            return false;
        }
    }
    return true;
}

// When a core can begin the next task pushed to it: now, if it holds nothing, or
// the end of what it is running, if a pipeline slot is all it has left.
uint64_t core_available_at(const SimQueue &q, uint32_t slot) {
    const SimCore &c = q.cores[slot];
    return (c.outstanding == 0) ? c.ready_at : c.finish[0];
}

// Push a mix group onto one package. Its cube and vector halves share that
// package's local memory, so unlike a single task the group cannot be spread over
// whichever cores happen to be free. A package with all three cores idle takes it
// first; failing that, the one whose last part would start earliest.
bool place_mix(SimQueue &q, const SimQueueEntryStore &e, uint64_t at) {
    const uint32_t n = q.package_count;
    uint32_t best = UINT32_MAX;
    uint64_t best_last = UINT64_MAX;
    bool best_idle = false;
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t pi = (q.rr + k) % n;
        const SimPackage &p = q.packages[pi];
        if (!package_has_room(q, p)) {
            continue;
        }
        const bool idle = q.free_m.test(p.aic) && q.free_m.test(p.aiv0) && q.free_m.test(p.aiv1);
        uint64_t last = 0;
        for (uint32_t core : {p.aic, p.aiv0, p.aiv1}) {
            const uint64_t avail = core_available_at(q, core);
            if (avail > last) {
                last = avail;
            }
        }
        const bool better = (best == UINT32_MAX) || (idle && !best_idle) || (idle == best_idle && last < best_last);
        if (better) {
            best = pi;
            best_last = last;
            best_idle = idle;
        }
    }
    if (best == UINT32_MAX) {
        return false;
    }
    const SimPackage &p = q.packages[best];
    const uint32_t slots[SIM_MIX_MAX_ENTRIES] = {p.aic, p.aiv0, p.aiv1};
    for (uint32_t i = 0; i < e.parts; ++i) {
        commit(q, slots[i], e, i, at + (i + 1) * g_mix_push_ticks + g_mix_arrival_ticks, at);
    }
    q.rr = (q.rr + 1) % n;
    return true;
}

// Push a cube or vector task to a core. An entirely free core of that type takes
// it, chosen by a rotating priority encode so work spreads over the group's
// packages rather than piling onto the lowest-numbered core. Only when every core
// of the type is already running something does the task go into a pipeline slot,
// and then behind the core that frees soonest, so it starts as early as any core
// can start it.
//
// The two tiers are ordered rather than collapsed into one earliest-start rule.
// A task landing on an idle core waits the full push-to-latch handshake, where one
// dropped into a pipeline slot is latched the instant the running task ends; an
// earliest-start rule would therefore pipeline behind a core about to finish and
// leave an idle core standing.
bool place_single(SimQueue &q, const SimQueueEntryStore &e, uint64_t at) {
    const CoreMask &type_m = (e.type == SimTaskType::Cube) ? q.aic_m : q.aiv_m;
    // A package occupies three consecutive slots: cube, aiv0, aiv1. Start a cube
    // scan on the package boundary, and a vector scan on one of that package's two
    // vector slots in turn, so both are offered work.
    const uint32_t base = (q.rr * 3) % q.core_count;
    const uint32_t from = (e.type == SimTaskType::Cube) ? base : (base + 1 + (q.vrr & 1u)) % q.core_count;
    const uint32_t idle = select_rotating(mask_and(q.free_m, type_m), from, q.core_count);
    if (idle != UINT32_MAX) {
        commit(q, idle, e, 0, at, at);
        q.rr = (q.rr + 1) % q.package_count;
        if (e.type != SimTaskType::Cube) ++q.vrr;
        return true;
    }
    // A cohort member is placed only on an idle core: one queued behind running work
    // would start when that work ends rather than with its peers, which is not
    // co-residency. It waits at the head of its ring for a core to come free.
    if (e.gated) {
        return false;
    }
    const CoreMask pipe = mask_and(q.pipe_m, type_m);
    uint32_t best = UINT32_MAX;
    uint64_t best_at = UINT64_MAX;
    for_each_core(pipe, [&](uint32_t i) {
        const uint64_t avail = q.cores[i].finish[0];
        if (avail < best_at) {
            best = i;
            best_at = avail;
        }
    });
    if (best == UINT32_MAX) {
        return false;
    }
    commit(q, best, e, 0, best_at < at ? at : best_at, at);
    q.rr = (q.rr + 1) % q.package_count;
    if (e.type != SimTaskType::Cube) ++q.vrr;
    return true;
}

bool rings_hold_work(const SimQueue &q) {
    for (uint32_t r = 0; r < SIM_RING_COUNT; ++r) {
        if (q.push[r] != q.pop[r]) return true;
    }
    return false;
}

// Push each ring's head out until nothing more can be placed.
//
// Mix goes first, strictly. It needs a whole package where a cube or vector needs
// one core, so a controller that always took the cheaper push would starve it.
// Beyond that the rings are independent: a cube that cannot be placed does not
// hold up queued vector work, which it did when all shapes shared one FIFO.
void dispatch_pending(SimQueue &q, uint64_t at) {
    if (q.package_count == 0) {
        return;
    }
    static constexpr SimTaskType kOrder[3] = {SimTaskType::Mix, SimTaskType::Cube, SimTaskType::Vector};
    for (uint32_t guard = 0; guard < SIM_QUEUE_DEPTH * 3; ++guard) {
        bool placed = false;
        for (SimTaskType t : kOrder) {
            const uint32_t r = ring_of(t);
            if (q.push[r] == q.pop[r]) {
                continue;
            }
            const SimQueueEntryStore &head = q.ring[r][q.pop[r] % SIM_QUEUE_DEPTH];
            const bool ok = (t == SimTaskType::Mix) ? place_mix(q, head, at) : place_single(q, head, at);
            if (!ok) {
                continue;  // this ring is head-of-line; the others are not
            }
            ++q.pop[r];
            placed = true;
            break;  // re-offer from the top so mix keeps its priority
        }
        if (!placed) {
            return;
        }
    }
}

// Put a stolen task on a core that has gone idle. Its compute was drawn when it
// was first placed and is carried over unchanged: a steal moves work, it does not
// resample it. The core is idle, so the task pays the full push-to-latch
// handshake it would have avoided by staying in a pipeline slot.
void place_stolen(SimQueue &q, uint32_t core_slot, uint64_t index, uint64_t dur, uint64_t start) {
    SimCore &c = q.cores[core_slot];
    if (c.ready_at > start) {
        start = c.ready_at;
    }
    const bool pipelined = (c.outstanding > 0);
    if ((core_slot % 3) == 0) {
        q.busy_aic += dur;
    } else {
        q.busy_aiv += dur;
    }
    const uint64_t fin = start + (pipelined ? 0 : g_link_ticks + g_core_poll_ticks + g_pickup_ticks) + dur;
    c.finish[c.outstanding] = fin;
    c.index[c.outstanding] = index;
    ++c.outstanding;
    c.ready_at = fin;
    refresh_core(q, core_slot);
    const uint64_t legible_at = fin + g_fin_ticks + g_link_ticks;
    if (legible_at < q.next_event) {
        q.next_event = legible_at;
    }
}

// Take the pipelined task off a core whose cancellation succeeded. The core keeps
// running what it had; only the slot behind it empties. The compute charged for
// the cancelled task is given back, because place_stolen charges it again where
// the task actually runs.
void cancel_pipelined(SimQueue &q, uint32_t v) {
    SimCore &c = q.cores[v];
    if ((v % 3) == 0) {
        q.busy_aic -= c.pipe_dur;
    } else {
        q.busy_aiv -= c.pipe_dur;
    }
    c.finish[1] = 0;
    c.index[1] = UINT64_MAX;
    --c.outstanding;
    c.ready_at = c.finish[0];
    c.pipe_stealable = false;
    c.pipe_task_id = -1;
    refresh_core(q, v);
}

// A core has gone idle with nothing queued for it. Rather than leave it idle
// while a task sits bound to a core that is still busy, the controller asks that
// core to give the task back.
//
// The cancellation races the victim's own promotion, which happens the instant
// its running task ends. It carries the task id it means to cancel, and a core
// honours it only while that exact task is still in its pipeline slot: a
// cancellation that arrives after the promotion refers to a task that is now
// running -- or, if the slot has since been refilled, to a different task
// entirely -- and is refused either way. Losing the race is the benign outcome;
// the task runs where it was.
//
// Excluded from stealing: mix parts, whose halves share one package's local
// memory, and cohort members, which have not started and whose placement the
// rendezvous is counting. Both are held off by `pipe_stealable` and `gated_count`.
bool try_steal(SimQueue &q, uint32_t free_slot, uint64_t at) {
    if (!q.free_m.test(free_slot) || at < q.cancel_busy_until) {
        return false;
    }
    const bool cube = q.aic_m.test(free_slot);
    // A steal is for a core the queue cannot feed, not a substitute for placing
    // work it already holds.
    const uint32_t r = ring_of(cube ? SimTaskType::Cube : SimTaskType::Vector);
    if (q.push[r] != q.pop[r]) {
        return false;
    }
    const CoreMask &type_m = cube ? q.aic_m : q.aiv_m;
    if (!mask_and(q.full_m, type_m).any()) {
        return false;  // nothing is pipelined; no victim exists to ask
    }
    // Most recently pipelined first. All else equal that task is the least likely
    // to have been acknowledged yet, so the cancellation is likeliest to beat the
    // promotion -- and it sits behind the running task with the most left to do,
    // so winning it is also worth the most.
    uint32_t victim = UINT32_MAX;
    uint64_t newest = 0;
    for_each_core(mask_and(q.full_m, type_m), [&](uint32_t i) {
        if (i == free_slot) {
            return;
        }
        const SimCore &c = q.cores[i];
        if (victim == UINT32_MAX || c.pipe_at > newest) {
            newest = c.pipe_at;
            victim = i;
        }
    });
    if (victim == UINT32_MAX) {
        return false;
    }
    ++q.steals_tried;
    q.cancel_busy_until = at + 2 * g_cancel_ticks;

    SimCore &v = q.cores[victim];
    const uint64_t arrives = at + g_cancel_ticks;
    const int32_t want_id = v.pipe_task_id;
    // Promotion is the moment the running task ends. Before it, the slot still
    // holds the task the cancellation names; after it, it does not.
    const bool promoted = (v.finish[0] <= arrives);
    if (promoted || v.pipe_task_id != want_id) {
        return false;
    }
    const uint64_t would_have_started = v.finish[0];
    const uint64_t index = v.index[1];
    const uint64_t dur = v.pipe_dur;
    cancel_pipelined(q, victim);
    // The controller learns the outcome a round trip after it asked.
    const uint64_t start = at + 2 * g_cancel_ticks;
    place_stolen(q, free_slot, index, dur, start);
    ++q.steals_won;
    q.steal_gain_ticks += static_cast<int64_t>(would_have_started) -
                          static_cast<int64_t>(start + g_link_ticks + g_core_poll_ticks + g_pickup_ticks);
    return true;
}

// The soonest committed end that can become legible. Recomputed only when the
// schedule changes, so an ordinary poll is one compare rather than a sweep.
void refresh_next_event(SimQueue &q) {
    uint64_t soonest = UINT64_MAX;
    for (uint32_t i = 0; i < q.core_count; ++i) {
        if (q.evt_at[i] < soonest) {
            soonest = q.evt_at[i];
        }
    }
    q.next_event = soonest;
}

// Move a prepared entry into its ring. `at` is when it is legible to the
// controller: the arrival of a fresh submit, or the retirement instant for an entry
// the controller already held. Returns false when the ring is full.
bool enqueue_entry(SimQueue &q, const SimQueueEntryStore &e, uint64_t at) {
    const uint32_t r = ring_of(e.type);
    if (q.push[r] - q.pop[r] >= q.ring_cap[r]) {
        return false;
    }
    SimQueueEntryStore &dst = q.ring[r][q.push[r] % SIM_QUEUE_DEPTH];
    dst = e;
    dst.enqueued_at = at;
    ++q.push[r];
    return true;
}

// Whether a position this queue was given has finished. The watermark covers the
// contiguous prefix and the look-ahead word carries the finishes above it, so the
// two together are exact across the whole horizon.
bool position_done(const SimQueue &q, uint64_t index) {
    if (index == UINT64_MAX || index < q.watermark) return true;
    if (index == q.watermark) return false;
    const uint64_t off = index - q.watermark - 1;
    if (off >= SIM_AHEAD_HORIZON) return false;
    const uint32_t slot = (q.ahead_base + static_cast<uint32_t>(off)) & (SIM_AHEAD_HORIZON - 1);
    return (q.ahead_bits[slot >> 6] & (1ULL << (slot & 63))) != 0;
}

// Place every held entry whose producers have all finished, oldest admission
// first, while its shape's ring has room. One that does not fit stays ready and is
// offered again at the next retirement or admission.
void promote_held(SimQueue &q, uint64_t at) {
    uint32_t pending = q.held_ready;
    while (pending != 0) {
        uint32_t s = static_cast<uint32_t>(__builtin_ctz(pending));
        for (uint32_t m = pending & (pending - 1); m != 0; m &= m - 1) {
            const uint32_t t = static_cast<uint32_t>(__builtin_ctz(m));
            if (q.held[t].seq < q.held[s].seq) s = t;
        }
        pending &= ~(1u << s);
        SimHeld &h = q.held[s];
        if (enqueue_entry(q, h.e, at > h.e.enqueued_at ? at : h.e.enqueued_at)) {
            q.held_ready &= ~(1u << s);
            q.held_used &= ~(1u << s);
            --q.held_n;
            ++q.held_promoted;
        }
    }
}

// The cores whose oldest end is legible by `upto`, in the order they retire: by
// time, and by core among equal times -- the order repeatedly taking the minimum
// would visit them in. Returns how many.
uint32_t collect_due(const SimQueue &q, uint64_t upto, uint32_t *due) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < q.core_count; ++i) {
        const uint64_t at = q.evt_at[i];
        if (at > upto) {
            continue;
        }
        uint32_t j = n++;
        while (j > 0 && q.evt_at[due[j - 1]] > at) {
            due[j] = due[j - 1];
            --j;
        }
        due[j] = i;
    }
    return n;
}

// Retire one core's oldest end at the instant it became legible.
void retire_core(SimQueue &q, uint32_t core, uint64_t due_at, [[maybe_unused]] uint64_t upto) {
#if ASIMGQ_SELF_PROFILE
    const uint64_t lag = upto - due_at;
    q.retire_lag_total += lag;
    if (lag > q.retire_lag_max) {
        q.retire_lag_max = lag;
    }
    ++q.retire_n;
#endif
    SimCore &c = q.cores[core];
#if ASIMGQ_SELF_PROFILE
    if (c.index[0] != UINT64_MAX) {
        q.finish_at[c.index[0] & (SIM_FINISH_TS_SLOTS - 1)] = c.finish[0];
    }
#endif
    mark_queue_done(q, c.index[0]);
    // The producers that have finished are now recorded, so anything held for
    // them can be placed -- before the core below is refreshed, so a core freed
    // by this same retirement can take it in the same step.
    promote_held(q, due_at);
    c.finish[0] = c.finish[1];
    c.index[0] = c.index[1];
    --c.outstanding;
    refresh_core(q, core);
    dispatch_pending(q, due_at);
    // The core may have come fully free with nothing to give it. Ask a core
    // that is still holding an unacknowledged task to hand it over.
    if (q.free_m.test(core)) {
        try_steal(q, core, due_at);
    }
}

// Retire every committed end legible by `upto`, oldest first. A core's FIN is
// what moves it back into a set the controller can draw from, and the head is
// pushed to it at the instant it freed. Work is proportional to what actually
// ended, not to the size of the group.
//
// What is due is read once and retired in order. A retirement changes no other
// due core's oldest end, so the order stands unless the retirement itself made
// something due inside the interval -- the task pipelined behind the one that
// ended, or one placed on a core it freed -- and only then is it read again.
// Placing lowers `next_event`, which is how such an end shows itself.
void retire_due(SimQueue &q, uint64_t upto) {
    if (q.next_event > upto) {
        return;
    }
    uint32_t due[SIM_QUEUE_MAX_PACKAGES * 3];
    uint32_t n = collect_due(q, upto, due);
    uint32_t k = 0;
    while (k < n) {
        const uint32_t core = due[k++];
        q.next_event = UINT64_MAX;
        retire_core(q, core, q.evt_at[core], upto);
        if (q.next_event <= upto || q.evt_at[core] <= upto) {
            n = collect_due(q, upto, due);
            k = 0;
        }
    }
    refresh_next_event(q);
}

// Total entries waiting in a queue's rings. Read across queues for the imbalance
// sample, so the loads are relaxed: a sample that catches a ring mid-push is off
// by one, which does not matter for a distribution taken hundreds of times a round.
uint32_t queued_depth(const SimQueue &q) {
    uint32_t d = 0;
    for (uint32_t r = 0; r < SIM_RING_COUNT; ++r) {
        d += __atomic_load_n(&q.push[r], __ATOMIC_RELAXED) - __atomic_load_n(&q.pop[r], __ATOMIC_RELAXED);
    }
    return d;
}

// How unevenly work is spread over the groups, and how much of that unevenness is
// costing this group core time. Sampled on a fraction of polls so the scan over
// every queue stays far inside the latency the poll models.
[[maybe_unused]] void sample_imbalance(SimQueue &q) {
    if (++q.imb_tick < SIM_IMBALANCE_SAMPLE_POLLS) {
        return;
    }
    q.imb_tick = 0;
    const uint32_t self = static_cast<uint32_t>(&q - &g_queues[0]);
    uint32_t deepest = 0;
    uint32_t shallowest = UINT32_MAX;
    uint32_t own = 0;
    uint32_t deepest_q = self;
    bool peer_has_work_i_could_run = false;
    const bool idle_aic = mask_and(q.free_m, q.aic_m).any();
    const bool idle_aiv = mask_and(q.free_m, q.aiv_m).any();
    for (uint32_t i = 0; i < SIM_MAX_QUEUES; ++i) {
        const SimQueue &p = g_queues[i];
        if (p.package_count == 0) {
            continue;
        }
        const uint32_t d = queued_depth(p);
        if (i == self) {
            own = d;
        } else if (idle_aic || idle_aiv) {
            const uint32_t qc = __atomic_load_n(&p.push[ring_of(SimTaskType::Cube)], __ATOMIC_RELAXED) -
                                __atomic_load_n(&p.pop[ring_of(SimTaskType::Cube)], __ATOMIC_RELAXED);
            const uint32_t qv = __atomic_load_n(&p.push[ring_of(SimTaskType::Vector)], __ATOMIC_RELAXED) -
                                __atomic_load_n(&p.pop[ring_of(SimTaskType::Vector)], __ATOMIC_RELAXED);
            if ((idle_aic && qc > 0) || (idle_aiv && qv > 0)) {
                peer_has_work_i_could_run = true;
            }
        }
        if (d > deepest) {
            deepest = d;
            deepest_q = i;
        }
        if (d < shallowest) {
            shallowest = d;
        }
    }
    if (shallowest == UINT32_MAX) {
        return;
    }
    ++q.imb_samples;
    const uint32_t spread = deepest - shallowest;
    q.imb_spread_sum += spread;
    if (spread > q.imb_spread_max) {
        q.imb_spread_max = spread;
    }
    q.imb_own_depth_sum += own;
    if (deepest_q == self) {
        ++q.imb_deepest;
    }
    if (peer_has_work_i_could_run) {
        ++q.imb_starved;
    }
}

// One manager access to a queue. The manager is held for the modelled latency in
// real time. The queue's model runs on its own clock: real time less everything
// this queue's model has overrun so far. So the modelled cores stand still while
// the model works past the latency it charges, and removing that excess from the
// reported window -- which the phase recorder does -- removes exactly time the
// modelled world did not use. On the real clock the cores would keep computing
// through it, and the same removal would delete time they spent.
struct Access {
    uint64_t entered;  // real: when the manager arrived
    uint64_t release;  // real: when the manager is let go
    uint64_t at;       // model: when the access completes on the queue's clock
};

Access begin_access(uint32_t queue_idx, uint64_t latency) {
    const uint64_t now = get_sys_cnt_aicpu();
    const uint64_t ofs = queue_idx < SIM_MAX_QUEUES ? g_ledger[queue_idx].overrun_ticks : 0;
    return Access{now, now + latency, now - ofs + latency};
}

// Close an access: charge the simulator's own work against the latency the call
// models, add any excess to the queue's clock offset, and otherwise hold the
// manager until the modelled latency is up. Only the excess inflates a
// measurement; work that fits is hidden by the wait. One clock read serves all of
// it.
uint64_t close_access(uint32_t queue_idx, const Access &a, uint64_t &overrun_total, uint64_t &overrun_calls) {
    const uint64_t now = get_sys_cnt_aicpu();
    if (now > a.release) {
        overrun_total += now - a.release;
        ++overrun_calls;
        g_ledger[queue_idx].overrun_ticks += now - a.release;
    } else {
        wait_until(a.release);
    }
    return now - a.entered;
}

void finish_poll(SimQueue &q, uint32_t queue_idx, const Access &a) {
    const uint64_t work = close_access(queue_idx, a, q.poll_overrun_total, q.poll_overrun_calls);
    q.poll_work_total += work;
    if (work > q.poll_work_max) {
        q.poll_work_max = work;
    }
    ++q.poll_work_calls;
}

// A submit models a 5 ns posted write, but the controller work it triggers --
// settling held entries and placing whatever that frees -- is a loop here and a
// single clock in hardware.
void finish_push(SimQueue &q, uint32_t queue_idx, const Access &a) {
    q.push_work_total += close_access(queue_idx, a, q.push_overrun_total, q.push_overrun_calls);
    ++q.push_calls;
}

}  // namespace

void configure(uint64_t reg_base, uint32_t num_cores) {
    g_reg_base = reg_base;
    g_num_cores = num_cores;
}

void set_latencies_ns(uint64_t push_ns, uint64_t read_ns, uint64_t ack_ns, uint64_t notice_ns) {
    g_push_ticks = ns_to_ticks(push_ns);
    (void)read_ns;  // no core register is read under the GroupQueue
    // `ack` is calibrated for the AICPU-to-core path and carries 628 ns of
    // propagation ahead of the core's own startup. A GroupQueue dispatch does not
    // travel that path -- it reaches an adjacent core over its own link -- so only
    // the startup remains, and it belongs to the core rather than to any message.
    g_pickup_ticks = ack_ns > SIM_LEGACY_PROPAGATION_NS ? ns_to_ticks(ack_ns - SIM_LEGACY_PROPAGATION_NS) : 0;
    // `notice` is what raising a FIN costs the core itself. The signal then travels
    // to the controller over the same link as everything else between them, so a
    // finish becomes legible one link after it is raised.
    g_fin_ticks = ns_to_ticks(notice_ns);
    g_link_ticks = ns_to_ticks(SIM_GQ_CORE_LINK_NS);
    g_core_poll_ticks = ns_to_ticks(SIM_CORE_POLL_NS);
    g_arrive_ticks = ns_to_ticks(SIM_QUEUE_ARRIVAL_NS);
}

void set_queue_latencies_ns(uint64_t report_ns, uint64_t poll_ns) {
    g_report_ticks = ns_to_ticks(report_ns);
    g_queue_poll_ticks = ns_to_ticks(poll_ns);
    g_mix_push_ticks = ns_to_ticks(SIM_MIX_PUSH_NS);
    g_mix_arrival_ticks = ns_to_ticks(SIM_GQ_CORE_LINK_NS);
    g_cancel_ticks = ns_to_ticks(SIM_GQ_CORE_LINK_NS);
}

void set_compute_ns_table(
    const uint64_t *ns_by_func, const uint64_t *sigma_ns_by_func, uint32_t n, uint64_t default_ns
) {
    g_default_compute_ticks = ns_to_ticks(default_ns);
    g_func_compute_ticks.assign(n, 0);
    g_func_compute_sigma_ticks.assign(n, 0);
    for (uint32_t i = 0; i < n; ++i) {
        g_func_compute_ticks[i] = ns_to_ticks(ns_by_func[i]);
        if (sigma_ns_by_func != nullptr) {
            g_func_compute_sigma_ticks[i] = ns_to_ticks(sigma_ns_by_func[i]);
        }
    }
}

void init() {
    for (uint32_t i = 0; i < SIM_MAX_QUEUES; ++i) {
        g_queues[i] = SimQueue{};
        // Distinct non-zero seed per queue; xorshift64 must never start at 0.
        g_queues[i].rng = 0x9E3779B97F4A7C15ULL * (i + 1) | 1ULL;
    }
    // Seed every core's COND so the bring-up handshake finds an initialised,
    // idle fabric. Nothing reads these afterwards.
    for (uint32_t i = 0; i < g_num_cores; ++i) {
        *reinterpret_cast<uint32_t *>(
            g_reg_base + static_cast<uint64_t>(i) * ASIMGQ_REG_BLOCK_SIZE + reg_offset(RegId::COND)
        ) = static_cast<uint32_t>(AICORE_IDLE_VALUE);
    }
}

void set_package(uint32_t package_idx, uint32_t queue_idx, uint32_t aic_core, uint32_t aiv0_core, uint32_t aiv1_core) {
    (void)package_idx;
    (void)aic_core;
    (void)aiv0_core;
    (void)aiv1_core;
    if (queue_idx >= SIM_MAX_QUEUES) {
        return;
    }
    SimQueue &q = g_queues[queue_idx];
    if (q.package_count >= SIM_QUEUE_MAX_PACKAGES) {
        return;
    }
    // The queue owns its cores outright, so they are named by their slot in it
    // rather than by a die-wide index nothing else here uses.
    const uint32_t pi = q.package_count++;
    SimPackage &p = q.packages[pi];
    p.aic = q.core_count++;
    p.aiv0 = q.core_count++;
    p.aiv1 = q.core_count++;
    q.aic_m.set(p.aic);
    q.aiv_m.set(p.aiv0);
    q.aiv_m.set(p.aiv1);
    for (uint32_t core : {p.aic, p.aiv0, p.aiv1}) {
        refresh_core(q, core);
    }
    q.ring_cap[ring_of(SimTaskType::Cube)] = q.package_count * SIM_RING_CAP_PER_CUBE_CORE;
    q.ring_cap[ring_of(SimTaskType::Vector)] = q.package_count * 2 * SIM_RING_CAP_PER_VECTOR_CORE;
    q.ring_cap[ring_of(SimTaskType::Mix)] = q.package_count * SIM_RING_CAP_PER_PACKAGE;
}

bool submit(
    uint32_t queue_idx, const uint64_t *gq_index, const int32_t *func_id, uint32_t parts, int32_t task_id,
    SimTaskType type, bool gated
) {
    const Access a = begin_access(queue_idx, g_push_ticks);
    const uint64_t deadline = a.release;
    if (queue_idx >= SIM_MAX_QUEUES || type == SimTaskType::Empty || parts == 0) {
        wait_until(deadline);
        return false;
    }
    SimQueue &q = g_queues[queue_idx];
    SimQueueEntryStore e = {};
    e.parts = parts > SIM_MIX_MAX_ENTRIES ? SIM_MIX_MAX_ENTRIES : parts;
    for (uint32_t i = 0; i < e.parts; ++i) {
        e.index[i] = gq_index[i];
        e.func_id[i] = func_id[i];
    }
    e.task_id = task_id;
    e.type = type;
    e.gated = gated;
    // The push is posted, so the manager is released after `deadline`; the entry
    // itself is not legible to the controller until the store completes, which is
    // the earliest a core can be given it.
    if (!enqueue_entry(q, e, a.at + g_arrive_ticks)) {
        wait_until(deadline);
        return false;  // ordinary back-pressure: the manager retries later
    }
    // The controller pushes as soon as it has somewhere to push to, so an entry
    // arriving while a core of its type is free starts there at once. Waiting for
    // the manager's next poll would invent idle time the hardware does not have,
    // and that gap falls exactly where cores are idle and work is scarce.
    dispatch_pending(q, a.at);
    finish_push(q, queue_idx, a);
    return true;
}

bool submit_grouped(
    uint32_t queue_idx, const uint64_t *gq_index, const int32_t *func_id, uint32_t parts, int32_t task_id,
    SimTaskType type, const uint64_t *dep_index, uint32_t dep_n
) {
    if (dep_n == 0) return submit(queue_idx, gq_index, func_id, parts, task_id, type, false);
    const Access a = begin_access(queue_idx, g_push_ticks);
    const uint64_t deadline = a.release;
    if (queue_idx >= SIM_MAX_QUEUES || type == SimTaskType::Empty || parts == 0) {
        wait_until(deadline);
        return false;
    }
    SimQueue &q = g_queues[queue_idx];
    if (dep_n > SIM_HELD_MAX_DEPS || q.held_n >= SIM_HELD_CAP) {
        // Not back-pressure: the manager sizes its intake by `held_room`, so
        // reaching here means that accounting is wrong. Reported, never absorbed --
        // a silently dropped task waits forever and takes the run down with it.
        ++q.held_refused;
        wait_until(deadline);
        return false;
    }
    const uint32_t s = static_cast<uint32_t>(__builtin_ctz(~q.held_used));
    SimHeld &h = q.held[s];
    h.e = SimQueueEntryStore{};
    h.e.parts = parts > SIM_MIX_MAX_ENTRIES ? SIM_MIX_MAX_ENTRIES : parts;
    for (uint32_t i = 0; i < h.e.parts; ++i) {
        h.e.index[i] = gq_index[i];
        h.e.func_id[i] = func_id[i];
    }
    h.e.task_id = task_id;
    h.e.type = type;
    h.e.gated = false;
    // Legible to the controller once the posted store completes, the same arrival
    // every submitted entry pays; nothing can be placed before that.
    h.e.enqueued_at = a.at + g_arrive_ticks;
    h.seq = q.held_seq++;
    // The manager named these from its own view, which lags the controller's: any
    // of them may have retired while this call was in flight, and those are not
    // waited for. A producer named twice is waited for once.
    h.remaining = 0;
    for (uint32_t i = 0; i < dep_n; ++i) {
        if (position_done(q, dep_index[i])) continue;
        uint32_t &w = q.held_waiters[dep_index[i] & (SIM_HELD_WAIT_SLOTS - 1)];
        if ((w & (1u << s)) == 0) {
            w |= 1u << s;
            ++h.remaining;
        }
    }
    q.held_used |= 1u << s;
    if (h.remaining == 0) {
        q.held_ready |= 1u << s;
    }
    ++q.held_n;
    if (q.held_n > q.held_high) q.held_high = q.held_n;
    ++q.held_admitted;
    promote_held(q, h.e.enqueued_at);
    dispatch_pending(q, h.e.enqueued_at);
    finish_push(q, queue_idx, a);
    return true;
}

void func_hist(uint64_t *out, uint32_t n) {
    uint64_t hist[16] = {};
    uint64_t other = 0;
    int32_t max_id = -1;
    for (const SimLedger &l : g_ledger) {
        for (uint32_t i = 0; i < 16; ++i)
            hist[i] += l.func_hist[i];
        other += l.func_other;
        if (l.func_max > max_id) max_id = l.func_max;
    }
    for (uint32_t i = 0; i < n && i < 16; ++i)
        out[i] = hist[i];
    if (n > 6) out[6] = other;
    if (n > 7) out[7] = static_cast<uint64_t>(max_id + 1);
}

void calib_coverage(uint64_t *hits, uint64_t *misses) {
    *hits = 0;
    *misses = 0;
    for (const SimLedger &l : g_ledger) {
        *hits += l.calib_hits;
        *misses += l.calib_misses;
    }
}

void queue_group_stats(uint32_t queue_idx, uint64_t *admitted, uint64_t *promoted, uint64_t *refused, uint32_t *high) {
    const SimQueue &q = g_queues[queue_idx];
    *admitted = q.held_admitted;
    *promoted = q.held_promoted;
    *refused = q.held_refused;
    *high = q.held_high;
}

SimQueueStatus read_queue_status(uint32_t queue_idx) {
    const Access a = begin_access(queue_idx, g_queue_poll_ticks);
    if (queue_idx >= SIM_MAX_QUEUES) {
        wait_until(a.release);
        return SimQueueStatus{0, 0, 0};
    }
    SimQueue &q = g_queues[queue_idx];
    // What the manager can see lags what has happened by the report path: a core
    // finishing, its controller telling the queue, and the queue writing the
    // register. Reading as of `report` before the access completes is that
    // propagation — the register shows the queue as it was, not as it is.
    const uint64_t legible = a.at > g_report_ticks ? a.at - g_report_ticks : 0;
    // The interval is replayed in time order: each end that came due is retired at
    // the instant it freed its core, and work waiting for that core starts then.
#if ASIMGQ_SELF_PROFILE
    const uint64_t t_retire0 = get_sys_cnt_aicpu();
#endif
    retire_due(q, legible);
#if ASIMGQ_SELF_PROFILE
    q.poll_retire_ticks += get_sys_cnt_aicpu() - t_retire0;
#endif
    // Entries are placed as they arrive and as each retirement frees a core, so the
    // rings are normally empty by now and this costs four compares.
    if (rings_hold_work(q)) {
        dispatch_pending(q, legible);
    }
#if ASIMGQ_SELF_PROFILE
    const uint64_t t_status0 = get_sys_cnt_aicpu();
    sample_imbalance(q);
#endif
    SimQueueStatus st{q.watermark, 0, SIM_HELD_CAP > q.held_n ? SIM_HELD_CAP - q.held_n : 0};
    // The published window spans at most two words of the rotating buffer, so it
    // is two reads and a splice rather than a walk.
    {
        const uint32_t wi = q.ahead_base >> 6;
        const uint32_t sh = q.ahead_base & 63;
        uint64_t win = q.ahead_bits[wi] >> sh;
        if (sh != 0) {
            win |= q.ahead_bits[(wi + 1) & (SIM_AHEAD_WORDS - 1)] << (64 - sh);
        }
        st.ahead_bits = static_cast<uint32_t>(win & ((1ULL << SIM_AHEAD_PUBLISHED) - 1));
    }
#if ASIMGQ_SELF_PROFILE
    q.poll_status_ticks += get_sys_cnt_aicpu() - t_status0;
#endif
    // The status read is one access: the watermark and the published look-ahead
    // share a register. The poll latency above is the whole of it.
    finish_poll(q, queue_idx, a);
    return st;
}

void release_cohort(uint32_t queue_idx, uint64_t at_real) {
    // A posted write into the queue, the same class of access as submitting an
    // entry, so it is charged the same.
    const Access a = begin_access(queue_idx, g_push_ticks);
    if (queue_idx >= SIM_MAX_QUEUES) {
        wait_until(a.release);
        return;
    }
    SimQueue &q = g_queues[queue_idx];
    // The caller's instant is on the real clock; the cohort starts on the queue's.
    const uint64_t ofs = g_ledger[queue_idx].overrun_ticks;
    const uint64_t at = at_real > ofs ? at_real - ofs : 0;
    for (uint32_t i = 0; i < q.core_count; ++i) {
        SimCore &c = q.cores[i];
        if (c.gated_count == 0) {
            continue;
        }
        for (uint32_t k = 0; k < c.outstanding; ++k) {
            if (c.finish[k] != SIM_GATED) {
                continue;
            }
            // No earlier than the cohort begins, and no earlier than this core
            // finishes what it was already running.
            const uint64_t start = at > c.gated_floor[k] ? at : c.gated_floor[k];
            // Same rule as an ordinary placement: the first member reaches its core
            // over the link and then starts up; one sitting behind running work is
            // already there and its startup has overlapped.
            const uint64_t ack = (k > 0) ? 0 : g_link_ticks + g_core_poll_ticks + g_pickup_ticks;
            c.finish[k] = start + ack + c.gated_dur[k];
            --q.cohort_staged;
        }
        c.gated_count = 0;
        // The core is free when the last of its committed work ends.
        c.ready_at = c.finish[c.outstanding - 1];
        refresh_core(q, i);
    }
    refresh_next_event(q);
    finish_push(q, queue_idx, a);
}

uint32_t read_cohort_staged(uint32_t queue_idx) {
    const Access a = begin_access(queue_idx, g_queue_poll_ticks);
    if (queue_idx >= SIM_MAX_QUEUES) {
        wait_until(a.release);
        return 0;
    }
    SimQueue &q = g_queues[queue_idx];
    // As of the same propagation delay as the status read: a member the controller
    // placed from its ring inside the interval is counted once it is replayed.
    retire_due(q, a.at > g_report_ticks ? a.at - g_report_ticks : 0);
    const uint32_t n = q.cohort_staged;
    finish_poll(q, queue_idx, a);
    return n;
}

uint64_t queue_cohort_piped(uint32_t queue_idx) {
    return queue_idx < SIM_MAX_QUEUES ? g_queues[queue_idx].cohort_piped : 0;
}

uint64_t now_ticks() { return get_sys_cnt_aicpu(); }

void queue_busy_ticks(uint32_t queue_idx, uint64_t *aic_busy, uint64_t *aiv_busy) {
    if (queue_idx >= SIM_MAX_QUEUES) {
        *aic_busy = *aiv_busy = 0;
        return;
    }
    *aic_busy = g_queues[queue_idx].busy_aic;
    *aiv_busy = g_queues[queue_idx].busy_aiv;
}

void queue_steal_stats(uint32_t queue_idx, uint64_t *tried, uint64_t *won, int64_t *gain_us) {
    if (queue_idx >= SIM_MAX_QUEUES) {
        *tried = *won = 0;
        *gain_us = 0;
        return;
    }
    const SimQueue &q = g_queues[queue_idx];
    *tried = q.steals_tried;
    *won = q.steals_won;
    *gain_us = q.steal_gain_ticks * 1000000LL / static_cast<int64_t>(get_sys_cnt_aicpu_frequency_hz());
}

void queue_imbalance(
    uint32_t queue_idx, uint64_t *samples, uint64_t *mean_spread, uint64_t *max_spread, uint64_t *mean_own_depth,
    uint64_t *deepest_pct, uint64_t *starved_pct
) {
    if (queue_idx >= SIM_MAX_QUEUES) {
        *samples = *mean_spread = *max_spread = *mean_own_depth = *deepest_pct = *starved_pct = 0;
        return;
    }
    const SimQueue &q = g_queues[queue_idx];
    const uint64_t n = q.imb_samples;
    *samples = n;
    *mean_spread = n ? q.imb_spread_sum / n : 0;
    *max_spread = q.imb_spread_max;
    *mean_own_depth = n ? q.imb_own_depth_sum / n : 0;
    *deepest_pct = n ? q.imb_deepest * 100 / n : 0;
    *starved_pct = n ? q.imb_starved * 100 / n : 0;
}

void queue_residency(uint32_t queue_idx, uint64_t *mean_ns, uint64_t *max_ns, uint64_t *n) {
    if (queue_idx >= SIM_MAX_QUEUES) {
        *mean_ns = *max_ns = *n = 0;
        return;
    }
    const SimQueue &q = g_queues[queue_idx];
    const uint64_t hz = get_sys_cnt_aicpu_frequency_hz();
    *n = q.residency_n;
    *mean_ns = q.residency_n ? q.ring_wait_total * 1000000000ULL / hz / q.residency_n : 0;
    *max_ns = q.residency_n ? q.bound_wait_total * 1000000000ULL / hz / q.residency_n : 0;
}

uint64_t queue_finish_ts(uint32_t queue_idx, uint64_t index) {
    if (queue_idx >= SIM_MAX_QUEUES) {
        return 0;
    }
    return g_queues[queue_idx].finish_at[index & (SIM_FINISH_TS_SLOTS - 1)];
}

void queue_retire_lag(uint32_t queue_idx, uint64_t *mean_ns, uint64_t *max_ns, uint64_t *n) {
    if (queue_idx >= SIM_MAX_QUEUES) {
        *mean_ns = *max_ns = *n = 0;
        return;
    }
    const SimQueue &q = g_queues[queue_idx];
    const uint64_t hz = get_sys_cnt_aicpu_frequency_hz();
    *n = q.retire_n;
    *mean_ns = q.retire_n ? q.retire_lag_total * 1000000000ULL / hz / q.retire_n : 0;
    *max_ns = q.retire_lag_max * 1000000000ULL / hz;
}

void queue_ahead_stats(uint32_t queue_idx, uint32_t *high_water, uint64_t *overflow) {
    if (queue_idx >= SIM_MAX_QUEUES) {
        *high_water = 0;
        *overflow = 0;
        return;
    }
    *high_water = g_queues[queue_idx].ahead_high_water;
    *overflow = g_queues[queue_idx].ahead_overflow;
}

uint64_t queue_self_overrun_ticks(uint32_t queue_idx) {
    if (queue_idx >= SIM_MAX_QUEUES) return 0;
    return g_ledger[queue_idx].overrun_ticks;
}

void queue_push_overrun(uint32_t queue_idx, uint64_t *total_us, uint64_t *calls, uint64_t *mean_work_ns) {
    const SimQueue &q = g_queues[queue_idx];
    const uint64_t hz = get_sys_cnt_aicpu_frequency_hz();
    *total_us = q.push_overrun_total * 1000000ULL / hz;
    *calls = q.push_overrun_calls;
    *mean_work_ns = q.push_calls ? q.push_work_total * 1000000000ULL / hz / q.push_calls : 0;
}

void queue_poll_overrun(uint32_t queue_idx, uint64_t *total_us, uint64_t *calls) {
    if (queue_idx >= SIM_MAX_QUEUES) {
        *total_us = *calls = 0;
        return;
    }
    const SimQueue &q = g_queues[queue_idx];
    *total_us = q.poll_overrun_total * 1000000ULL / get_sys_cnt_aicpu_frequency_hz();
    *calls = q.poll_overrun_calls;
}

void queue_poll_profile(uint32_t queue_idx, uint64_t *retire_ns, uint64_t *status_ns, uint64_t *calls) {
    if (queue_idx >= SIM_MAX_QUEUES) {
        *retire_ns = *status_ns = *calls = 0;
        return;
    }
    const SimQueue &q = g_queues[queue_idx];
    const uint64_t hz = get_sys_cnt_aicpu_frequency_hz();
    const uint64_t n = q.poll_work_calls ? q.poll_work_calls : 1;
    *retire_ns = q.poll_retire_ticks * 1000000000ULL / hz / n;
    *status_ns = q.poll_status_ticks * 1000000000ULL / hz / n;
    *calls = q.poll_work_calls;
}

void queue_poll_work(uint32_t queue_idx, uint64_t *total_ticks, uint64_t *max_ticks, uint64_t *calls) {
    if (queue_idx >= SIM_MAX_QUEUES) {
        *total_ticks = *max_ticks = *calls = 0;
        return;
    }
    *total_ticks = g_queues[queue_idx].poll_work_total;
    *max_ticks = g_queues[queue_idx].poll_work_max;
    *calls = g_queues[queue_idx].poll_work_calls;
}

}  // namespace asimgq

// The phase recorder asks whichever simulated device is compiled in; this build
// carries the GroupQueue model, so the answer is the sum over the queue this
// thread manages.
uint64_t simulated_device_self_overrun_ticks() {
    const int idx = platform_aicpu_affinity_thread_idx();
    return idx < 0 ? 0 : asimgq::queue_self_overrun_ticks(static_cast<uint32_t>(idx));
}
