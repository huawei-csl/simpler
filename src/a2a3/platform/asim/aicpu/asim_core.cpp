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
#include "aicpu/asim_core.h"

#include <vector>

#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/device_time.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "common/chip_swimlane_profiling.h"
#include "common/platform_config.h"

namespace asim {

namespace {

// One simulated compute core: a 2-deep pipeline (active + pushed) with its
// deadlines in the get_sys_cnt_aicpu() tick domain. Each core is one cache line
// so that a thread polling core i never false-shares with a thread polling core
// i+1 — the cores stay as independent as the real per-core CoreExecState (which
// is likewise alignas(64)). Every core is accessed by exactly one scheduler
// thread (HBG assigns cores to threads), so no atomics are needed.
struct alignas(64) SimCore {
    int32_t active_id = AICPU_TASK_INVALID;  // running task, or none
    uint64_t active_ack_at = 0;              // tick at which active is acknowledged
    uint64_t active_fin_at = 0;              // tick at which active finishes
    uint64_t last_fin_at = 0;                // when the most recent task on this core ended
    // How long a pushed task sits before it starts running: zero on an idle core
    // beyond the latch, and the rest of the active task's compute on a pipelined
    // one. The GroupQueue arm's ring residency is the same quantity.
    // What the simulator itself spends inside this core's status read, against
    // the latency that read models. The modelled latency is only honest while the
    // work fits inside it, and the comparison against the GroupQueue arm is only
    // fair if both arms' overheads are known rather than assumed.
    uint64_t poll_work_total = 0;
    uint64_t poll_work_calls = 0;
    // Work that did not fit inside the latency the read models. Only this inflates
    // a measurement -- work that fits is hidden by the spin to the deadline -- so
    // it is the figure a delta between the arms must be corrected by. Expected to
    // stay zero here, since advancing one core is O(1) against a 92-195 ns read,
    // but a correction applied to one arm on an assumption about the other would
    // just replace one bias with another.
    uint64_t poll_overrun_total = 0;
    uint64_t poll_overrun_calls = 0;
    // The same, for a push: its 5 ns models a posted write, which drawing a
    // compute sample and updating the pipeline can exceed.
    uint64_t push_overrun_total = 0;
    uint64_t push_overrun_calls = 0;

    uint64_t residency_total = 0;
    uint64_t residency_max = 0;
    uint64_t residency_n = 0;
    int32_t pushed_id = AICPU_TASK_INVALID;  // pipelined task, or none
    uint64_t pushed_dur = 0;                 // pushed task's compute duration (ticks)
    uint64_t busy = 0;                       // compute this core has been given, in ticks
    uint64_t dispatches = 0;                 // entries pushed to this core
    uint32_t cond_word = 0;                  // last published COND word
    uint64_t rng = 0;                        // per-core PRNG state (compute sampling)
};

std::vector<SimCore> g_cores;
uint64_t g_reg_base = 0;

// What modelling has cost, per scheduler thread, in counter ticks: every access's
// work beyond the latency it models. It accumulates monotonically across runs, for
// the same reason as the GroupQueue's ledger -- the phase recorder subtracts the
// growth inside a phase -- and it is the thread's clock offset. A thread's cores
// run on real time less this, so they stand still while the model works past
// the latency it charges, and subtracting that excess from the reported window
// removes exactly time the modelled world did not use.
uint64_t g_self_overrun_ticks[PLATFORM_MAX_AICPU_THREADS] = {};

// The calling thread's ledger slot. Every core is accessed by its owning thread
// alone, so its model clock is that thread's.
uint64_t *self_ledger() {
    const int t = platform_aicpu_affinity_thread_idx();
    return (t >= 0 && t < PLATFORM_MAX_AICPU_THREADS) ? &g_self_overrun_ticks[t] : nullptr;
}

// The swimlane record a real core writes per task: start and end from its own
// clock, and the identity it copied out of the payload. No kernel runs here, so
// the simulated core writes it, taking the next slot of the buffer the AICPU
// published for that core -- the same reservation a real core makes, in the same
// dispatch order, so buffer rotation and host collection are unchanged.
struct RecordCursor {
    uint64_t buf = 0;
    uint32_t seq = UINT32_MAX;  // never a published seq, so the first write loads the head
    uint32_t slot = 0;
};
enum class Recording : uint8_t { Unresolved, Off, On };
Recording g_recording = Recording::Unresolved;
const uint64_t *g_record_heads = nullptr;  // core index -> &ChipSwimlaneActiveHead
std::vector<RecordCursor> g_record_cursor;

void emit_record(
    uint32_t core_idx, uint64_t token, uint32_t reg_task_id, uint64_t receive, uint64_t start, uint64_t end
) {
    if (g_recording == Recording::Unresolved) {
        // The rotation table is filled by the collector's init, which runs after
        // this device's init() and before the first dispatch.
        const uint64_t table = is_chip_swimlane_enabled() ? get_platform_chip_swimlane_aicore_rotation_table() : 0;
        g_record_heads = reinterpret_cast<const uint64_t *>(table);
        g_recording = table != 0 ? Recording::On : Recording::Off;
    }
    if (g_recording != Recording::On || g_record_heads[core_idx] == 0) {
        return;
    }
    const auto *head = reinterpret_cast<const ChipSwimlaneActiveHead *>(g_record_heads[core_idx]);
    RecordCursor &cur = g_record_cursor[core_idx];
    const uint32_t seq = head->current_buf_seq;
    if (seq != cur.seq) {
        cur.seq = seq;
        cur.buf = head->current_buf_ptr;
        cur.slot = 0;
    }
    if (cur.buf == 0 || cur.slot >= PLATFORM_AICORE_BUFFER_SIZE) {
        return;
    }
    ChipSwimlaneAicoreTaskRecord &r = reinterpret_cast<ChipSwimlaneAicoreTaskBuffer *>(cur.buf)->records[cur.slot++];
    r.start_time = start;
    r.end_time = end;
    r.task_token_raw = token;
    r.reg_task_id = reg_task_id;
    r.receive_to_start_cycles = static_cast<uint32_t>(start - receive);
}

// Injected latencies and the M0 compute duration, in the sys-cnt tick domain.
uint64_t g_push_ticks = 0;
uint64_t g_read_ticks = 0;
uint64_t g_ack_ticks = 0;
uint64_t g_notice_ticks = 0;                       // FIN write -> AICPU-observable delay
std::vector<uint64_t> g_func_compute_ticks;        // index = func_id, value = mean compute ticks
std::vector<uint64_t> g_func_compute_sigma_ticks;  // index = func_id, value = compute stddev ticks
uint64_t g_default_compute_ticks = 0;              // fallback for a func_id with no table entry

uint64_t ns_to_ticks(uint64_t ns) {
    const uint64_t freq = get_sys_cnt_aicpu_frequency_hz();
    return static_cast<uint64_t>(static_cast<__uint128_t>(ns) * freq / 1'000'000'000ULL);
}

// Active-wait until the sys-cnt clock reaches `deadline`. Spin (never sleep/yield
// to the OS — this is the dispatch path); a bare architecture hint is the only
// pause. The caller is held here so the modeled MMIO latency is real elapsed
// wall-clock, exactly as the AICPU is held waiting on a real nGnRE access.
void wait_until(uint64_t deadline) {
    while (get_sys_cnt_aicpu() < deadline) {
#if defined(__aarch64__)
        __asm__ volatile("yield");
#endif
    }
}

// Every register the simulated device touches has to fall inside the stride, or a
// core's write would land in its neighbour's block.
static_assert(ASIM_REG_BLOCK_SIZE > reg_offset(RegId::COND), "aSim block must hold COND");
static_assert(ASIM_REG_BLOCK_SIZE > reg_offset(RegId::DATA_MAIN_BASE), "aSim block must hold DATA_MAIN_BASE");

uint32_t *cond_slot(uint32_t core_idx) {
    return reinterpret_cast<uint32_t *>(
        g_reg_base + static_cast<uint64_t>(core_idx) * ASIM_REG_BLOCK_SIZE + reg_offset(RegId::COND)
    );
}

// xorshift64: 3 shifts + 3 xors, cheap enough to hide inside the push latency.
// Each core owns its state and is touched by exactly one scheduler thread.
uint64_t next_rand(SimCore &c) {
    uint64_t x = c.rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    c.rng = x;
    return x;
}

// Per-func_id compute (the kernel a task runs). Falls back to the default when
// the func_id is out of range or has no calibrated entry (0). A calibrated
// sigma draws a per-task duration instead of returning the mean: real cores
// vary per task, and identical durations would finish cores in lockstep
// batches, collapsing the scheduler's dispatch/complete phase count well below
// the real one.
uint64_t g_func_hist[16] = {};
// One past the highest func_id the current table sets, so the next run clears
// exactly what this one wrote.
uint32_t g_func_table_high = 0;
uint64_t g_func_other = 0;
int32_t g_func_max = -1;

uint64_t sample_compute_ticks(SimCore &c, int32_t func_id) {
    if (func_id >= 0 && func_id < 16) {
        ++g_func_hist[func_id];
    } else {
        ++g_func_other;
        if (func_id > g_func_max) g_func_max = func_id;
    }
    uint64_t mean = g_default_compute_ticks;
    uint64_t sigma = 0;
    if (func_id >= 0 && static_cast<size_t>(func_id) < g_func_compute_ticks.size()) {
        const uint64_t t = g_func_compute_ticks[func_id];
        if (t != 0) {
            mean = t;
            if (static_cast<size_t>(func_id) < g_func_compute_sigma_ticks.size()) {
                sigma = g_func_compute_sigma_ticks[func_id];
            }
        }
    }
    if (sigma == 0) {
        return mean;
    }
    // Three uniform draws over [-2^20, 2^20) sum to a mean-0 deviate whose
    // stddev is exactly 2^20 (Irwin-Hall), so scaling by sigma >> 20 gives
    // stddev `sigma`, bounded to +-3 sigma. Integer-only: no float on the path.
    int64_t sum = 0;
    for (int k = 0; k < 3; ++k) {
        sum += static_cast<int64_t>(next_rand(c) & ((1ULL << 21) - 1)) - (1LL << 20);
    }
    int64_t dur = static_cast<int64_t>(mean) + ((static_cast<int64_t>(sigma) * sum) >> 20);
    if (dur < 1) {
        dur = 1;
    }
    return static_cast<uint64_t>(dur);
}

// Advance one core's state machine to `now`, updating its published COND word to
// the latest ACK/FIN event reached, subject to the FIN notice delay. Publishes only ACK/FIN events; when the
// active task finishes while a pushed task waits, the core auto-acks the pushed
// task and never shows the bare FIN of the active one (the forbidden Fully
// Busy -> Partially Busy transition). O(1): at most one promotion.
void advance(SimCore &c, uint64_t now) {
    while (c.active_id != AICPU_TASK_INVALID) {
        if (now >= c.active_fin_at) {
            if (c.pushed_id != AICPU_TASK_INVALID) {
                // Active finished with a pipelined task waiting: auto-ack it at
                // the instant the active one finished; it becomes active.
                const uint64_t promoted_ack = c.active_fin_at;
                c.last_fin_at = c.active_fin_at;
                c.active_id = c.pushed_id;
                c.active_ack_at = promoted_ack;
                c.active_fin_at = promoted_ack + c.pushed_dur;
                c.pushed_id = AICPU_TASK_INVALID;
                c.pushed_dur = 0;
                continue;
            }
            // The core writes FIN at active_fin_at, but the AICPU cannot
            // observe it until the notice delay elapses; until then COND still
            // reads ACK. Only the observation is delayed — a pipelined task is
            // still promoted at active_fin_at above, because the core does not
            // wait for the AICPU to look.
            if (now < c.active_fin_at + g_notice_ticks) {
                return;
            }
            c.last_fin_at = c.active_fin_at;
            c.cond_word = static_cast<uint32_t>(MAKE_FIN_VALUE(c.active_id));
            c.active_id = AICPU_TASK_INVALID;
            return;
        }
        if (now >= c.active_ack_at) {
            c.cond_word = static_cast<uint32_t>(MAKE_ACK_VALUE(c.active_id));
        }
        // Not yet acked: COND retains the previous word (idle or a prior FIN).
        return;
    }
}

}  // namespace

void configure(uint64_t reg_base, uint32_t num_cores) {
    g_reg_base = reg_base;
    // Sizing only: init() resets every core, so assigning fresh SimCores here
    // would write the same state twice on every run.
    if (g_cores.size() != num_cores) {
        g_cores.assign(num_cores, SimCore{});
    }
}

void set_latencies_ns(uint64_t push_ns, uint64_t read_ns, uint64_t ack_ns, uint64_t notice_ns) {
    g_push_ticks = ns_to_ticks(push_ns);
    g_read_ticks = ns_to_ticks(read_ns);
    g_ack_ticks = ns_to_ticks(ack_ns);
    g_notice_ticks = ns_to_ticks(notice_ns);
}

void set_compute_ns_table(
    const uint64_t *ns_by_func, const uint64_t *sigma_ns_by_func, uint32_t n, uint64_t default_ns
) {
    g_default_compute_ticks = ns_to_ticks(default_ns);
    // The table spans every func_id the runtime can name (1024) while a graph uses
    // a few dozen, so it is cleared over the span a previous run actually set
    // rather than rewritten whole.
    if (g_func_compute_ticks.size() != n) {
        g_func_compute_ticks.assign(n, 0);
        g_func_compute_sigma_ticks.assign(n, 0);
        g_func_table_high = 0;
    } else {
        for (uint32_t i = 0; i < g_func_table_high; ++i) {
            g_func_compute_ticks[i] = 0;
            g_func_compute_sigma_ticks[i] = 0;
        }
        g_func_table_high = 0;
    }
    // The table spans every func_id the runtime can name, of which a graph uses a
    // few dozen; an unused entry is 0 ns and converts to 0 ticks, so converting it
    // is a divide whose answer the assign above already wrote.
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t mean_ns = ns_by_func[i];
        if (mean_ns == 0) {
            continue;
        }
        g_func_compute_ticks[i] = ns_to_ticks(mean_ns);
        if (i + 1 > g_func_table_high) g_func_table_high = i + 1;
        if (sigma_ns_by_func != nullptr && sigma_ns_by_func[i] != 0) {
            g_func_compute_sigma_ticks[i] = ns_to_ticks(sigma_ns_by_func[i]);
        }
    }
}

void init() {
    g_recording = Recording::Unresolved;
    g_record_cursor.assign(g_cores.size(), RecordCursor{});
    for (uint32_t i = 0; i < g_cores.size(); ++i) {
        g_cores[i] = SimCore{};
        // Distinct non-zero seed per core; xorshift64 must never start at 0.
        g_cores[i].rng = 0x9E3779B97F4A7C15ULL * (i + 1) | 1ULL;
        g_cores[i].cond_word = static_cast<uint32_t>(AICORE_IDLE_VALUE);
        *cond_slot(i) = g_cores[i].cond_word;
    }
}

uint32_t core_index_for_addr(uint64_t reg_addr) {
    return static_cast<uint32_t>((reg_addr - g_reg_base) / ASIM_REG_BLOCK_SIZE);
}

// Both calls follow the same shape: capture t0 at entry, do the (O(1)) state work
// against the logical completion time `deadline` -- t0 + lat on the thread's model
// clock -- then hold the caller to t0 + lat in real time. Work that fits is
// absorbed by the hold; work that does not is added to the thread's ledger, which
// holds its model clock back by the same amount.

void asim_push(uint32_t core_idx, int32_t task_id, int32_t func_id, uint64_t task_token_raw) {
    const uint64_t release = get_sys_cnt_aicpu() + g_push_ticks;
    uint64_t *ledger = self_ledger();
    const uint64_t deadline = release - (ledger != nullptr ? *ledger : 0);
    SimCore &c = g_cores[core_idx];
    const uint64_t dur = sample_compute_ticks(c, func_id);
    c.busy += dur;
    ++c.dispatches;
    if (c.active_id == AICPU_TASK_INVALID) {
        // Fully Free -> Partially Busy: an idle core latches and acks quickly.
        c.active_id = task_id;
        c.active_ack_at = deadline + g_ack_ticks;
        c.active_fin_at = c.active_ack_at + dur;
        ++c.residency_n;
        emit_record(
            core_idx, task_token_raw, static_cast<uint32_t>(task_id), deadline, c.active_ack_at, c.active_fin_at
        );
    } else {
        // Partially Free -> Fully Busy: the pipelined task is auto-acked only
        // when the active task finishes. (The scheduler never pushes before the
        // active task is acked, so the pushed slot is empty here.)
        c.pushed_id = task_id;
        c.pushed_dur = dur;
        const uint64_t sit = c.active_fin_at > deadline ? c.active_fin_at - deadline : 0;
        c.residency_total += sit;
        if (sit > c.residency_max) {
            c.residency_max = sit;
        }
        ++c.residency_n;
        // advance() promotes it at the active task's finish with no latch of its
        // own, so that instant is both its receive and its start.
        emit_record(
            core_idx, task_token_raw, static_cast<uint32_t>(task_id), c.active_fin_at, c.active_fin_at,
            c.active_fin_at + dur
        );
    }
    const uint64_t done_at = get_sys_cnt_aicpu();
    if (done_at > release) {
        c.push_overrun_total += done_at - release;
        ++c.push_overrun_calls;
        if (ledger != nullptr) {
            *ledger += done_at - release;
        }
    }
    wait_until(release);
}

uint32_t asim_read_status(uint32_t core_idx) {
    const uint64_t entered_at = get_sys_cnt_aicpu();
    const uint64_t release = entered_at + g_read_ticks;
    uint64_t *ledger = self_ledger();
    const uint64_t deadline = release - (ledger != nullptr ? *ledger : 0);
    SimCore &c = g_cores[core_idx];
    advance(c, deadline);
    *cond_slot(core_idx) = c.cond_word;
    const uint64_t done_at = get_sys_cnt_aicpu();
    c.poll_work_total += done_at - entered_at;
    ++c.poll_work_calls;
    if (done_at > release) {
        c.poll_overrun_total += done_at - release;
        ++c.poll_overrun_calls;
        if (ledger != nullptr) {
            *ledger += done_at - release;
        }
    }
    wait_until(release);
    return c.cond_word;
}

void asim_push_overrun(const int32_t *core_ids, uint32_t n_cores, uint64_t *total_us, uint64_t *calls) {
    uint64_t total = 0, count = 0;
    for (uint32_t k = 0; k < n_cores; ++k) {
        const int32_t i = core_ids[k];
        if (i < 0 || static_cast<size_t>(i) >= g_cores.size()) continue;
        total += g_cores[i].push_overrun_total;
        count += g_cores[i].push_overrun_calls;
    }
    *calls = count;
    *total_us = total * 1000000ULL / get_sys_cnt_aicpu_frequency_hz();
}

// Compute issued to this core range, in ticks. Comparing it across an A/B is
// what says two runs executed the same graph, which a skip-golden run cannot.
void asim_func_hist(uint64_t *out, uint32_t n) {
    for (uint32_t i = 0; i < n && i < 16; ++i) out[i] = g_func_hist[i];
    if (n > 6) out[6] = g_func_other;
    if (n > 7) out[7] = static_cast<uint64_t>(g_func_max + 1);
}

// Cores are handed to threads round-robin, so a thread's core ids are not a
// contiguous range: they must be named individually or the threads' windows
// overlap and the same core is counted several times.
void asim_busy_ticks(const int32_t *core_ids, uint32_t n_cores, uint64_t *total, uint64_t *dispatches) {
    uint64_t sum = 0, n = 0;
    for (uint32_t k = 0; k < n_cores; ++k) {
        const int32_t i = core_ids[k];
        if (i < 0 || static_cast<size_t>(i) >= g_cores.size()) continue;
        sum += g_cores[i].busy;
        n += g_cores[i].dispatches;
    }
    *total = sum;
    *dispatches = n;
}

void asim_poll_overrun(
    const int32_t *core_ids, uint32_t n_cores, uint64_t *total_us, uint64_t *calls, uint64_t *poll_calls,
    uint64_t *poll_work_us
) {
    uint64_t total = 0, count = 0, pc = 0, pw = 0;
    for (uint32_t k = 0; k < n_cores; ++k) {
        const int32_t i = core_ids[k];
        if (i < 0 || static_cast<size_t>(i) >= g_cores.size()) continue;
        total += g_cores[i].poll_overrun_total;
        count += g_cores[i].poll_overrun_calls;
        pc += g_cores[i].poll_work_calls;
        pw += g_cores[i].poll_work_total;
    }
    const uint64_t hz = get_sys_cnt_aicpu_frequency_hz();
    *calls = count;
    *total_us = total * 1000000ULL / hz;
    *poll_calls = pc;
    *poll_work_us = pw * 1000000ULL / hz;
}

void asim_poll_work(uint32_t first_core, uint32_t n_cores, uint64_t *mean_ns, uint64_t *total_us, uint64_t *calls) {
    uint64_t total = 0, count = 0;
    for (uint32_t i = first_core; i < first_core + n_cores && i < g_cores.size(); ++i) {
        total += g_cores[i].poll_work_total;
        count += g_cores[i].poll_work_calls;
    }
    const uint64_t hz = get_sys_cnt_aicpu_frequency_hz();
    *calls = count;
    *mean_ns = count ? total * 1000000000ULL / hz / count : 0;
    *total_us = total * 1000000ULL / hz;
}

void asim_residency(uint32_t first_core, uint32_t n_cores, uint64_t *mean_ns, uint64_t *max_ns, uint64_t *n) {
    uint64_t total = 0, worst = 0, count = 0;
    for (uint32_t i = first_core; i < first_core + n_cores && i < g_cores.size(); ++i) {
        total += g_cores[i].residency_total;
        count += g_cores[i].residency_n;
        if (g_cores[i].residency_max > worst) {
            worst = g_cores[i].residency_max;
        }
    }
    const uint64_t hz = get_sys_cnt_aicpu_frequency_hz();
    *n = count;
    *mean_ns = count ? total * 1000000000ULL / hz / count : 0;
    *max_ns = worst * 1000000000ULL / hz;
}

uint64_t asim_finish_ts(uint32_t core_idx) {
    if (core_idx >= g_cores.size()) {
        return 0;
    }
    return g_cores[core_idx].last_fin_at;
}

}  // namespace asim

// The calling thread's ledger, which the phase recorder subtracts from the window
// it reports. Exact because this thread's cores run on the clock the ledger
// offsets: the excess it removes is time in which they did not move.
uint64_t simulated_device_self_overrun_ticks() {
    const uint64_t *ledger = asim::self_ledger();
    return ledger != nullptr ? *ledger : 0;
}
