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
#pragma once

#include <cstdint>

// aSim: an AICPU-hosted simulated compute-core device.
//
// Each simulated core is a passive 2-deep pipelined state machine (an active
// task plus one pipelined "pushed" task) modeling the Ascend compute-core
// register protocol. The scheduler drives it through two active calls that
// replace the passive MMIO load/store to a real core's SPR window:
//
//   asim_push(core, task_id)   — dispatch: admit the task, schedule its ACK/FIN
//   asim_read_status(core)     — poll: advance the machine to `now`, return COND
//
// Timing is real wall-clock (Mode A): the injected MMIO latency is a cntvct
// busy-spin inside each call, and a task's compute duration elapses because its
// FIN deadline is `now + submit + compute` and real time passes while the
// scheduler does other work. No per-core thread runs; the machine is advanced
// lazily whenever the scheduler reads it. The returned word is bit-identical to
// the real COND encoding: [bit 31 state | bits 30..0 task_id], ACK=0 / FIN=1.

namespace asim {

// Configure the aSim device before bring-up. `reg_base` is the base address of
// the aSim-owned per-core register backing; core i lives at
// reg_base + i * ASIM_REG_BLOCK_SIZE. `num_cores` is how many are modeled.
// Per-core stride of the simulated register file. SIM_REG_BLOCK_SIZE is sized for
// the PMU window (highest offset 0x12A0), which the simulated device does not
// model: it touches DATA_MAIN_BASE (0xA0) on dispatch and COND (0x4C8) on poll,
// and nothing else. Striding by the PMU-sized block would put every core's COND on
// its own page, so seeding 72 cores would take 72 first-touch faults inside the
// first run's device_wall -- a cost real silicon pays at power-on and never again.
constexpr uint32_t ASIM_REG_BLOCK_SIZE = 0x500;

void configure(uint64_t reg_base, uint32_t num_cores);

// Set the injected MMIO / handshake latencies, in nanoseconds. Converted to
// counter ticks internally. push = DATA_MAIN_BASE write, read = COND read,
// ack = idle-core push->ack delay (paper: read ~77ns, push ~64ns).
// `notice_ns` is the FIN-write -> AICPU-observable delay (0 = instant). It
// delays only the observation: a pipelined task is still promoted when the
// active one finishes, as on real hardware.
void set_latencies_ns(uint64_t push_ns, uint64_t read_ns, uint64_t ack_ns, uint64_t notice_ns);

// Compute-duration model, keyed by func_id (the kernel a task runs). `ns_by_func`
// is a per-test table of length `n` (index = func_id, value = mean compute ns,
// 0 = unknown); it is provisioned host-side per test and carried to the device
// in Runtime. A func_id with no entry falls back to `default_ns`.
// `sigma_ns_by_func` (may be null) carries the per-func_id compute stddev; a
// non-zero entry makes each task of that func_id draw its own duration, so
// cores finish scattered rather than in lockstep batches.
void set_compute_ns_table(
    const uint64_t *ns_by_func, const uint64_t *sigma_ns_by_func, uint32_t n, uint64_t default_ns
);

// Reset every core to Fully Free and seed its published COND word to
// AICORE_IDLE_VALUE, so the scheduler bring-up handshake sees idle cores
// without any real AICore having run.
void init();

// Map a per-core register base address (CoreExecState.reg_addr) to a core
// index. Valid only for addresses within the aSim register backing.
uint32_t core_index_for_addr(uint64_t reg_addr);

// Dispatch: the scheduler wrote `task_id` to core `core_idx`'s DATA_MAIN_BASE.
// `func_id` is the kernel this task runs — selects its compute duration from the
// table. Admits the task into the 2-deep pipeline and schedules its deadlines.
void asim_push(uint32_t core_idx, int32_t task_id, int32_t func_id);

// Poll: advance core `core_idx`'s state machine to the current clock and return
// its COND word (the last ACK/FIN event reached by `now`). Charges the read
// latency inline. Also mirrors the word into the register backing so any
// residual passive reader (cold-path handshake) stays coherent.
uint32_t asim_read_status(uint32_t core_idx);

// Work that did not fit inside the latency a status read models, over cores
// [first_core, first_core + n_cores), and how many reads overran. Only this
// inflates a measurement, so it is what a delta against the GroupQueue arm must be
// corrected by -- measured here rather than assumed to be zero.
// Cores are handed to a thread round-robin, so the caller names them explicitly:
// a contiguous window would report some other thread's cores.
void asim_poll_overrun(
    const int32_t *core_ids, uint32_t n_cores, uint64_t *total_us, uint64_t *calls, uint64_t *poll_calls,
    uint64_t *poll_work_us
);

// Compute issued to a core range, in ticks -- the A/B work-equivalence check.
void asim_busy_ticks(const int32_t *core_ids, uint32_t n_cores, uint64_t *total, uint64_t *dispatches);

// Dispatches charged per func_id -- compared across models, says whether both ran the same work.
void asim_func_hist(uint64_t *out, uint32_t n);

// The simulator's own cost inside the status reads of cores
// [first_core, first_core + n_cores): the mean per call, the total, and how many
// calls. The modelled read latency is only honest while that work fits inside it,
// and a delta against the GroupQueue arm is only fair if both arms' overheads are
// measured rather than assumed.
void asim_poll_work(uint32_t first_core, uint32_t n_cores, uint64_t *mean_ns, uint64_t *total_us, uint64_t *calls);

// How long a pushed task sits before it starts running, over cores
// [first_core, first_core + n_cores), in nanoseconds: zero on an idle core beyond
// the latch, and the remainder of the active task's compute on a pipelined one.
// The same quantity the GroupQueue arm reports as ring residency.
void asim_residency(uint32_t first_core, uint32_t n_cores, uint64_t *mean_ns, uint64_t *max_ns, uint64_t *n);

// When the most recent task on `core_idx` ended, in the AICPU's own clock domain.
// A diagnostic read of state the core already holds, not a modelled access: it
// charges nothing, because no COND word carries it. A poll that retires both a
// running and a pipelined task sees only the later of the two ends here.
uint64_t asim_finish_ts(uint32_t core_idx);

}  // namespace asim
