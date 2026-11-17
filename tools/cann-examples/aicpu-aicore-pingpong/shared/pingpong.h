// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//
// Shared GM layout for the AICPU <-> AICore ping-pong. Compiled by three
// different compilers (ccec for AICore, the aarch64 cross gcc for AICPU, the
// host gcc), so every field is a fixed-width C type and every offset is
// static_asserted.
//
// THE ONE RULE: one writer per 64-byte cache line.
//
// That is load-bearing, not tidiness. `dcci(p, SINGLE_CACHE_LINE,
// CACHELINE_OUT)` writes back a whole 64 B line from a cache whose GM reads are
// not coherent with AICPU writes, so an AICPU-owned word sharing a line with an
// AICore-written field is restored to the AICore's stale copy on every publish.
// aicore-notification-perf measured exactly that as a hang in 3 of 6 runs
// (see its shared/handshake.h). Every block below names its sole writer.
//
// No std::atomic anywhere in GM: GM-side atomics from device glibc trap. The
// ping-pong protocol needs no read-modify-write anyway -- the parity gate means
// only one side is ever eligible to write -- so plain volatile words suffice.

#pragma once

#include <stdint.h>

#ifndef __CCE_AICORE__
#include <stddef.h>
#endif

// ---------------------------------------------------------------------------
// Topology. Mirrors src/a5/platform/onboard/host/host_regs.cpp:86-102, which
// builds the 108-entry register table that get_coreid() & AICORE_COREID_MASK
// indexes on BOTH halves of the mix pair (src/a5/platform/onboard/aicore/
// kernel.cpp:168 compiles that indexing into cube and vector alike).
//
//   die 0:  AIC 0..17     AIV 18..53
//   die 1:  AIC 54..71    AIV 72..107
// ---------------------------------------------------------------------------
#define PP_COREID_MASK 0x0FFFu
#define PP_AIC_PER_DIE 18u
#define PP_SLOTS_PER_DIE 54u  // 18 AIC + 36 AIV
#define PP_MAX_SLOTS 108u
#define PP_MAX_BLOCKDIM 36u  // == PLATFORM_MAX_PHYSICAL_CORES, so blockDim=36 covers every cluster

#define PP_SLOT_DIE(slot) ((slot) / PP_SLOTS_PER_DIE)
#define PP_SLOT_LOCAL(slot) ((slot) % PP_SLOTS_PER_DIE)
#define PP_SLOT_IS_AIC(slot) (PP_SLOT_LOCAL(slot) < PP_AIC_PER_DIE)
// Cluster k of a die owns AIC slot k and AIV slots 18 + 2k and 18 + 2k + 1.
// The three share one 3 KiB SSBUF and one L2, and are three halves of a single
// mix block, so a pair inside one cluster is the only AICore pair with a path
// that does not go through GM.
#define PP_SLOT_CLUSTER(slot)                                                 \
    (PP_SLOT_LOCAL(slot) < PP_AIC_PER_DIE ? PP_SLOT_LOCAL(slot)               \
                                          : (PP_SLOT_LOCAL(slot) - PP_AIC_PER_DIE) / 2u)

// Per-core probes, both timed against the 1 GHz system counter.
//
// The clock probe is a dependent xorshift chain. It has no closed form, so
// every iteration must actually execute, and its tick count is proportional to
// the core's clock PERIOD. Cycles per iteration is unknown and need not be
// known: the question is only whether two cores, or two launches, differ.
//
// The chase probe is a dependent walk over a single random cycle through
// PP_CHASE_LINES distinct 64-byte lines, so no step's address can be predicted
// from the last and every step costs one full load latency. The AICore reads
// it with the GM bypass, which makes each step a true GM access; the AICPU
// reads it through its cache, so only a walk larger than the 4 MB per-cluster
// L3 gives it a true GM number: 262144 lines is 16 MB, four times that.
#define PP_CHASE_LINES 262144u
#define PP_CHASE_U64 (PP_CHASE_LINES * 8u)

#define PP_MAX_CPU_SLOTS 32u
#define PP_MAX_CPUS 8u
#define PP_MAX_ROUNDS 4096u
#define PP_MAX_SLOTLIST 128u

// Modes. CENSUS runs the AICore side alone and cannot hang: every block writes
// one line and returns.
#define PP_MODE_CENSUS 0u
#define PP_MODE_MEASURE 1u
#define PP_MODE_PROBE 2u
// Both endpoints are AICore blocks and no AICPU is launched. Round r names its
// pair as (index of A << 16) | index of B into slot_list; A times the round and
// advances the round counter, B mirrors.
#define PP_MODE_CORE2CORE 3u

// Written by the probe so a zero-filled line cannot be mistaken for a hit.
#define PP_PROBE_MAGIC 0xD0D0CAFEBEEF0042ull

// token = (role << 56) | counter, with this bit set on the final token of a
// round. Without it the AICore could only stop on its run budget, so every
// measured run would burn the full timeout.
#define PP_TOKEN_END_BIT (1ull << 62)

// Written by a block to mark its attendance line as real rather than zero-fill.
#define PP_ATTEND_MAGIC 0x5AFEC0DEu

// Return codes published by a participating AICore block.
#define PP_RC_OK 0u
#define PP_RC_RUN_BUDGET 1u
#define PP_RC_NO_ARENA 2u

#pragma pack(push, 4)

// --- line block: host -> device control. HOST is the sole writer; the device
// never writes here, which is what lets the AICore derive the active round
// without any shared mutable word. ---
struct PpCtl {
    volatile uint32_t go;
    volatile uint32_t mode;
    volatile uint32_t n_rounds;
    volatile uint32_t n_blocks_expected;
    volatile uint32_t core_slot[2];  // the two AICore slots that participate
    volatile int32_t cpu_id[2];      // the two AICPU cpu_ids that participate
    volatile uint64_t iters;
    volatile uint64_t warmup;
    volatile uint64_t spin_budget;
    volatile uint64_t aicore_run_budget_ticks;
    // Deliberately SHORTER than aicore_run_budget_ticks: if a scheduled cpu_id
    // never receives a thread its rounds are never served, the round counter
    // never advances, and every AICPU thread would spin forever. The AICPU
    // gives up first, the AICore then sees no progress and ends on its own
    // budget, and the whole op still returns inside the 45 s STARS timeout
    // that would otherwise kill aicpu-sd.
    volatile uint64_t aicpu_deadline_ticks;
    volatile uint64_t throttle_ticks;
    // Full-matrix sweep. round r names its (AICPU, AICore slot) pair through
    // pair_list[r]; see that member for why the schedule is a table.
    volatile uint32_t n_cpus;
    volatile uint32_t n_slots;
    volatile uint32_t n_reps;
    volatile uint32_t clock_iters;
    volatile int32_t cpu_list[PP_MAX_CPUS];
    volatile uint32_t chase_steps;
    volatile uint32_t _pad0;
    uint64_t _pad[1];
};

// --- one line per AICore slot. Sole writer: the block that landed on it. ---
struct PpAttendance {
    volatile uint32_t attended;  // PP_ATTEND_MAGIC
    volatile uint32_t coreid_raw;
    volatile uint32_t block_idx;
    volatile uint32_t block_num;
    volatile uint32_t sub_block_id;
    volatile uint32_t is_aiv;
    volatile uint64_t t_entry;
    volatile uint64_t t_exit;
    // Sweep results, reported by the block that owns this slot. Keeping them
    // here rather than in a shared struct is what keeps one writer per line
    // when all 108 blocks are resident at once.
    volatile uint64_t rounds_served;
    volatile uint64_t trips_total;
    volatile uint64_t sweep_rc;
};

// --- one line per AICore slot. Sole writer: the block that landed on it. ---
struct PpCoreProbe {
    volatile uint64_t attended;
    volatile uint64_t clock_ticks;
    volatile uint64_t chase_ticks;
    volatile uint64_t sink;  // keeps both dependent chains from being deleted
    uint64_t _pad[4];
};

// --- one line per participating AICore role. Sole writer: that block. ---
struct PpCoreState {
    volatile uint32_t attended;
    volatile uint32_t coreid_raw;
    volatile uint32_t coreid_masked;
    volatile uint32_t rc;
    volatile uint64_t rounds_seen;
    volatile uint64_t trips_mirrored;
    volatile uint64_t sys_cnt_read_cost_ticks;
    uint64_t _pad[3];
};

// --- the round counter. Written only by whichever AICPU thread is currently
// active, and only one is ever active, so there is a single writer at any
// instant. AICore blocks only ever BYPASS-READ it -- they never cache it and
// never write it -- so no CACHELINE_OUT from the AICore side can clobber it.
// That is what makes a shared counter safe here when the general rule forbids
// two writers on a line. ---
struct PpRound {
    volatile uint64_t round;
    uint64_t _pad[7];
};

// --- one line per participating AICPU role. Sole writer: that thread.
// Separate from PpCoreState: the AICore owns that one, and a line with two
// writers is the exact failure this layout exists to prevent. ---
struct PpCpuState {
    volatile uint32_t attended;
    volatile int32_t cpu_id;
    volatile uint32_t rc;
    volatile uint32_t _pad0;
    volatile uint64_t elapsed_ticks;
    volatile uint64_t trips;
    volatile uint64_t timer_hz;
    volatile uint64_t clock_ticks;
    volatile uint64_t chase_ticks;
    volatile uint64_t sink;
};

// --- the ping-pong tokens. Each gets its own line and exactly one writer.
// token = (role << 56) | counter, so it is globally unique and monotonic:
// the mirroring side's rule is just "if it differs from what I last published,
// publish it", with no ABA and no initialisation-order race. ---
struct PpBall {
    volatile uint64_t token;
    uint64_t _pad[7];
};

#pragma pack(pop)

// Each block starts on its own 64-byte line.
#define PP_LINE 64u

struct PpArena {
    struct PpCtl ctl;
    uint8_t _pad_ctl[PP_LINE * 3 - sizeof(struct PpCtl)];

    struct PpRound sched;
    // Host-written, device-read-only: the AICore slots this sweep samples.
    // Sampling rather than enumerating all 108 keeps the schedule short; the
    // list is explicit so the choice is visible and reproducible.
    volatile uint32_t slot_list[PP_MAX_SLOTLIST];
    // Host-written, device-read-only: the round schedule, one entry per round,
    // packed as (cpu index << 16) | slot index. It is a shuffled table rather
    // than an arithmetic mapping so that a round's AICPU, its AICore slot and
    // its repeat number are all uncorrelated with when in the sweep it runs:
    // anything that drifts over a sweep then cannot present as a property of
    // one endpoint. Both the AICPU driver and the AICore partner read it to
    // agree on who is active, so it is the single source of the schedule.
    volatile uint32_t pair_list[PP_MAX_ROUNDS];
    // Host-written, device-read-only: one random cycle through the lines, so
    // chase[i * 8] holds the index of the next line to read. Lives in the
    // arena on purpose -- a probe in its own allocation would measure the
    // distance to that allocation rather than to the one being swept.
    volatile uint64_t chase[PP_CHASE_U64];
    struct PpAttendance attendance[PP_MAX_SLOTS];
    struct PpCoreProbe core_probe[PP_MAX_SLOTS];
    struct PpCoreState core_state[2];
    struct PpCpuState cpu_state[PP_MAX_CPUS];
    // One pair is active at a time, so a single ball line each suffices. The
    // AICore writes its ball with st_dev only, never dcci, so a parked block
    // can hold no cached copy to write back later.
    struct PpBall cpu_ball[2];
    struct PpBall core_ball[2];

    volatile uint64_t result_ticks[PP_MAX_ROUNDS];
    volatile uint32_t result_rc[PP_MAX_ROUNDS];
    volatile uint32_t result_slot[PP_MAX_ROUNDS];
    volatile int32_t result_cpu[PP_MAX_ROUNDS];
};

#ifndef __CCE_AICORE__
#ifdef __cplusplus
static_assert(sizeof(struct PpCtl) <= PP_LINE * 3, "PpCtl must fit its three lines");
static_assert(sizeof(struct PpAttendance) == PP_LINE, "PpAttendance must be exactly one line");
static_assert(sizeof(struct PpCoreState) == PP_LINE, "PpCoreState must be exactly one line");
static_assert(sizeof(struct PpBall) == PP_LINE, "PpBall must be exactly one line");
static_assert(sizeof(struct PpRound) == PP_LINE, "PpRound must be exactly one line");
static_assert(offsetof(struct PpArena, sched) == PP_LINE * 3, "sched must start on a line");
static_assert(offsetof(struct PpArena, pair_list) % PP_LINE == 0, "pair_list must start on a line");
static_assert(offsetof(struct PpArena, attendance) % PP_LINE == 0, "attendance must start on a line");
static_assert(sizeof(struct PpCpuState) == PP_LINE, "PpCpuState must be exactly one line");
static_assert(sizeof(struct PpCoreProbe) == PP_LINE, "PpCoreProbe must be exactly one line");
static_assert(offsetof(struct PpArena, chase) % PP_LINE == 0, "chase must start on a line");
static_assert(offsetof(struct PpArena, core_probe) % PP_LINE == 0, "core_probe must start on a line");
static_assert(offsetof(struct PpArena, core_state) % PP_LINE == 0, "core_state must start on a line");
static_assert(offsetof(struct PpArena, cpu_state) % PP_LINE == 0, "cpu_state must start on a line");
static_assert(offsetof(struct PpArena, cpu_ball) % PP_LINE == 0, "cpu_ball must start on a line");
static_assert(offsetof(struct PpArena, core_ball) % PP_LINE == 0, "core_ball must start on a line");
#endif
#endif

// DeviceArgs block handed to the AICPU half through the dispatcher bootstrap.
// Offsets 0..95 are the bootstrap's own slots; this tool's fields start at 96,
// exactly as in aicpu-pingpong.
#pragma pack(push, 4)
struct PpDeviceArgs {
    uint64_t reserved_pre[12];
    uint64_t arena_addr;
    uint64_t mode;
    uint64_t role_hint;
    uint64_t _reserved[5];
};
#pragma pack(pop)
