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

// aSim: an AICPU-hosted simulated GroupQueue device.
//
// The device is a set of GroupQueues, each the controller for the packages and
// cores it serves. There is no frontend controller in between: a queue holds
// every one of its cores' states itself, learns each transition from that core's
// own ACK and FIN, and pushes work out the moment a core can take it, rather than
// arbitrating between packages bidding for the head of a ring.
//
// Every queue is owned outright by one AICPU scheduler — the manager — and no
// other thread touches it. That is the design's point rather than an incidental
// property: a structure shared between schedulers would need them to arbitrate
// for it, and a lock between AICPUs costs more than the work it guards.
//
// A core is in one of three states, and the controller draws its target from the
// first two: free (nothing committed), running with its pipeline slot open, or
// full. A task goes to a free core of its type; only when every core of that type
// is running does it go into a pipeline slot, behind whichever frees soonest. A
// mix group is the exception to placing on any core: its halves share one
// package's local memory, so the group lands wholly inside a package.
//
// The scheduler drives its queue through two active calls, which replace the
// per-core register write and read a manager used to issue:
//
//   submit(queue, task)        — hand a ready task to the queue
//   read_queue_status(queue)   — its completed prefix, and what it holds
//
// Timing is real wall-clock (Mode A): each call busy-spins its modeled latency,
// and a task's compute elapses because its end is scheduled at
// `start + ack + compute` while real time passes as the scheduler works. Nothing
// is simulated in between: a task's end is computed once, when it is pushed to a
// core, so a poll compares the clock against the soonest end rather than replaying
// an interval.

// The device model's own measurement scaffolding -- per-poll timestamps, the
// imbalance sample, residency and retire-lag tracking -- runs on the same AICPU
// inside the window being measured, so it is charged to the arm under test. It is
// compiled out by default and switched on only when the question is the
// simulator's own cost rather than the design's.
#ifndef ASIMGQ_SELF_PROFILE
#define ASIMGQ_SELF_PROFILE 0
#endif

namespace asimgq {

// Configure the aSim device before bring-up. `reg_base` is the base address of
// the aSim-owned per-core register backing; core i lives at
// reg_base + i * ASIMGQ_REG_BLOCK_SIZE. `num_cores` is how many are modeled.
//
// The stride is sized for the registers the simulated device touches --
// DATA_MAIN_BASE (0xA0) on dispatch and COND (0x4C8) on poll -- not for the PMU
// window SIM_REG_BLOCK_SIZE covers, which it does not model. Striding by the
// larger block puts every core's COND on its own page, so seeding the cores costs
// one first-touch fault each inside the first run's device_wall: work real silicon
// does at power-on and never again.
constexpr uint32_t ASIMGQ_REG_BLOCK_SIZE = 0x500;
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

// Reset every queue and seed each core's published COND word to
// AICORE_IDLE_VALUE, so the bring-up handshake sees an initialised, idle fabric
// without any real AICore having run. Nothing reads a COND word afterwards.
void init();

// Number of GroupQueues the device models, and how far a queue's completed
// prefix may run ahead of what the manager has consumed.
constexpr uint32_t SIM_MAX_QUEUES = 8;
constexpr uint32_t SIM_QUEUE_WINDOW = 4096;

// Positions the per-position finish time is kept for. Only live positions are
// ever read back, and a manager's live window is a few dozen, so this is sized to
// stay cache-resident rather than to span the index space.
constexpr uint32_t SIM_FINISH_TS_SLOTS = 128;

// Set the GroupQueue latencies, in nanoseconds. `report_ns` is how long the
// controller's update of the manager's control register -- the watermark, and the
// positions finished ahead of it -- takes to become legible to the AICPU. It is a
// device-to-AICPU visibility hop, so it takes the calibration's measured `notice`
// rather than the barrier-forced store that anchors the opposite direction.
// `poll_ns` is what one read of that register costs the manager.
void set_queue_latencies_ns(uint64_t report_ns, uint64_t poll_ns);

// Register one AI package with the device: its three cores, and which
// GroupQueue serves it. A package is the three cores that share local memory, so
// the device needs the grouping the scheduler discovered at handshake to know
// where a mix group can land.
void set_package(uint32_t package_idx, uint32_t queue_idx, uint32_t aic_core, uint32_t aiv0_core, uint32_t aiv1_core);

// Shape of a queued task, which is what decides where the controller may push it.
// Cube and Vector each occupy one core, on any package. Mix is one entry
// describing a package-wide task: a cube half and up to two vector halves, pushed
// to the three cores of a single package.
enum class SimTaskType : uint8_t { Empty = 0, Vector = 1, Cube = 2, Mix = 3 };

// Cores a mix group can occupy: one cube plus up to two vectors, in that order.
constexpr uint32_t SIM_MIX_MAX_ENTRIES = 3;

// AICPU -> queue: how long a submitted entry takes to become legible to the
// controller. The push itself is posted, so the AICPU regains control after
// SIM_MIX_PUSH_NS-class 5 ns and does not wait for this; the entry simply cannot
// be placed until the store completes. Anchored on the barrier-forced MMIO store.
constexpr uint64_t SIM_QUEUE_ARRIVAL_NS = 310;

// Every message between a controller and one of its own cores, one way: a push
// out, a FIN back, each leg of a cancellation. The queue sits beside the packages
// it serves, so this is a dedicated on-die link rather than the MMIO path an AICPU
// takes to reach the same core. It replaces the transport only: a pushed task
// still waits SIM_CORE_POLL_NS for the core to read it.
constexpr uint64_t SIM_GQ_CORE_LINK_NS = 30;

// The calibration file's `ack` spans push to kernel start for the *AICPU-to-core*
// path, and covers three sequential costs, only the last of which the record
// measures directly:
//
//   310  the store lands on the core          SIM_CORE_ARRIVAL_NS
//   318  the core's spin loop reads it        SIM_CORE_POLL_NS
//   ack - 628  dcci + ack + kernel entry      receive_to_start, per case
//
// The split is anchored on the microbenchmark's three readings of the same MMIO
// submit: 5 ns posted, 310 ns barrier-forced complete, 633 ns observed by the
// target core. The 323 ns between the last two is the core getting round to its
// next read -- the only row in that table with real jitter (CV 37% against sd~0
// for the transport rows), which is a phase offset against a spin period rather
// than a transport cost. `receive_to_start` is stamped *after* the poll that
// noticed, so it does not overlap this.
//
// A GroupQueue dispatch replaces only the arrival term, since the controller sits
// beside the cores. The poll is the core's own loop and survives the move.
constexpr uint64_t SIM_CORE_ARRIVAL_NS = 310;
constexpr uint64_t SIM_CORE_POLL_NS = 318;
constexpr uint64_t SIM_LEGACY_PROPAGATION_NS = SIM_CORE_ARRIVAL_NS + SIM_CORE_POLL_NS;

// What placing a mix group costs the controller, in nanoseconds: it pushes the
// entries one at a time, and each then travels the core link.
constexpr uint64_t SIM_MIX_PUSH_NS = 5;

// A cancellation races the core's own promotion, so it carries the id of the task
// it means to cancel and a core honours it only while that exact task is still
// unacknowledged in its pipeline slot. The answer must be definitive: a lost
// answer has no safe default, since assuming success runs the task twice and
// assuming failure leaves it cancelled and never re-issued. Each leg is an
// ordinary controller-to-core message, so it costs SIM_GQ_CORE_LINK_NS.

// How many status reads pass between inter-queue imbalance samples. The sample
// scans every queue, so it is taken on a fraction of polls to keep that scan well
// inside the latency the poll models -- hundreds of samples a round is plenty for
// a distribution, and the cost stays off the measurement.
constexpr uint32_t SIM_IMBALANCE_SAMPLE_POLLS = 64;

// Hand a ready task to a queue. This replaces choosing a core: the manager gives
// the task to the queue's controller, which pushes it to a core itself. `type`
// selects the ring it joins — a queue keeps one ring per shape so a cube at the
// head cannot hold up queued vector work, and each ring's head is pushed
// independently. `gq_index` is the position the manager assigned, which is what
// the queue reports back on completion. `parts` is 1 for a cube or vector task and
// up to SIM_MIX_MAX_ENTRIES for a mix, whose `gq_index` / `func_id` arrays carry
// one element per subtask. Returns false when that ring is full, which is
// ordinary back-pressure.
bool submit(
    uint32_t queue_idx, const uint64_t *gq_index, const int32_t *func_id, uint32_t parts, int32_t task_id,
    SimTaskType type, bool gated = false
);

// Positions one held entry may wait on, and how many entries a controller can
// hold. A group's live set is bounded by what the manager will submit, which it
// throttles on this queue's occupancy, so these bound the hold array rather than
// the group.
// Task slots in one controller. A slot is occupied from submit until the manager
// learns the task finished, so this bounds what one GC can hold at all.
constexpr uint32_t SIM_HELD_CAP = 32;
// Dependency comparators per task slot. A task with more unmet producers than
// this cannot be expressed to the controller, so the manager holds it back.
constexpr uint32_t SIM_HELD_MAX_DEPS = 4;

// Hand the controller a task whose producers are members of the same group. The
// controller holds it until every named position has retired on its own cores and
// then places it, without the manager involved -- that locality is the whole point
// of a group. `dep_index` names positions this queue already holds; `dep_n` of
// zero is an ordinary ready submission and behaves exactly like submit().
//
// A named position that this queue does not hold is a violation of the grouping
// contract, not a case to tolerate: the entry would wait on a retirement no
// controller here will ever report.
bool submit_grouped(
    uint32_t queue_idx, const uint64_t *gq_index, const int32_t *func_id, uint32_t parts, int32_t task_id,
    SimTaskType type, const uint64_t *dep_index, uint32_t dep_n
);

// How the hold array is being used: entries admitted with unmet in-group
// producers, how many the controller later placed itself, and how many it refused
// for want of room. `promoted` against `admitted` is the direct measure of work
// the AICPU did not have to do.
void queue_group_stats(uint32_t queue_idx, uint64_t *admitted, uint64_t *promoted, uint64_t *refused, uint32_t *high);

// The core the controller placed `index` on. Placement is the controller's
// choice; this is how the manager learns it for the bookkeeping it keys by core.
uint32_t queue_entry_core(uint32_t queue_idx, uint64_t index);

// How many compute samples came from the calibration table versus the default.
void calib_coverage(uint64_t *hits, uint64_t *misses);

// Dispatches charged per func_id -- compared across models, says whether both ran the same work.
void func_hist(uint64_t *out, uint32_t n);

// The controller's own view, for diagnosing a manager that has stopped retiring.
void queue_debug(
    uint32_t queue_idx, uint64_t *watermark, uint64_t *high_water, uint32_t *packages, uint32_t *free_packages,
    uint64_t *ring_push, uint64_t *ring_pop
);

// What one read of a queue's status register yields. The completed prefix is
// what a manager needs on every pass; the other two are what it needs to assemble
// a cohort, and they ride the same read because they are the same register file.
// Carrying them here is what keeps cohort assembly from costing a manager any
// access it was not already making.
// Positions a queue can report as finished ahead of its prefix. One per core it
// serves: that is how many tasks can be in flight on its cores at once, so a
// deeper buffer would hold entries no core could have produced. The buffer lives
// in the AICPU package alongside the watermark; the controller writes both from
// across the die.
constexpr uint32_t SIM_LOOKAHEAD = 18;

// What one word of the look-ahead list costs the manager to read. The watermark
// and this buffer sit inside the AICPU package -- the controller updates them from
// across the die, but the manager reads them locally -- so a read is a short local
// access rather than the MMIO trip a core's register would take. Still one read
// per word: a list of N live entries costs N+1, the extra being the stale entry
// that ends the scan.
//
// The queue's own sorted insert is not charged: it happens when a core finishes,
// inside the queue, with the manager uninvolved, and a shift network over 18
// entries fits well inside the report path already modelled.
// One further look-ahead word, once a read is already in flight. The reads
// pipeline, so a word past the first costs its issue slot rather than another
// access latency.
constexpr uint64_t SIM_AHEAD_WORD_NS = 2;

struct SimQueueStatus {
    uint64_t watermark;       // contiguous completed prefix
    uint64_t gated_ready_at;  // when the last of this queue's gated members could start
    uint32_t free_packages;   // packages with all three cores free
    uint32_t staged_cohort;   // cohort entries placed but not started
    // Finished positions past the watermark, ascending: ahead[0] is the lowest
    // finished position above it, ahead[1] the next, and so on. An entry at or
    // below the watermark is stale and ends the list, so the scan terminates
    // itself and no count register is needed — and because the watermark is the
    // contiguous prefix, position `watermark` is by definition unfinished, so
    // every live entry is strictly greater and zero is a safe empty fill.
    //
    // Without these a task that finishes out of order cannot retire until
    // everything before it has, so it cannot release its consumers either: one
    // long task holds up every shorter one behind it.
    uint64_t ahead[SIM_LOOKAHEAD];
    // Entries each ring can still take, indexed by SimTaskType. A manager sizes
    // its pop from the shared ready queues by this, so it never claims work its
    // group has nowhere to put. Only the owning manager fills a ring, so a reading
    // can only become stale in the safe direction -- the device drains entries out
    // between the read and the submit, never in.
    uint32_t ring_room[4];
    // What this group can seat right now, by shape. `free_slots` are cores (for
    // mix, packages) holding nothing, which can start a task immediately;
    // `pipe_slots` are those running one task with the pipeline slot still open,
    // which can hold a task to start next. A manager publishes both so its peers
    // can see whether it still has somewhere better to put work than they do.
    uint32_t free_slots[4];
    uint32_t pipe_slots[4];
    // Tasks this group holds of each shape: resident on its cores plus queued.
    // Free slots say whether a manager can take work; this says how much it has
    // already taken, which is what decides whether it should take more.
    uint32_t load[4];
    // Entries the controller can still hold for tasks whose in-group producers
    // have not retired. A manager sizes its intake by this, so it never offers
    // work the controller has nowhere to put -- overflow is prevented on the
    // manager's side rather than reported back from the device.
    uint32_t held_room;
};

// Start every cohort member this queue holds, no earlier than `at` and no earlier
// than the core it sits on comes free. The manager calls this
// once the cohort is placed device-wide; until then the members occupy their
// cores without running, which is what co-residency costs. Charged as a posted
// write into the queue, like submitting an entry.
void release_cohort(uint32_t queue_idx, uint64_t at);

// The clock a manager reads to stamp a release.
uint64_t now_ticks();

// Depth of each of a GroupQueue's ready rings — how many submitted-but-unstarted
// tasks one can hold — and the most packages a queue can serve. It must be able to
// hold every entry a manager can have live at once: submit reports a full ring as
// back-pressure, and the manager has already claimed the index by then, so a ring
// shallower than the index window would drop a dispatch that could never retire.
constexpr uint32_t SIM_QUEUE_DEPTH = 64;

// How deep each ring may go, per core of its shape (per package, for mix). A ring
// bounded by what its own cores can run is what keeps a manager from taking more
// work than its group can absorb: the surplus stays in the shared ready queues,
// where a manager whose cores are idle can take it instead. Without that bound a
// manager's intake is limited only by its index window, so whichever drains
// fastest accumulates the most work and then becomes the makespan.
// A group may hold three tasks per slot of a shape: one running, one pipelined
// behind it, and one waiting in the ring. The first two live on the cores, so this
// is the ring's share -- one entry per core (per package, for mix).
constexpr uint32_t SIM_RING_CAP_PER_CUBE_CORE = 1;
constexpr uint32_t SIM_RING_CAP_PER_VECTOR_CORE = 1;
constexpr uint32_t SIM_RING_CAP_PER_PACKAGE = 1;
constexpr uint32_t SIM_QUEUE_MAX_PACKAGES = 32;

// A queued task as the device stores it. A mix entry describes a package-wide
// task, which the controller expands onto the three cores of one package, so
// `index` and `func_id` carry one element per subtask. A cube or vector entry uses
// element 0 only.
struct SimQueueEntryStore {
    uint64_t index[SIM_MIX_MAX_ENTRIES];
    int32_t func_id[SIM_MIX_MAX_ENTRIES];
    uint32_t parts;
    // When the entry became legible to a controller. A core cannot start work
    // that has not arrived, so this floors the start a grant assigns.
    uint64_t enqueued_at;
    int32_t task_id;
    SimTaskType type;
    // A cohort member is placed like any other entry but does not start: every
    // block of a require_sync_start task has to begin together, because the
    // kernel's blocks wait on each other and a missing peer never arrives.
    bool gated;
};

// Work that did not fit inside the latency a poll models, summed, and how many
// polls overran. This is the only part of the simulator's cost that inflates a
// measurement: work that fits is hidden by the spin to the deadline.
// The same, for the push path. A submit models a 5 ns posted write; settling the
// sub-ready array and placing what that frees is one clock of hardware and a loop
// here. Only the excess over the 5 ns inflates a window, and this is what says
// whether a reported delta is the structure or the simulator.
// Counter ticks this queue's model spent beyond the latency it was modelling,
// summed over every call. Work that fits inside the modelled latency is hidden by
// the spin to the deadline and costs the measurement nothing; only this excess
// lands in the window, so it is what a reported window must exclude to describe
// the design rather than the simulator.
uint64_t queue_self_overrun_ticks(uint32_t queue_idx);

void queue_push_overrun(uint32_t queue_idx, uint64_t *total_us, uint64_t *calls, uint64_t *mean_work_ns);

void queue_poll_overrun(uint32_t queue_idx, uint64_t *total_us, uint64_t *calls);

// Where a poll's own work goes, per call in nanoseconds: retiring the ends that
// came due, and assembling the status register. The remainder is the rest of the
// call. Splits the figure queue_poll_work reports so the cost can be attributed.
void queue_poll_profile(uint32_t queue_idx, uint64_t *retire_ns, uint64_t *status_ns, uint64_t *calls);

// The simulator's own cost inside read_queue_status, in counter ticks: the
// total spent retiring cores and pushing work, the worst single call, and how
// many calls were made. The modeled poll latency is only honest while that work
// fits inside it, so this is what says whether a measured window reflects the
// structure or the simulator.
void queue_poll_work(uint32_t queue_idx, uint64_t *total_ticks, uint64_t *max_ticks, uint64_t *calls);

// Why a queue is not placing work. `polls` is every status read; `stalled` are
// those where entries were waiting and none could be pushed; `stall_idle_cores`
// sums the cores standing idle at those moments, and `stall_mix_head` counts how
// many had a mix group at the head. Idle cores behind an unplaceable head are the
// cost of pushing only the head, and this is what measures it.
// Core time this queue's cores have actually been committed to work, split by
// type, in counter ticks. Against the scheduling window it says whether a group
// is saturated or starved — and which half of it.
void queue_busy_ticks(uint32_t queue_idx, uint64_t *aic_busy, uint64_t *aiv_busy);

// Inter-queue load imbalance as this manager sampled it.
//
//   mean_spread / max_spread  entries between the deepest and shallowest queue
//   mean_own_depth            this queue's own ring occupancy
//   deepest_pct               share of samples where this queue was the deepest
//   starved_pct               share of samples where this queue had an idle core
//                             of a type another queue had queued work for
//
// The last is the one that costs makespan: work sitting at a peer that this group
// had a core for. The others describe the distribution that produces it.
void queue_imbalance(
    uint32_t queue_idx, uint64_t *samples, uint64_t *mean_spread, uint64_t *max_spread, uint64_t *mean_own_depth,
    uint64_t *deepest_pct, uint64_t *starved_pct
);

// Cancellations this queue sent to reclaim a bound task, how many beat the
// core's own promotion, and the core-time the successful ones bought -- the wait
// the task would have served in its pipeline slot, less what the round trip and
// the idle core's own latch cost. Negative on a steal from a core that was about
// to finish anyway.
void queue_steal_stats(uint32_t queue_idx, uint64_t *tried, uint64_t *won, int64_t *gain_us);

// The wait between reaching the queue and running, split in two, in nanoseconds:
// `ring_ns` is the mean time an entry is held unassigned because no core of its
// type could take it, `bound_ns` the mean time it then spends in a core's
// pipeline slot. Saturation shows up in the first, premature binding in the
// second. The manager's own ready -> submit link sees neither: it ends when the
// task is handed to the queue.
void queue_residency(uint32_t queue_idx, uint64_t *ring_ns, uint64_t *bound_ns, uint64_t *n);

// Cube-core idle time this queue accumulated, in core-microseconds, integrated
// over the manager's polls: the total, the part falling while the queue holds an
// unstarted cohort member, and the wall time it held one. A cohort blocks every
// core it is staged on until the whole cohort is released device-wide, so these
// say how much of the group's starvation that barrier owns.
void queue_aic_idle(uint32_t queue_idx, uint64_t *idle_us, uint64_t *idle_gated_us, uint64_t *gated_us);

// When the core running `index` ended, in the manager's own clock domain. This is
// a diagnostic read of state the queue already holds, not a modelled access: it
// charges nothing, because no hardware register carries it and a design that
// needed one would have to justify it separately.
uint64_t queue_finish_ts(uint32_t queue_idx, uint64_t index);

// How long a finished task waits between becoming reportable and a poll making it
// legible to the manager, in nanoseconds. A core cannot be given its next task
// until the manager reacts to the one that ended, so this is the slack between a
// core going idle and the manager being able to do anything about it.
void queue_retire_lag(uint32_t queue_idx, uint64_t *mean_ns, uint64_t *max_ns, uint64_t *n);

// How deep the look-ahead list has ever been, and how many finishes found it full
// and fell back to waiting for the watermark. A non-zero drop count means the
// buffer's depth, not the ordering, is what bounds out-of-order retirement.
void queue_ahead_stats(uint32_t queue_idx, uint32_t *high_water, uint64_t *dropped);

void queue_stall_stats(
    uint32_t queue_idx, uint64_t *polls, uint64_t *stalled, uint64_t *stall_idle_cores, uint64_t *stall_mix_head,
    uint64_t *stall_cube_head, uint64_t *stall_idle_aic, uint64_t *stall_idle_aiv
);

// Poll a GroupQueue's status register: charge the read, retire what has become
// legible, offer the head to a controller, and report the register. One read can
// retire many positions, which is the whole point of a watermark — the manager no
// longer asks each core in turn.
SimQueueStatus read_queue_status(uint32_t queue_idx);

}  // namespace asimgq
