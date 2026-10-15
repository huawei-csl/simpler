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
// Timing is real time less the model's own excess (Mode A): each call holds its
// caller for its modeled latency in real time, and a task's compute elapses
// because its end is scheduled at `start + ack + compute` while the scheduler
// works. Model times are on the queue's clock -- real time less every tick its
// model has spent beyond the latencies it charged -- so the cores stand still
// while the model overruns. Nothing
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
// Dependency comparators per task slot, each naming one producer's range of
// positions. A task with more unmet producers than this cannot be expressed to the
// controller, so the manager holds it back.
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

// A producer a held entry waits on: the `count` consecutive positions from `first`,
// in queue `queue`'s own numbering. A producer's tasks take consecutive positions,
// so one range names all of them.
struct SimDep {
    uint64_t first;
    uint32_t count;
    uint32_t queue;
};

// How long a completion one GroupQueue records takes to become legible to another.
// Each queue forwards what it records to its peers, which keep a copy; the trip is
// a store across the die between two controllers, anchored where the
// AICPU-to-queue store is.
constexpr uint64_t SIM_GQ_PEER_LINK_NS = 310;

// Whether this run's queues forward their completions to each other. A run that
// feeds silos needs it; any other leaves it off, so no queue spends simulator work
// publishing what no peer reads. Set before any queue is accessed.
void set_peer_forwarding(bool on);

// Hand the controller `n` single-part entries that wait on the same producers,
// which may sit in any queue. Each entry takes one hold slot, and is placed once
// every named position has finished -- and, for a position in another queue, once
// that finish has crossed the link. Returns false, admitting nothing, when the hold
// slots or comparators would not cover the request.
bool submit_held(
    uint32_t queue_idx, const uint64_t *gq_index, const int32_t *func_id, const SimTaskType *type, uint32_t n,
    int32_t task_id, const SimDep *deps, uint32_t dep_n
);

// How long a completion one GroupQueue records takes to reach the others: the
// controllers exchange their watermark and look-ahead state through device memory.
constexpr uint64_t SIM_GQ_EXCHANGE_NS = 300;
// A controller broadcasts its state when it changes, at most once per this period.
// The model sends on a fixed grid of the period, in the first slot at or after the
// completion: an isolated completion waits up to one period the hardware would not
// charge it, and a burst goes out together.
constexpr uint64_t SIM_GQ_BROADCAST_NS = 100;
// Entries one controller's partition store holds: its current subpartition and as
// much of the next as fits.
constexpr uint32_t SIM_PART_STORE = 512;
// Indices one partition may span: an index is 11 bits.
constexpr uint32_t SIM_PART_INDICES = 2048;
// A controller's look-ahead: the finished indices past its watermark it can hold.
// An entry starts only once the watermark is within this many indices of it, so
// its finish has a bit to land in.
constexpr uint32_t SIM_PART_LOOKAHEAD = 512;

// A producer a partition task waits on, named by supertask: the runtime's task id
// and the queues that run a piece of it. One comparator, however many queues --
// numbered across the partition, a supertask's tasks are one range of indices.
struct SimPartDep {
    int32_t task;
    uint32_t queues;  // bit q: queue q runs a piece of it
};

// One entry of a partition, a single-part task on this queue, its partition-local
// index, and the set of conditions it waits on.
struct SimPartEntry {
    uint64_t index;  // this queue's position, which its manager retires
    int32_t func_id;
    int32_t task_id;  // the supertask it is a piece of
    SimTaskType type;
    uint32_t set;
    uint32_t pindex;  // its index in the partition
};
// What a set of entries waits for: the producers deps[first .. first + n), and the
// partition's watermark reaching `threshold` -- every index below it finished. A
// set's entries are one supertask's piece. An empty set's entries finish without
// a core once its conditions are met, as if they had run.
struct SimPartSet {
    uint32_t first;
    uint32_t n;
    uint32_t threshold;
    bool empty;
};

// Switch this run's queues to partition mode, or out of it. In it, each queue keeps
// a store for the partitions it is given and exchanges completions with the others.
// Call before any queue is accessed in the run.
void set_partition_mode(bool on);

// Give this queue entries of partition `part`, which spans `size` indices. Every
// queue learns a partition's size from its own manager, with n == 0 when its share
// is empty. An entry waits for its queue to reach its partition -- the watermark of
// the one before covering all of it -- then for its set's producers and threshold:
// a producer finished here is known at once, elsewhere SIM_GQ_EXCHANGE_NS after
// that queue's next broadcast. Every entry of one supertask's piece on this queue
// arrives in one call. Returns false, admitting nothing, when the store cannot
// take them.
bool submit_partition(
    uint32_t queue_idx, int32_t part, uint32_t size, const SimPartEntry *entries, uint32_t n, const SimPartSet *sets,
    uint32_t set_n, const SimPartDep *deps, uint32_t dep_n
);

// The partition this queue is running now, and how many entries its store can
// still take. Part of the status a manager reads anyway.
int32_t queue_partition(uint32_t queue_idx);
uint32_t part_store_room(uint32_t queue_idx);

// The partition store over the run: the most entries resident at once; completions
// learned from the other queues; releases timed on one, and how many were learned
// only after this queue's replay had passed them, by how much in all, and by the
// kind of condition learned last -- arrival, a producer here or in another queue,
// the threshold, the partition switch; sets that waited on the watermark after
// their producers had finished, and for how long -- for a threshold, and for the
// look-ahead; empty pieces finished; partitions this queue moved past; and entries
// that arrived after their queue had reached their partition, and how long after.
struct SimPartStats {
    uint64_t resident_peak;
    uint64_t events_in;
    uint64_t timed;
    uint64_t late;
    uint64_t late_ns;
    uint64_t late_by[5];
    uint64_t wm_waits;
    uint64_t wm_wait_ns;
    uint64_t la_waits;
    uint64_t la_wait_ns;
    uint64_t skips;
    uint64_t switches;
    uint64_t behind;
    uint64_t behind_ns;
};
void queue_part_stats(uint32_t queue_idx, SimPartStats *out);

// Whether a status read of this queue now could report anything the last one did
// not, or let the controller act on anything it has not seen: a finish that has
// become legible, a timed release that is due, a peer position a held entry waits
// on that has since been published, or a pre-stage. Reads nothing a manager could
// not learn by polling and charges nothing; a manager that skips a read this calls
// idle loses nothing, since every event carries its own instant and is replayed at
// the next read.
bool poll_may_report(uint32_t queue_idx);

// The manager has retired everything the status read it just made reported. Only
// a read whose result is retired from counts: poll_may_report() compares against
// this, and a read made for another purpose leaves finishes unretired.
void status_consumed(uint32_t queue_idx);

// For the stall diagnostics: the queue's replay state and every dependency set
// it holds -- what each waits on here and in its peers. Returns the length written.
int describe_held(uint32_t queue_idx, char *buf, uint32_t cap);

// Admissions whose named producers had all become legible to this queue before the
// entries arrived, and by how long in all.
void queue_admit_lateness(uint32_t queue_idx, uint64_t *late, uint64_t *late_ns);

// What crossed between queues for this one: positions it published for its peers,
// ranges in other queues its held entries resolved, releases timed on a peer's
// completion, and how many it learned only after its own replay had passed them --
// placed late, by `late_ns` in all.
void queue_peer_stats(
    uint32_t queue_idx, uint64_t *published, uint64_t *resolved, uint64_t *timed, uint64_t *late, uint64_t *late_ns
);

// Hand the controller an early-dispatch pre-stage: a task staged before its
// producers finish, whose payload is gated on a doorbell. It is placed as soon as
// a core can take it and then holds that core, not started, until post_doorbell()
// names (bell_core, bell_token) -- the manager's core and dispatch token.
bool submit_belled(
    uint32_t queue_idx, const uint64_t *gq_index, const int32_t *func_id, uint32_t parts, int32_t task_id,
    SimTaskType type, uint32_t bell_core, uint32_t bell_token
);

// The manager rang the doorbell of `token` on its core `bell_core`. Callable from
// any scheduler thread: it only posts the doorbell, which the queue holding the
// pre-stage takes on its next access. The store reaches the controller after the
// queue-arrival latency and the core one link later.
void post_doorbell(uint32_t bell_core, uint32_t token);

// Map a per-core register base address to the index post_doorbell() takes.
uint32_t core_index_for_addr(uint64_t reg_addr);

// Doorbells this queue took, and how many of its pre-stages still wait for one.
// Any left waiting at the end of a run is a pre-stage that never started.
void doorbell_counts(uint32_t queue_idx, uint64_t *taken, uint64_t *waiting);

// For the stall diagnostics: each pre-stage of this queue still waiting, with its
// key and what the mailbox of its manager core holds. Returns the length written.
int describe_waiting_bells(uint32_t queue_idx, char *buf, uint32_t cap);

// How the hold array is being used: entries admitted with unmet in-group
// producers, how many the controller later placed itself, and how many it refused
// for want of room. `promoted` against `admitted` is the direct measure of work
// the AICPU did not have to do.
void queue_group_stats(uint32_t queue_idx, uint64_t *admitted, uint64_t *promoted, uint64_t *refused, uint32_t *high);

// How many compute samples came from the calibration table versus the default.
void calib_coverage(uint64_t *hits, uint64_t *misses);

// Dispatches charged per func_id -- compared across models, says whether both ran the same work.
void func_hist(uint64_t *out, uint32_t n);

// What one read of a queue's status register yields: the completed prefix and the
// finishes past it, which a manager needs on every pass, and the room left for
// held entries, which it sizes its intake by. Only what a manager reads is
// assembled; anything more would be simulator work inside the poll.
// Positions a queue can report as finished ahead of its prefix, held as one bit
// each: bit i is the completion of `watermark + 1 + i`. The watermark's own
// position is not stored -- it is not complete by definition, or the prefix would
// already have passed it -- so the window covers one position more than it has
// bits.
//
// A bit vector rather than a list of indices because the prefix advance is then a
// find-first-zero and a shift, both single-cycle, where a list has to be searched
// associatively once per position the prefix crosses. It also makes the whole
// record one word to read instead of one word per live entry.
//
// The horizon is hard: a position past `watermark + SIM_AHEAD_HORIZON` has no bit
// and could not be recorded. `GqIndexSpace` refuses to issue one, so the case is
// unreachable rather than merely unlikely -- see `ahead_overflow`, which latches
// if that guard is ever wrong.

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
// Positions the look-ahead covers above the watermark, held inside the queue.
// Far past anything the workloads here reach -- paged_attention's furthest finish
// is 21 above the watermark -- because the consequence of exceeding it is a
// stranded task rather than a slow one, and a graph this has never run is not
// evidence about its disorder. At one bit each it is 128 bytes per queue, still a
// fifth of what the list of indices it replaced occupied.
constexpr uint32_t SIM_AHEAD_HORIZON = 1024;
constexpr uint32_t SIM_AHEAD_WORDS = SIM_AHEAD_HORIZON / 64;

// Of those, how many the status register carries. The register is one 64-bit
// word -- a 32-bit watermark beside this many bits -- so the manager's whole read
// is a single access however much finished out of order, where a list of indices
// cost one access per live entry. Positions above this window are reported on a
// later poll, once the watermark has moved under them; the same was true of the
// list, which carried its lowest entries first.
//
// One word also makes the read atomic, which relative bits require and absolute
// indices did not: a bitmap read against a stale watermark misreads every
// position, where a stale index still says what it means.
constexpr uint32_t SIM_AHEAD_PUBLISHED = 32;

struct SimQueueStatus {
    uint64_t watermark;  // contiguous completed prefix
    // Finished positions past the watermark: bit i is set when `watermark + 1 + i`
    // has finished, over the first SIM_AHEAD_PUBLISHED positions above it.
    //
    // Without these a task that finishes out of order cannot retire until
    // everything before it has, so it cannot release its consumers either: one
    // long task holds up every shorter one behind it.
    uint32_t ahead_bits;
    // Entries the controller can still hold for tasks whose in-group producers
    // have not retired. A manager sizes its intake by this, so it never offers
    // work the controller has nowhere to put -- overflow is prevented on the
    // manager's side rather than reported back from the device.
    uint32_t held_room;
};

// Start every cohort member this queue holds, no earlier than `at` -- an instant on
// the real clock, which the queue converts to its own -- and no earlier than the
// core it sits on comes free. The manager calls this
// once the cohort is placed device-wide; until then the members occupy their
// cores without running, which is what co-residency costs. Charged as a posted
// write into the queue, like submitting an entry.
void release_cohort(uint32_t queue_idx, uint64_t at);

// Cohort members this queue holds placed but not started: a field of the status
// register, read on its own while a cohort assembles so the ordinary poll does not
// carry it. Charged as a status read.
uint32_t read_cohort_staged(uint32_t queue_idx);

// Cohort members this queue had to place behind running work rather than on an
// idle core, over the run.
uint64_t queue_cohort_piped(uint32_t queue_idx);

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
// No doorbell: an entry that is not an early-dispatch pre-stage.
constexpr uint32_t SIM_NO_BELL = UINT32_MAX;

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
    // An early-dispatch pre-stage: placed like any other entry, but its slot holds
    // without starting until the doorbell keyed (bell_core, bell_token) is rung --
    // the manager's core and dispatch token, because the manager does not know
    // which core the controller chose. If the doorbell is rung before the entry is
    // placed, it is placed as an ordinary entry.
    uint32_t bell_core = SIM_NO_BELL;
    uint32_t bell_token = 0;
    // The supertask this entry is a piece of, for an entry of a partition; -1
    // otherwise. The queue counts a piece finished when every entry of it has. The
    // entry's partition and its index there, which its finish marks known.
    int32_t piece_task = -1;
    int32_t piece_part = -1;
    uint32_t piece_index = 0;
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

// The furthest above the watermark any finish has landed, and how many fell
// outside the horizon entirely. The first says how much of the window the
// workload's completion disorder actually uses; the second must stay zero, since
// a position past the horizon cannot be recorded at all.
void queue_ahead_stats(uint32_t queue_idx, uint32_t *high_water, uint64_t *overflow);

// Poll a GroupQueue's status register: charge the read, retire what has become
// legible, and report the register. One read can retire many positions, which is
// the whole point of a watermark — the manager no longer asks each core in turn.
SimQueueStatus read_queue_status(uint32_t queue_idx);

}  // namespace asimgq
