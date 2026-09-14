#pragma once

#include <stdint.h>

#include <atomic>

// =============================================================================
// Task grouping: the contract between whoever builds the graph and the
// GroupQueue that runs it.
//
// A *group* is a set of tasks whose internal edges the controller resolves on its
// own cores. The manager submits a member once the edges entering the group are
// met; everything inside is the controller's business, and never becomes visible
// outside the GroupQueue that runs it.
//
// This is a contract, not a hint. The controller may assume that a task's
// producers are either already retired or members of the same group, and a graph
// that violates that waits on a completion no controller will ever see. The
// annotation is therefore checked where it is produced, not paid for at run time.
//
// A *sink* is a member with a consumer outside its group. Only a sink's
// completion reaches the chip-wide flags that every manager reads; an internal
// task finishing is a fact local to one controller and its manager.
//
// Provenance of the numbers below: derived from the case's own dependency graph
// (`--enable-dep-gen`), not guessed. paged_attention Case1 is 65,536 tasks in 256
// weakly-connected components of exactly 257, contiguous in local id, with zero
// edges between them -- one group per batch element, holding a 64-block
// QK->PV->SF->UP fan plus the reduction tree over its results.
//
// Contiguity is what lets the annotation be two integers rather than a table:
// group(local_id) = local_id / GROUP_SIZE. A graph whose groups are not
// contiguous needs a real per-task annotation, which is where a compiler-emitted
// one belongs.
// =============================================================================

// Tasks per group, or 0 when the graph carries no grouping and every task is its
// own group -- which is exactly the behaviour before any of this existed.
#ifndef SIMPLER_GQ_GROUP_SIZE
#define SIMPLER_GQ_GROUP_SIZE 0
#endif

namespace gq_group {

constexpr int32_t GROUP_SIZE = SIMPLER_GQ_GROUP_SIZE;
constexpr bool ENABLED = GROUP_SIZE > 0;

// GROUP_SIZE as a divisor. Arithmetic over group extents is compiled even in an
// ungrouped build, where it is never reached but must still not divide by zero.
constexpr int32_t GROUP_EXTENT = GROUP_SIZE > 0 ? GROUP_SIZE : 1;

// Which group a task belongs to. Ungrouped graphs put every task in its own
// group, so a controller resolves nothing locally and the behaviour is unchanged.
inline int32_t group_of(int32_t local_id) { return ENABLED ? local_id / GROUP_EXTENT : local_id; }

// Whether two tasks are resolved by the same controller.
inline bool same_group(int32_t a, int32_t b) { return group_of(a) == group_of(b); }

// The ready queue this contract is heading for holds *groups*, not tasks: a group
// becomes ready when the edges entering it are met -- which, since only a sink's
// completion leaves a group, means its dependencies are on sink tasks alone. A
// Scheduler thread pops one group and submits the whole of it, internal edges
// included, to its controller. That makes the queue 256 entries where PA's task
// queue is 65,536, and it makes ownership implicit: whoever popped the group
// scheduled it, so nothing has to be claimed or routed. The claim table below is
// the task-granular stepping stone to that.
//
// A group is scheduled by exactly one Scheduler thread: the first to reach any of
// its members claims it, and every later member of that group goes to the same
// thread. That is what makes a position meaningful to the controller it is sent
// to -- positions are handed out per thread, so a producer scheduled by a
// different thread names an index its consumer's controller cannot read.
//
// Indexed by group, so the table is one entry per 257 tasks rather than per task.
constexpr int32_t MAX_GROUPS = 8192;
constexpr int32_t NO_OWNER = -1;

inline std::atomic<int32_t> g_group_owner[MAX_GROUPS];

// The Scheduler thread running on this OS thread, set once as it enters its
// dispatch loop. A push decides a group's owner and can happen on any thread's
// completion path, which is otherwise given no thread index.
inline thread_local int32_t t_sched_thread = 0;

// Set for a group that some edge enters from outside it. Such a group cannot be
// opened wholesale -- its entry tasks wait on producers no controller of ours
// holds -- so it keeps the per-task readiness path.
inline bool g_group_external[MAX_GROUPS];

// Set once a group has been opened: its tasks are already queued on their owner's
// rows, so the completion path must not queue them a second time.
inline std::atomic<int32_t> g_group_opened[MAX_GROUPS];

// The ready group queue. A group is ready when the edges entering it are met;
// for a self-contained group that is true from the start, so the whole set is
// seeded once and handed out by a cursor. Groups that external edges reach are
// not in here at all -- they still arrive task by task.
inline int32_t g_ready_groups[MAX_GROUPS];
inline int32_t g_ready_group_count;
inline std::atomic<uint32_t> g_ready_group_cursor;

inline void reset_group_owners() {
    for (int32_t i = 0; i < MAX_GROUPS; ++i) {
        g_group_owner[i].store(NO_OWNER, std::memory_order_relaxed);
        g_group_opened[i].store(0, std::memory_order_relaxed);
        g_group_external[i] = false;
    }
    g_ready_group_count = 0;
    g_ready_group_cursor.store(0, std::memory_order_relaxed);
}

// The next ready group for `thread_idx`, or -1 when none is left. Taking it is
// what claims it: one group belongs to one thread for its whole life.
inline int32_t take_ready_group(int32_t thread_idx) {
    if (!ENABLED) return -1;
    const uint32_t i = g_ready_group_cursor.fetch_add(1, std::memory_order_acq_rel);
    if (i >= static_cast<uint32_t>(g_ready_group_count)) return -1;
    const int32_t g = g_ready_groups[i];
    g_group_owner[g].store(thread_idx, std::memory_order_release);
    return g;
}

inline bool group_is_opened(int32_t group) {
    if (!ENABLED || group < 0 || group >= MAX_GROUPS) return false;
    return g_group_opened[group].load(std::memory_order_acquire) != 0;
}

inline void mark_group_opened(int32_t group) {
    if (group >= 0 && group < MAX_GROUPS) g_group_opened[group].store(1, std::memory_order_release);
}

// The thread that owns `group`, claiming it for `thread_idx` when it is free.
// Groups past the table fall back to the calling thread, which costs them the
// local resolution but never misroutes: such a task carries no group deps.
inline int32_t claim_group(int32_t group, int32_t thread_idx) {
    if (group < 0 || group >= MAX_GROUPS) return thread_idx;
    std::atomic<int32_t> &owner = g_group_owner[group];
    int32_t cur = owner.load(std::memory_order_acquire);
    if (cur != NO_OWNER) return cur;
    int32_t expected = NO_OWNER;
    if (owner.compare_exchange_strong(expected, thread_idx, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return thread_idx;
    }
    return expected;
}

// The owner of `group`, or NO_OWNER while it is unclaimed.
inline int32_t group_owner(int32_t group) {
    if (group < 0 || group >= MAX_GROUPS) return NO_OWNER;
    return g_group_owner[group].load(std::memory_order_acquire);
}

}  // namespace gq_group
