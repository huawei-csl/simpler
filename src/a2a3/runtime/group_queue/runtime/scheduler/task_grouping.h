#pragma once

#include <stdint.h>

#include <atomic>

#include "host_build_graph/shared_memory.h"

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
// The grouping is *declared*, never inferred. `rt_group_begin` / `rt_group_end`
// around a run of submissions names that run a group, and the orchestrator stamps
// each member's descriptor with the group and the run it covers. Nothing here
// derives a group from a task id, a component walk, or any other property of the
// graph: a scheduler cannot tell whether an edge it sees stays inside a group, so
// only whoever builds the graph can say. Today the declaration is written by hand
// in the orchestration source; a compiler that knows the graph's structure emits
// the same two calls.
//
// Because a declaration covers consecutive submissions, a group's members are the
// id run `group_first .. group_first + group_extent`, and every member carries it.
// That is what lets a thread claiming a group by any one of its tasks feed the rest
// without a per-run table: the annotation is already in the descriptor it holds.
// =============================================================================

// Whether the grouping path is compiled in at all. 0 leaves every task ungrouped
// however the graph is annotated, which is the scheduler before any of this
// existed and the baseline the grouped runs are measured against.
#ifndef SIMPLER_GQ_GROUPING
#define SIMPLER_GQ_GROUPING 1
#endif

namespace gq_group {

constexpr bool ENABLED = SIMPLER_GQ_GROUPING != 0;

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

// The ready group queue. A group is ready once every edge entering it from
// outside is met. Edges run forward and a group is a contiguous id range, so a
// group's external producers all live in lower-numbered groups: groups become
// ready in order, and the queue is the position of the lowest group not yet
// handed out. A graph of self-contained groups has no external edges at all, so
// every group is ready from the start and the cursor just runs to the end.
// Whether the ready group queue delivers each task, decided once when the graph
// is seeded. The completion path asks this for every waiter it releases, so it
// has to be a bit test rather than a walk of the task's shape and predicate.
constexpr int32_t MAX_TASKS_TRACKED = 1 << 17;
inline uint64_t g_delivered[MAX_TASKS_TRACKED / 64];

inline void set_delivered(int32_t id, bool yes) {
    if (id < 0 || id >= MAX_TASKS_TRACKED) return;
    const uint64_t bit = 1ULL << (id & 63);
    if (yes) {
        g_delivered[id >> 6] |= bit;
    } else {
        g_delivered[id >> 6] &= ~bit;
    }
}

inline bool is_delivered(int32_t id) {
    if (id < 0 || id >= MAX_TASKS_TRACKED) return false;
    return (g_delivered[id >> 6] & (1ULL << (id & 63))) != 0;
}

// The declaration, as the orchestrator stamped it onto every task. A run reaches it
// through one pointer, latched before any thread is released into its dispatch loop,
// because the descriptor the annotation rides on is the one the scheduler already
// holds whenever it asks -- which is what keeps a lookup off any table of its own.
constexpr int32_t NO_GROUP = NO_TASK_GROUP;

inline const SharedMemoryTaskHeader *g_tasks;
inline int32_t g_group_count;

// Whether this run's graph declared any group. A graph that declares none is run
// exactly as a scheduler without grouping runs it: every group-aware path is a
// decision about a group, and there are none to decide about. ENABLED alone is not
// that test -- it only says the path is compiled in.
inline bool active() { return ENABLED && g_group_count > 0; }

inline void bind_declaration(const SharedMemoryTaskHeader *tasks) {
    g_tasks = tasks;
    g_group_count = 0;
}

inline const TaskDescriptor *descriptor_of(int32_t local_id) {
    if (!ENABLED || g_tasks == nullptr) return nullptr;
    if (local_id < 0 || local_id >= g_tasks->total_tasks) return nullptr;
    return &const_cast<SharedMemoryTaskHeader *>(g_tasks)->get_task_by_task_id(local_id);
}

// The annotation off a descriptor already in hand, which is where a caller holding
// the task should ask: reaching it by id costs the load a second time.
inline int32_t group_of(const TaskDescriptor &d) { return ENABLED ? d.group : NO_GROUP; }

inline int32_t group_of(int32_t local_id) {
    const TaskDescriptor *d = descriptor_of(local_id);
    return d != nullptr ? d->group : NO_GROUP;
}

// A group's members are the id run its declaration covers, so membership is a range
// test -- and the run is carried by every member. A consumer asking which of its
// producers its own controller holds therefore reads its own descriptor once and
// answers for every fanin without touching theirs, which is what keeps a per-fanin
// question off the producers' cache lines.
struct GroupRun {
    int32_t group = NO_GROUP;
    int32_t first = 0;
    int32_t end = 0;

    bool holds(int32_t local_id) const { return group != NO_GROUP && local_id >= first && local_id < end; }
};

inline GroupRun run_of(int32_t local_id) {
    const TaskDescriptor *d = descriptor_of(local_id);
    if (d == nullptr || d->group == NO_GROUP || d->group_extent <= 0) return GroupRun{};
    return GroupRun{d->group, d->group_first, d->group_first + d->group_extent};
}

// Whether two tasks are resolved by the same controller.
inline bool same_group(int32_t a, int32_t b) {
    const int32_t ga = group_of(a);
    return ga != NO_GROUP && ga == group_of(b);
}

// A group's member run, indexed by group. Built by walking the declaration rather
// than the tasks: each step reads one member's descriptor and jumps the whole run it
// names, so the walk costs one iteration per group, not per task.
inline int32_t g_group_first[MAX_GROUPS];
inline int32_t g_group_extent[MAX_GROUPS];

inline int32_t index_declaration() {
    if (!ENABLED || g_tasks == nullptr) return 0;
    const int32_t total = g_tasks->total_tasks;
    int32_t groups = 0;
    for (int32_t id = 0; id < total;) {
        const TaskDescriptor *d = descriptor_of(id);
        if (d == nullptr) break;
        const int32_t g = d->group;
        if (g < 0 || g >= MAX_GROUPS || d->group_extent <= 0) {
            ++id;  // undeclared: its own group, and it owns no run to skip
            continue;
        }
        g_group_first[g] = d->group_first;
        g_group_extent[g] = d->group_extent;
        if (g >= groups) groups = g + 1;
        id = d->group_first + d->group_extent;
    }
    g_group_count = groups;
    return groups;
}

inline int32_t group_member_count(int32_t group) {
    if (!ENABLED || group < 0 || group >= g_group_count) return 0;
    return g_group_extent[group];
}

inline int32_t group_member(int32_t group, int32_t index) {
    if (!ENABLED || group < 0 || group >= g_group_count) return -1;
    if (index < 0 || index >= g_group_extent[group]) return -1;
    return g_group_first[group] + index;
}

// Where to start looking for a group to take. Groups are handed out to whichever
// thread finds one ready, not in order: a graph whose groups depend on each other
// would otherwise offer only one at a time and leave every other controller idle,
// and a controller only owns its own slice of the cores.
inline std::atomic<int32_t> g_scan_from;

// How far past the scan point to look. Judging a group ready costs a walk of its
// fanin, so the window bounds what one probe can spend.
constexpr int32_t GROUP_SCAN_WINDOW = 24;

inline void reset_group_owners() {
    for (int32_t i = 0; i < MAX_GROUPS; ++i) {
        g_group_owner[i].store(NO_OWNER, std::memory_order_relaxed);
        g_group_opened[i].store(0, std::memory_order_relaxed);
        g_group_external[i] = false;
    }
    for (int32_t i = 0; i < MAX_TASKS_TRACKED / 64; ++i) g_delivered[i] = 0;
    g_group_count = 0;
    g_scan_from.store(0, std::memory_order_relaxed);
}

// Take `group` for `thread_idx`, once. Several groups may be in flight at a time,
// one per thread; a group still belongs to one thread for its whole life, because
// an edge can only be expressed between positions in the same controller.
inline bool try_claim_group(int32_t group, int32_t thread_idx) {
    if (!ENABLED || group < 0 || group >= g_group_count) return false;
    int32_t expected = NO_OWNER;
    if (!g_group_owner[group].compare_exchange_strong(
            expected, thread_idx, std::memory_order_acq_rel, std::memory_order_acquire
        )) {
        return false;
    }
    return true;
}

// Carry the scan mark past `group`, which is already taken. Groups are claimed
// out of order, so the mark cannot simply follow the group just claimed: leaving
// it on a group someone else holds anchors the window there for good, because
// that group can never be claimed again to move it on. Returns where the mark
// now stands.
inline int32_t bump_scan_mark(int32_t group) {
    int32_t from = group;
    if (g_scan_from.compare_exchange_strong(from, group + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return group + 1;
    }
    return from;  // another thread moved it; use where it landed
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
