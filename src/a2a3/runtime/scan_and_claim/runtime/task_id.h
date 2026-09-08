/*
 * Frozen at 324e90a7, the last commit before main's host_build_graph refactor.
 *
 * scan_and_claim forked this runtime before its shared Graph/TaskId layer was
 * split per runtime and redesigned (in-graph-task naming, sorted CSR, classified
 * scalar sources). This runtime does not execute recorded Graphs at all -- the
 * window scan skips TaskKind::GRAPH and GRAPH_NODE -- so it tracks none of that;
 * it keeps the layout it forked against, which is also what its shared-memory
 * image encodes. Sitting in this directory, it wins a flat #include over the
 * shared copy by the same-directory rule.
 */

/**
 * TaskId — minimal standalone header.
 *
 * Factored out of runtime_types.h so that tensor.h can include it
 * without pulling in scheduler-internal constants (heap sizes, timeouts, etc.).
 */

#pragma once

#include <cstdint>

/**
 * TaskId: 64-bit encoding shared by both runtimes.
 *
 * raw encoding: (ring_id << 32) | local_id
 *
 * ring_id:  which ring layer (0..CHIP_MAX_RING_DEPTH-1)
 * local_id: per-ring monotonic counter
 *
 * Invalid sentinel: raw == UINT64_MAX (no valid task has this encoding).
 */
struct TaskId {
    uint64_t raw;

    static constexpr TaskId make(uint8_t ring_id, uint32_t local_id) {
        return TaskId{(static_cast<uint64_t>(ring_id) << 32) | static_cast<uint64_t>(local_id)};
    }

    static constexpr TaskId invalid() { return TaskId{UINT64_MAX}; }

    constexpr uint8_t ring() const { return static_cast<uint8_t>(raw >> 32); }
    constexpr uint32_t local() const { return static_cast<uint32_t>(raw & 0xFFFFFFFFu); }
    constexpr bool is_valid() const { return raw != UINT64_MAX; }
    constexpr bool is_invalid() const { return raw == UINT64_MAX; }

    constexpr bool operator==(const TaskId &other) const { return raw == other.raw; }
    constexpr bool operator!=(const TaskId &other) const { return raw != other.raw; }
};

static_assert(sizeof(TaskId) == 8, "TaskId must stay 8 bytes (shared memory ABI)");
