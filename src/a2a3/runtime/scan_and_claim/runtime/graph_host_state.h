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

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

struct ChipTaskSlotState;
struct GraphHostState;

inline constexpr size_t GRAPH_MAX_DEFINITIONS = 16;

struct GraphHostStateDeleter {
    void operator()(GraphHostState *state) const noexcept;
};

using GraphHostStatePtr = std::unique_ptr<GraphHostState, GraphHostStateDeleter>;

struct GraphHostUpload {
    ChipTaskSlotState *outer_slot;
    uint64_t full_key;
    uint64_t definition_hash;
};

// The run's distinct Definition images (already deduplicated by the host-side
// Definition cache), for upload as shared device objects ahead of submissions.
struct GraphHostDefinition {
    uint64_t full_key;
    const std::byte *data;
    size_t bytes;
};

struct GraphHostDefinitionList {
    std::vector<GraphHostDefinition> entries;
};

GraphHostStatePtr make_graph_host_state();
size_t graph_host_upload_count(const GraphHostState &state);
std::optional<GraphHostUpload> graph_host_upload(GraphHostState &state, size_t index);
GraphHostDefinitionList graph_host_definitions(GraphHostState &state);
