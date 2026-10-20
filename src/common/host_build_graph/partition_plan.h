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
#include <type_traits>

// The partition plan: what the compiler hands the group controllers, through their
// managers, for a graph that declares partitions. The host builds it in the
// compiler's place once the graph is complete.
//
// Every group numbers its share of each partition -- its subpartition -- itself, in
// one local id stream of its own that runs across the partitions: a piece of a
// supertask takes one id per block and active subtask, a conditional task's home
// slot one id, and a producer run in another group one id, the proxy, placed just
// below the first of its dependents here. A record names its producers by local ids
// only. A producer that has a proxy anywhere carries a global id, which its pieces
// announce on the ring when they finish, and each group's map takes it to the proxy.
//
// Conditional work -- a predicated supertask narrow enough to run whole in one group
// -- goes as a chain: the conditional tasks of one partition that share a predicate
// and depend on each other. A chain that is skipped finishes through its tasks' home
// slots. A chain that runs is dealt to a group once decided, and takes ids from that
// group's dynamic region of the partition.
//
// The wire format is 32-bit words, every offset in words from the plan's first one.
// It holds no pointer, so it crosses to the device inside the shared-memory image.
namespace part_plan {

// The groups a plan is built for: one controller and its manager each.
constexpr uint32_t kGroups = 4;
// Each controller's look-ahead past its local watermark, in local ids. A
// subpartition that spans more no longer fits it whole.
constexpr uint32_t kLookahead = 512;
// Producers one record names; the rest end below its threshold.
constexpr uint32_t kNamed = 4;
constexpr uint32_t kNone = 0xFFFFFFFFU;

struct Header {
    uint32_t words;   // the plan's size
    uint32_t groups;  // kGroups
    uint32_t partitions;
    uint32_t chains;
    uint32_t sub_offset;    // SubPartition[partitions][groups]
    uint32_t chain_offset;  // Chain[chains]
    uint32_t gids;          // global ids handed out: 1 .. gids
    uint32_t reserved;
};

// One group's share of one partition.
struct SubPartition {
    uint32_t base;  // its first local id
    uint32_t span;  // the local ids it takes, its dynamic region included
    uint32_t items;
    uint32_t item_offset;  // Item[items], in local id order
    uint32_t map;
    uint32_t map_offset;  // MapEntry[map]
    uint32_t dyn_lo;      // the dynamic region: ids for the chains dealt here
    uint32_t dyn_count;
    uint32_t chain_first;  // the partition's chains, the same for every group
    uint32_t chain_count;
};

struct Range {
    uint32_t lo;
    uint32_t hi;
};

enum ItemKind : uint32_t {
    kPiece = 0,  // the blocks of a supertask this group runs
    kSlot = 1,   // a conditional task's home slot: one id, whatever its blocks
};
// Item flags.
constexpr uint32_t kFlagPredicated = 1U << 0;  // a piece posted decided: to run, or empty
constexpr uint32_t kFlagHold = 1U << 1;        // it names a producer the controllers do not run
constexpr uint32_t kFlagDecides = 1U << 2;     // the slot whose decision deals its chain
constexpr uint32_t kFlagBarrier = 1U << 3;     // its threshold is the subpartition's first id
constexpr uint32_t kFlagAnnounce = 1U << 4;    // some other group maps its global id

struct Item {
    int32_t task;
    uint32_t local;  // its first local id
    uint32_t block_lo;
    uint32_t block_hi;
    uint32_t kind;
    uint32_t flags;
    uint32_t threshold;  // the local watermark must reach it
    uint32_t gid;        // matched against its own map when it finishes, and announced if kFlagAnnounce; 0 for none
    uint32_t chain;      // a slot's chain, else kNone
    uint32_t named;
    Range ranges[kNamed];
};

struct MapEntry {
    uint32_t gid;
    uint32_t local;
    uint32_t expected;  // the announcements the id waits for, from every group
};

struct Chain {
    uint32_t part;
    uint32_t owner;  // the group whose manager decides it
    uint32_t tasks;
    uint32_t ids;          // the local ids it takes when it runs
    uint32_t task_offset;  // ChainTask[tasks]
    uint32_t external;     // its producers outside it, which must be complete
    uint32_t ext_offset;   // int32 task ids
    uint32_t reserved;
};

// One task of a chain, as it runs where the chain is dealt: its ids and its producers
// in the chain, relative to the chain's first id.
struct ChainTask {
    int32_t task;
    uint32_t offset;
    uint32_t gid;
    uint32_t named;
    Range ranges[kNamed];
};

template <typename T>
constexpr uint32_t words_of() {
    static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>);
    static_assert(sizeof(T) % sizeof(uint32_t) == 0);
    return static_cast<uint32_t>(sizeof(T) / sizeof(uint32_t));
}

template <typename T>
const T *at(const uint32_t *plan, uint32_t offset) {
    return reinterpret_cast<const T *>(plan + offset);
}

}  // namespace part_plan
