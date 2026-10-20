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
// The partition plan, built on the host in the compiler's place: every group's
// local numbering of its share of each partition, the proxies that stand for
// producers run in other groups, the global ids those producers announce, each
// group's map, and the chains of conditional work. See partition_plan.h.

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "common/unified_log.h"
#include "host_build_graph/orchestrator.h"
#include "host_build_graph/partition_plan.h"

namespace {

using part_plan::kGroups;
using part_plan::kNamed;
using part_plan::kNone;

struct TaskInfo {
    bool runs = false;        // the controllers run it
    bool cond = false;        // conditional work: predicated, and run whole in one group
    bool spread = false;      // split by block range across every group
    bool predicated = false;  // a spread task posted decided
    bool cohort = false;      // a sync-start spread task, released whole
    bool hold = false;        // it names a producer the controllers do not run
    int32_t home = -1;        // the group of a narrow task, or a conditional task's home slot
    int32_t part = -1;
    int32_t level = 0;
    int32_t blocks = 0;
    int32_t subs = 0;
    int32_t chain = -1;
    uint32_t gid = 0;
    std::array<int32_t, kGroups> local{};   // its item's first local id in each group, or -1
    std::array<uint32_t, kGroups> count{};  // the ids of that item
    std::array<int32_t, kGroups> proxy{};   // the id standing for it in each group, or -1
    std::array<bool, kGroups> end_proxy{};  // a later partition's item in that group names it
};

struct ChainInfo {
    int32_t part = -1;
    int32_t owner = -1;
    uint32_t ids = 0;
    std::vector<int32_t> tasks;     // in task order
    std::vector<uint32_t> offsets;  // per task: its first id relative to the chain's start
    std::vector<int32_t> external;  // producers outside it
};

struct SubInfo {
    uint32_t base = 0;
    uint32_t span = 0;
    uint32_t dyn_lo = 0;
    uint32_t dyn_count = 0;
    std::vector<part_plan::Item> items;
    std::vector<part_plan::MapEntry> map;
};

int32_t find_root(std::vector<int32_t> &parent, int32_t x) {
    while (parent[static_cast<size_t>(x)] != x) {
        parent[static_cast<size_t>(x)] = parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
        x = parent[static_cast<size_t>(x)];
    }
    return x;
}

// The blocks [lo, hi) group `g` runs of a task split across every group.
void spread_blocks(int32_t blocks, uint32_t g, uint32_t &lo, uint32_t &hi) {
    lo = static_cast<uint32_t>((static_cast<int64_t>(blocks) * g) / kGroups);
    hi = static_cast<uint32_t>((static_cast<int64_t>(blocks) * (g + 1)) / kGroups);
}

bool same_predicate(const ChipTaskSlotState &a, const ChipTaskSlotState &b) {
    const DispatchPredicate &pa = a.to_payload().predicate;
    const DispatchPredicate &pb = b.to_payload().predicate;
    return a.to_descriptor().predicate_writer == b.to_descriptor().predicate_writer && pa.addr == pb.addr &&
           pa.target == pb.target && pa.elem_size == pb.elem_size && pa.op == pb.op;
}

}  // namespace

bool OrchestratorState::build_partition_plan(
    const std::vector<int32_t> &starts, const std::vector<int32_t> &level, const std::vector<int32_t> &order
) {
    SharedMemoryTaskHeader &tasks = sm_header->tasks;
    const int32_t total = starts.back();
    const uint32_t parts = static_cast<uint32_t>(starts.size() - 1);
    std::vector<TaskInfo> info(static_cast<size_t>(total));
    uint64_t stamp_waves = 0, dealt_waves = 0;
    bool deal = false;
    auto is_dummy = [&](int32_t id) {
        const ChipTaskSlotState &s = tasks.get_slot_state_by_task_id(id);
        return s.task_kind == TaskKind::DUMMY || s.active_mask.to_shape() == ResourceShape::DUMMY;
    };
    for (uint32_t p = 0; p < parts; ++p) {
        for (int32_t id = starts[p]; id < starts[p + 1]; ++id) {
            TaskInfo &t = info[static_cast<size_t>(id)];
            t.part = static_cast<int32_t>(p);
            t.level = level[static_cast<size_t>(id)];
            t.local.fill(-1);
            t.proxy.fill(-1);
            const ChipTaskSlotState &slot = tasks.get_slot_state_by_task_id(id);
            const TaskDescriptor &desc = tasks.get_task_by_task_id(id);
            if (desc.silo == TASK_SILO_NONE || tasks.is_completed(id) || slot.logical_block_num <= 0) continue;
            int32_t subs = 0;
            for (int32_t sub = 0; sub < SUBTASK_SLOT_COUNT; ++sub) {
                if (slot.active_mask.subtask_active(static_cast<SubtaskSlot>(sub))) ++subs;
            }
            if (subs == 0) continue;
            t.runs = true;
            t.blocks = slot.logical_block_num;
            t.subs = subs;
            t.spread = desc.silo == TASK_SILO_SPREAD;
            t.predicated = slot.task_attrs.has_predicate() && t.spread;
            t.cond = slot.task_attrs.has_predicate() && !t.spread;
            t.cohort = slot.task_attrs.requires_sync_start();
            if (!t.spread) {
                t.home = static_cast<int32_t>(static_cast<uint32_t>(desc.silo - TASK_SILO_FIRST) % kGroups);
            }
        }
    }
    // Where narrow tasks run. Dealt, each goes to the group given the fewest blocks of
    // its core types so far, taken in the order the plan numbers each partition -- by
    // level -- so every level's work spreads over the groups. The plan deals them
    // unless the host's stamp spreads the work as well: as few waves of a group's
    // cores per level and core type, and, on a tie, no more blocks in the busiest
    // group of each type. The stamp keeps a task with its latest producer, so fewer
    // producers need a global id. Conditional work keeps the stamp either way, where
    // its chain's slots sit.
    {
        const std::array<uint32_t, 2> cores{
            std::max<uint32_t>(1, static_cast<uint32_t>(total_cluster_count) / kGroups),
            std::max<uint32_t>(1, static_cast<uint32_t>(total_aiv_count) / kGroups)
        };
        auto shape_of = [&](int32_t id) {
            const ActiveMask &mask = tasks.get_slot_state_by_task_id(id).active_mask;
            return std::array<bool, 2>{
                mask.subtask_active(SubtaskSlot::AIC),
                mask.subtask_active(SubtaskSlot::AIV0) || mask.subtask_active(SubtaskSlot::AIV1)
            };
        };
        std::vector<int32_t> dealt(static_cast<size_t>(total), -1);
        std::array<std::array<uint64_t, 2>, kGroups> work{};
        for (const int32_t id : order) {
            const TaskInfo &t = info[static_cast<size_t>(id)];
            if (!t.runs || t.cond) continue;
            const std::array<bool, 2> type = shape_of(id);
            auto load = [&](uint32_t g) {
                return (type[0] ? work[g][0] : 0) + (type[1] ? work[g][1] : 0);
            };
            auto add = [&](uint32_t g, uint32_t blocks) {
                for (uint32_t k = 0; k < 2; ++k) {
                    if (type[k]) work[g][k] += blocks;
                }
            };
            if (t.spread) {
                for (uint32_t g = 0; g < kGroups; ++g) {
                    uint32_t lo = 0, hi = 0;
                    spread_blocks(t.blocks, g, lo, hi);
                    add(g, hi - lo);
                }
                continue;
            }
            uint32_t best = 0;
            for (uint32_t g = 1; g < kGroups; ++g) {
                if (load(g) < load(best)) best = g;
            }
            dealt[static_cast<size_t>(id)] = static_cast<int32_t>(best);
            add(best, static_cast<uint32_t>(t.blocks));
        }
        // Per level of every partition and per core type, the waves of a group's cores
        // its busiest group needs, summed; and per core type, the blocks of the busiest
        // group over the whole graph, summed.
        auto spread_of = [&](bool use_dealt, uint64_t &busiest) {
            uint64_t sum = 0;
            std::array<std::array<uint64_t, 2>, kGroups> all{};
            size_t i = 0;
            while (i < order.size()) {
                const TaskInfo &head = info[static_cast<size_t>(order[i])];
                std::array<std::array<uint64_t, 2>, kGroups> lv{};
                size_t j = i;
                for (; j < order.size(); ++j) {
                    const TaskInfo &t = info[static_cast<size_t>(order[j])];
                    if (t.part != head.part || t.level != head.level) break;
                    if (!t.runs) continue;
                    const std::array<bool, 2> type = shape_of(order[j]);
                    for (uint32_t g = 0; g < kGroups; ++g) {
                        uint32_t lo = 0, hi = 0;
                        if (t.spread) {
                            spread_blocks(t.blocks, g, lo, hi);
                        } else {
                            const int32_t home = use_dealt && !t.cond ? dealt[static_cast<size_t>(order[j])] : t.home;
                            hi = home == static_cast<int32_t>(g) ? static_cast<uint32_t>(t.blocks) : 0;
                        }
                        for (uint32_t k = 0; k < 2; ++k) {
                            if (type[k]) {
                                lv[g][k] += hi - lo;
                                all[g][k] += hi - lo;
                            }
                        }
                    }
                }
                for (uint32_t k = 0; k < 2; ++k) {
                    uint64_t most = 0;
                    for (uint32_t g = 0; g < kGroups; ++g) {
                        most = std::max<uint64_t>(most, (lv[g][k] + cores[k] - 1) / cores[k]);
                    }
                    sum += most;
                }
                i = j;
            }
            busiest = 0;
            for (uint32_t k = 0; k < 2; ++k) {
                uint64_t most = 0;
                for (uint32_t g = 0; g < kGroups; ++g) {
                    most = std::max<uint64_t>(most, all[g][k]);
                }
                busiest += most;
            }
            return sum;
        };
        uint64_t stamp_busiest = 0, dealt_busiest = 0;
        stamp_waves = spread_of(false, stamp_busiest);
        dealt_waves = spread_of(true, dealt_busiest);
        deal = dealt_waves < stamp_waves || (dealt_waves == stamp_waves && dealt_busiest < stamp_busiest);
        if (deal) {
            for (const int32_t id : order) {
                if (dealt[static_cast<size_t>(id)] >= 0)
                    info[static_cast<size_t>(id)].home = dealt[static_cast<size_t>(id)];
            }
        }
    }
    // Producers, seen through dependency-only tasks: those the controllers run, and
    // every other one still to finish, which a manager has to wait for itself.
    std::vector<std::vector<int32_t>> prods(static_cast<size_t>(total));
    std::vector<std::vector<int32_t>> others(static_cast<size_t>(total));
    std::vector<int32_t> walk;
    std::vector<int32_t> seen(static_cast<size_t>(total), -1);
    for (int32_t id = 0; id < total; ++id) {
        if (!info[static_cast<size_t>(id)].runs) continue;
        const TaskPayload &payload = tasks.get_slot_state_by_task_id(id).to_payload();
        walk.assign(payload.fanin_data(), payload.fanin_data() + payload.fanin_count);
        while (!walk.empty()) {
            const int32_t prod = walk.back();
            walk.pop_back();
            if (prod < 0 || prod >= id || seen[static_cast<size_t>(prod)] == id) continue;
            seen[static_cast<size_t>(prod)] = id;
            if (tasks.is_completed(prod)) continue;
            if (info[static_cast<size_t>(prod)].runs) {
                prods[static_cast<size_t>(id)].push_back(prod);
            } else if (is_dummy(prod)) {
                const TaskPayload &pp = tasks.get_slot_state_by_task_id(prod).to_payload();
                walk.insert(walk.end(), pp.fanin_data(), pp.fanin_data() + pp.fanin_count);
            } else {
                others[static_cast<size_t>(id)].push_back(prod);
                info[static_cast<size_t>(id)].hold = true;
            }
        }
    }
    // Where each task's item sits: a piece in every group of a spread task, the
    // group of a narrow one, a conditional task's home slot.
    auto in_group = [&](const TaskInfo &t, uint32_t g) {
        if (!t.runs) return false;
        if (t.spread) {
            uint32_t lo = 0, hi = 0;
            spread_blocks(t.blocks, g, lo, hi);
            return lo < hi;
        }
        return t.home == static_cast<int32_t>(g);
    };
    // Whether a consumer in group `g` sees producer `t` through a proxy: whenever any
    // of it runs, or may run, outside `g`.
    auto needs_proxy = [&](const TaskInfo &t, uint32_t g) {
        return t.spread || t.home != static_cast<int32_t>(g);
    };
    auto expected = [&](const TaskInfo &t) {
        if (!t.spread) return 1U;
        uint32_t n = 0;
        for (uint32_t g = 0; g < kGroups; ++g) {
            if (in_group(t, g)) ++n;
        }
        return n;
    };
    // Chains: the conditional tasks of one partition that share a predicate and
    // depend on each other.
    std::vector<int32_t> parent(static_cast<size_t>(total));
    std::iota(parent.begin(), parent.end(), 0);
    for (int32_t id = 0; id < total; ++id) {
        if (!info[static_cast<size_t>(id)].cond) continue;
        for (const int32_t prod : prods[static_cast<size_t>(id)]) {
            const TaskInfo &pt = info[static_cast<size_t>(prod)];
            if (pt.cond && pt.part == info[static_cast<size_t>(id)].part &&
                same_predicate(tasks.get_slot_state_by_task_id(id), tasks.get_slot_state_by_task_id(prod))) {
                parent[static_cast<size_t>(find_root(parent, id))] = find_root(parent, prod);
            }
        }
    }
    std::vector<ChainInfo> chains;
    std::vector<int32_t> chain_of_root(static_cast<size_t>(total), -1);
    for (int32_t id = 0; id < total; ++id) {
        TaskInfo &t = info[static_cast<size_t>(id)];
        if (!t.cond) continue;
        const int32_t root = find_root(parent, id);
        if (chain_of_root[static_cast<size_t>(root)] < 0) {
            chain_of_root[static_cast<size_t>(root)] = static_cast<int32_t>(chains.size());
            chains.push_back(ChainInfo{});
            chains.back().part = t.part;
            chains.back().owner = t.home;
        }
        t.chain = chain_of_root[static_cast<size_t>(root)];
        ChainInfo &c = chains[static_cast<size_t>(t.chain)];
        c.tasks.push_back(id);
        c.offsets.push_back(c.ids);
        c.ids += static_cast<uint32_t>(t.blocks * t.subs);
    }
    uint32_t cross_chain = 0;
    for (ChainInfo &c : chains) {
        for (const int32_t id : c.tasks) {
            for (const int32_t prod : prods[static_cast<size_t>(id)]) {
                const TaskInfo &pt = info[static_cast<size_t>(prod)];
                if (pt.chain == info[static_cast<size_t>(id)].chain) continue;
                if (pt.cond) ++cross_chain;
                c.external.push_back(prod);
            }
            c.external.insert(
                c.external.end(), others[static_cast<size_t>(id)].begin(), others[static_cast<size_t>(id)].end()
            );
        }
        std::sort(c.external.begin(), c.external.end());
        c.external.erase(std::unique(c.external.begin(), c.external.end()), c.external.end());
    }
    // Global ids: every conditional task, whose home slot a run elsewhere completes,
    // and every producer some group sees through a proxy.
    auto consumer_groups = [&](const TaskInfo &t, auto &&fn) {
        for (uint32_t g = 0; g < kGroups; ++g) {
            if (in_group(t, g)) fn(g);
        }
    };
    std::vector<bool> needs_gid(static_cast<size_t>(total), false);
    for (int32_t id = 0; id < total; ++id) {
        const TaskInfo &t = info[static_cast<size_t>(id)];
        if (!t.runs) continue;
        if (t.cond) needs_gid[static_cast<size_t>(id)] = true;
        consumer_groups(t, [&](uint32_t g) {
            for (const int32_t prod : prods[static_cast<size_t>(id)]) {
                TaskInfo &pt = info[static_cast<size_t>(prod)];
                if (pt.part + 1 < t.part) continue;
                if (needs_proxy(pt, g)) needs_gid[static_cast<size_t>(prod)] = true;
                if (pt.part < t.part) pt.end_proxy[g] = true;
            }
        });
    }
    uint32_t gids = 0;
    for (int32_t id = 0; id < total; ++id) {
        if (needs_gid[static_cast<size_t>(id)]) info[static_cast<size_t>(id)].gid = ++gids;
    }
    // Whether some group sees a task through a proxy.
    std::vector<bool> remote_proxy(static_cast<size_t>(total), false);
    for (int32_t id = 0; id < total; ++id) {
        const TaskInfo &t = info[static_cast<size_t>(id)];
        if (!t.runs) continue;
        consumer_groups(t, [&](uint32_t g) {
            for (const int32_t prod : prods[static_cast<size_t>(id)]) {
                const TaskInfo &pt = info[static_cast<size_t>(prod)];
                if (pt.part + 1 >= t.part && needs_proxy(pt, g)) remote_proxy[static_cast<size_t>(prod)] = true;
            }
        });
    }
    // Every group numbers its share of each partition in level order. A producer seen
    // through a proxy takes one id just below the first of its dependents here; one a
    // later partition names, at the end of its own partition's share; the chains
    // dealt here, a region at the level the partition's conditional work starts.
    std::vector<SubInfo> subs(static_cast<size_t>(parts) * kGroups);
    std::array<uint32_t, kGroups> next{};
    // `order` holds each partition's tasks in level order, one partition after another.
    std::vector<size_t> order_start(parts + 1, order.size());
    for (size_t i = order.size(); i-- > 0;) {
        order_start[static_cast<size_t>(info[static_cast<size_t>(order[i])].part)] = i;
    }
    for (uint32_t p = parts; p-- > 0;) {
        order_start[p] = std::min(order_start[p], order_start[p + 1]);
    }
    uint32_t below_violations = 0;
    for (uint32_t p = 0; p < parts; ++p) {
        const auto first = order.begin() + static_cast<ptrdiff_t>(order_start[p]);
        const auto last = order.begin() + static_cast<ptrdiff_t>(order_start[p + 1]);
        // The partition's chains, and the ids the most a group may be dealt take: the
        // largest of them, as many as one group's share of the count.
        std::vector<uint32_t> sizes;
        int32_t cond_level = INT32_MAX;
        for (const ChainInfo &c : chains) {
            if (c.part == static_cast<int32_t>(p)) sizes.push_back(c.ids);
        }
        for (auto it = first; it != last; ++it) {
            if (info[static_cast<size_t>(*it)].cond)
                cond_level = std::min(cond_level, info[static_cast<size_t>(*it)].level);
        }
        std::sort(sizes.begin(), sizes.end(), std::greater<>());
        const size_t share = (sizes.size() + kGroups - 1) / kGroups;
        const uint32_t dyn = std::accumulate(sizes.begin(), sizes.begin() + static_cast<ptrdiff_t>(share), 0U);
        for (uint32_t g = 0; g < kGroups; ++g) {
            SubInfo &sp = subs[static_cast<size_t>(p) * kGroups + g];
            sp.base = next[g];
            uint32_t cur = next[g];
            bool dyn_placed = sizes.empty();
            auto place_proxy = [&](int32_t prod) {
                TaskInfo &pt = info[static_cast<size_t>(prod)];
                if (pt.proxy[g] >= 0) return;
                pt.proxy[g] = static_cast<int32_t>(cur);
                sp.map.push_back(part_plan::MapEntry{pt.gid, cur, expected(pt)});
                ++cur;
            };
            // A record names four producers and its threshold covers the rest, so a
            // proxy only a threshold covers goes where its producer's level falls here,
            // keeping the threshold low; one a record names goes just below the record.
            std::vector<std::pair<int32_t, int32_t>> by_level;  // (level, producer), placed at that level
            for (auto it = first; it != last; ++it) {
                const TaskInfo &t = info[static_cast<size_t>(*it)];
                if (!in_group(t, g)) continue;
                uint32_t same = 0;
                for (const int32_t prod : prods[static_cast<size_t>(*it)]) {
                    if (info[static_cast<size_t>(prod)].part == static_cast<int32_t>(p)) ++same;
                }
                if (same <= kNamed) continue;
                for (const int32_t prod : prods[static_cast<size_t>(*it)]) {
                    const TaskInfo &pt = info[static_cast<size_t>(prod)];
                    if (pt.part == static_cast<int32_t>(p) && needs_proxy(pt, g)) by_level.emplace_back(pt.level, prod);
                }
            }
            std::sort(by_level.begin(), by_level.end());
            by_level.erase(std::unique(by_level.begin(), by_level.end()), by_level.end());
            size_t next_level = 0;
            for (auto it = first; it != last; ++it) {
                const int32_t id = *it;
                TaskInfo &t = info[static_cast<size_t>(id)];
                if (!in_group(t, g)) continue;
                while (next_level < by_level.size() && by_level[next_level].first < t.level) {
                    place_proxy(by_level[next_level++].second);
                }
                if (!dyn_placed && t.level >= cond_level) {
                    sp.dyn_lo = cur;
                    sp.dyn_count = dyn;
                    cur += dyn;
                    dyn_placed = true;
                }
                for (const int32_t prod : prods[static_cast<size_t>(id)]) {
                    const TaskInfo &pt = info[static_cast<size_t>(prod)];
                    if (pt.part == static_cast<int32_t>(p) && needs_proxy(pt, g)) place_proxy(prod);
                }
                uint32_t lo = 0, hi = static_cast<uint32_t>(t.blocks);
                if (t.spread) spread_blocks(t.blocks, g, lo, hi);
                t.local[g] = static_cast<int32_t>(cur);
                t.count[g] = t.cond ? 1U : (hi - lo) * static_cast<uint32_t>(t.subs);
                part_plan::Item item{};
                item.task = id;
                item.local = cur;
                item.block_lo = t.cond ? 0 : lo;
                item.block_hi = t.cond ? 0 : hi;
                item.kind = t.cond ? part_plan::kSlot : part_plan::kPiece;
                item.flags = (t.predicated ? part_plan::kFlagPredicated : 0) | (t.hold ? part_plan::kFlagHold : 0) |
                             (t.cohort ? part_plan::kFlagCohort : 0);
                // Every item with a global id retires its own map entry when it finishes;
                // only one some other group maps takes a cycle of the ring.
                item.gid = t.gid;
                if (remote_proxy[static_cast<size_t>(id)]) item.flags |= part_plan::kFlagAnnounce;
                item.chain = t.cond ? static_cast<uint32_t>(t.chain) : kNone;
                if (t.cond && chains[static_cast<size_t>(t.chain)].tasks.front() == id)
                    item.flags |= part_plan::kFlagDecides;
                sp.items.push_back(item);
                if (t.cond) {
                    sp.map.push_back(part_plan::MapEntry{t.gid, cur, 1});
                }
                cur += t.count[g];
            }
            if (!dyn_placed) {
                sp.dyn_lo = cur;
                sp.dyn_count = dyn;
                cur += dyn;
            }
            for (auto it = first; it != last; ++it) {
                const TaskInfo &t = info[static_cast<size_t>(*it)];
                if (t.runs && t.end_proxy[g] && needs_proxy(t, g)) place_proxy(*it);
            }
            sp.span = cur - sp.base;
            next[g] = cur;
            // What each item waits on, in this group's ids: the four producers ending
            // last and the threshold where the fifth ends. A producer in the partition
            // before stands as its piece in this group or as the id at the end of its
            // own subpartition here that stands for it. One further back has finished
            // before the item is posted: the item goes out only once every group maps
            // its partition, which a group does once its watermark has passed the
            // partition two before.
            for (part_plan::Item &item : sp.items) {
                const TaskInfo &t = info[static_cast<size_t>(item.task)];
                std::vector<part_plan::Range> cands;
                for (const int32_t prod : prods[static_cast<size_t>(item.task)]) {
                    const TaskInfo &pt = info[static_cast<size_t>(prod)];
                    if (pt.part + 1 < t.part) continue;
                    part_plan::Range r{};
                    if (needs_proxy(pt, g)) {
                        r = part_plan::Range{
                            static_cast<uint32_t>(pt.proxy[g]), static_cast<uint32_t>(pt.proxy[g]) + 1
                        };
                    } else {
                        r = part_plan::Range{
                            static_cast<uint32_t>(pt.local[g]), static_cast<uint32_t>(pt.local[g]) + pt.count[g]
                        };
                    }
                    if (r.hi > item.local) ++below_violations;
                    cands.push_back(r);
                }
                std::sort(cands.begin(), cands.end(), [](const part_plan::Range &x, const part_plan::Range &y) {
                    return x.hi > y.hi;
                });
                item.named = static_cast<uint32_t>(std::min<size_t>(cands.size(), kNamed));
                for (uint32_t k = 0; k < item.named; ++k) {
                    item.ranges[k] = cands[k];
                }
                item.threshold = cands.size() > kNamed ? cands[kNamed].hi : 0;
            }
        }
    }
    // Check that the plan enforces every edge: a producer in the same partition or the
    // one before is named or below the threshold, and one further back is left to the
    // order partitions are posted in; a proxy is mapped in its producer's partition,
    // counting every piece, and every piece announces; a chain task names its
    // producers in the chain and lists the rest as external.
    uint32_t unenforced = 0;
    for (uint32_t p = 0; p < parts; ++p) {
        for (uint32_t g = 0; g < kGroups; ++g) {
            const SubInfo &sp = subs[static_cast<size_t>(p) * kGroups + g];
            for (const part_plan::Item &item : sp.items) {
                for (const int32_t prod : prods[static_cast<size_t>(item.task)]) {
                    const TaskInfo &pt = info[static_cast<size_t>(prod)];
                    if (pt.part + 1 < static_cast<int32_t>(p)) continue;
                    part_plan::Range h{};
                    if (needs_proxy(pt, g)) {
                        if (pt.proxy[g] < 0) {
                            ++unenforced;
                            continue;
                        }
                        h = part_plan::Range{
                            static_cast<uint32_t>(pt.proxy[g]), static_cast<uint32_t>(pt.proxy[g]) + 1
                        };
                        const SubInfo &home = subs[static_cast<size_t>(pt.part) * kGroups + g];
                        const bool mapped =
                            std::any_of(home.map.begin(), home.map.end(), [&](const part_plan::MapEntry &m) {
                                return m.gid == pt.gid && m.local == h.lo && m.expected == expected(pt);
                            });
                        if (!mapped || pt.gid == 0) ++unenforced;
                    } else {
                        h = part_plan::Range{
                            static_cast<uint32_t>(pt.local[g]), static_cast<uint32_t>(pt.local[g]) + pt.count[g]
                        };
                    }
                    bool named = false;
                    for (uint32_t k = 0; k < item.named; ++k) {
                        named = named || (item.ranges[k].lo == h.lo && item.ranges[k].hi == h.hi);
                    }
                    if (!named && h.hi > item.threshold) ++unenforced;
                }
                const TaskInfo &t = info[static_cast<size_t>(item.task)];
                if (t.hold && (item.flags & part_plan::kFlagHold) == 0) ++unenforced;
                if (item.gid != t.gid ||
                    (remote_proxy[static_cast<size_t>(item.task)] && (item.flags & part_plan::kFlagAnnounce) == 0)) {
                    ++unenforced;
                }
            }
        }
    }
    for (size_t ci = 0; ci < chains.size(); ++ci) {
        const ChainInfo &c = chains[ci];
        for (const int32_t id : c.tasks) {
            for (const int32_t prod : prods[static_cast<size_t>(id)]) {
                const bool inside = info[static_cast<size_t>(prod)].chain == static_cast<int32_t>(ci);
                if (!inside && !std::binary_search(c.external.begin(), c.external.end(), prod)) ++unenforced;
            }
            for (const int32_t prod : others[static_cast<size_t>(id)]) {
                if (!std::binary_search(c.external.begin(), c.external.end(), prod)) ++unenforced;
            }
        }
    }
    if (unenforced > 0) {
        LOG_ERROR("Partition plan leaves %u dependencies unenforced; running the graph without partitions", unenforced);
        return false;
    }
    // The cohorts, in the order every controller is given their pieces: by partition,
    // then by the level order its stream follows.
    std::vector<part_plan::Cohort> cohorts;
    for (const int32_t id : order) {
        const TaskInfo &t = info[static_cast<size_t>(id)];
        if (!t.runs || !t.cohort) continue;
        part_plan::Cohort c{};
        c.task = id;
        c.subs = static_cast<uint32_t>(t.subs);
        for (uint32_t g = 0; g < kGroups; ++g) {
            uint32_t lo = 0, hi = 0;
            spread_blocks(t.blocks, g, lo, hi);
            c.blocks[g] = hi - lo;
        }
        cohorts.push_back(c);
    }
    // Serialize: header, subpartitions, items, maps, chains, chain tasks, externals,
    // cohorts.
    using part_plan::words_of;
    uint32_t n_items = 0, n_map = 0, n_chain_tasks = 0, n_ext = 0;
    for (const SubInfo &sp : subs) {
        n_items += static_cast<uint32_t>(sp.items.size());
        n_map += static_cast<uint32_t>(sp.map.size());
    }
    uint32_t intra_over = 0;
    for (const ChainInfo &c : chains) {
        n_chain_tasks += static_cast<uint32_t>(c.tasks.size());
        n_ext += static_cast<uint32_t>(c.external.size());
    }
    const uint32_t sub_offset = words_of<part_plan::Header>();
    const uint32_t item_offset = sub_offset + static_cast<uint32_t>(subs.size()) * words_of<part_plan::SubPartition>();
    const uint32_t map_offset = item_offset + n_items * words_of<part_plan::Item>();
    const uint32_t chain_offset = map_offset + n_map * words_of<part_plan::MapEntry>();
    const uint32_t ctask_offset = chain_offset + static_cast<uint32_t>(chains.size()) * words_of<part_plan::Chain>();
    const uint32_t ext_offset = ctask_offset + n_chain_tasks * words_of<part_plan::ChainTask>();
    const uint32_t cohort_offset = ext_offset + n_ext;
    const uint32_t words = cohort_offset + static_cast<uint32_t>(cohorts.size()) * words_of<part_plan::Cohort>();
    const uint64_t capacity = task_allocator.capacity() * static_cast<uint64_t>(CHIP_MAX_FANIN);
    const int32_t plan_at = CHIP_ALIGN_UP(fanin_pool_cursor, ARG_POOL_ALIGN / static_cast<int32_t>(sizeof(int32_t)));
    if (static_cast<uint64_t>(plan_at) + words > capacity) {
        LOG_ERROR(
            "Partition plan needs %u words past the fanin pool's %d, which holds %" PRIu64
            "; running the graph without "
            "partitions",
            words, plan_at, capacity
        );
        return false;
    }
    uint32_t *plan = reinterpret_cast<uint32_t *>(fanin_pool + plan_at);
    std::memset(plan, 0, static_cast<size_t>(words) * sizeof(uint32_t));
    part_plan::Header header{words,        kGroups,      parts, static_cast<uint32_t>(chains.size()),
                             sub_offset,   chain_offset, gids,  static_cast<uint32_t>(cohorts.size()),
                             cohort_offset};
    std::memcpy(plan + cohort_offset, cohorts.data(), cohorts.size() * sizeof(part_plan::Cohort));
    std::memcpy(plan, &header, sizeof(header));
    uint32_t item_at = item_offset, map_at = map_offset;
    std::vector<uint32_t> chain_first_of(parts + 1, 0);
    for (const ChainInfo &c : chains) {
        ++chain_first_of[static_cast<size_t>(c.part) + 1];
    }
    for (uint32_t p = 0; p < parts; ++p) {
        chain_first_of[p + 1] += chain_first_of[p];
    }
    uint32_t span_max = 0, span_over = 0, map_max = 0;
    uint64_t span_sum = 0;
    for (uint32_t p = 0; p < parts; ++p) {
        for (uint32_t g = 0; g < kGroups; ++g) {
            const SubInfo &sp = subs[static_cast<size_t>(p) * kGroups + g];
            part_plan::SubPartition rec{
                sp.base,
                sp.span,
                static_cast<uint32_t>(sp.items.size()),
                item_at,
                static_cast<uint32_t>(sp.map.size()),
                map_at,
                sp.dyn_lo,
                sp.dyn_count,
                chain_first_of[p],
                chain_first_of[p + 1] - chain_first_of[p],
            };
            std::memcpy(plan + sub_offset + (p * kGroups + g) * words_of<part_plan::SubPartition>(), &rec, sizeof(rec));
            std::memcpy(plan + item_at, sp.items.data(), sp.items.size() * sizeof(part_plan::Item));
            std::memcpy(plan + map_at, sp.map.data(), sp.map.size() * sizeof(part_plan::MapEntry));
            item_at += static_cast<uint32_t>(sp.items.size()) * words_of<part_plan::Item>();
            map_at += static_cast<uint32_t>(sp.map.size()) * words_of<part_plan::MapEntry>();
            span_max = std::max(span_max, sp.span);
            span_sum += sp.span;
            if (sp.span > part_plan::kLookahead) ++span_over;
            map_max = std::max(map_max, static_cast<uint32_t>(sp.map.size()));
        }
    }
    uint32_t ctask_at = ctask_offset, ext_at = ext_offset;
    for (uint32_t ci = 0; ci < chains.size(); ++ci) {
        const ChainInfo &c = chains[ci];
        part_plan::Chain rec{
            static_cast<uint32_t>(c.part),
            static_cast<uint32_t>(c.owner),
            static_cast<uint32_t>(c.tasks.size()),
            c.ids,
            ctask_at,
            static_cast<uint32_t>(c.external.size()),
            ext_at,
            0,
        };
        std::memcpy(plan + chain_offset + ci * words_of<part_plan::Chain>(), &rec, sizeof(rec));
        for (size_t k = 0; k < c.tasks.size(); ++k) {
            const int32_t id = c.tasks[k];
            part_plan::ChainTask ct{};
            ct.task = id;
            ct.offset = c.offsets[k];
            ct.gid = info[static_cast<size_t>(id)].gid;
            std::vector<part_plan::Range> cands;
            for (const int32_t prod : prods[static_cast<size_t>(id)]) {
                if (info[static_cast<size_t>(prod)].chain != static_cast<int32_t>(ci)) continue;
                const size_t j = static_cast<size_t>(std::find(c.tasks.begin(), c.tasks.end(), prod) - c.tasks.begin());
                const TaskInfo &pt = info[static_cast<size_t>(prod)];
                cands.push_back(
                    part_plan::Range{c.offsets[j], c.offsets[j] + static_cast<uint32_t>(pt.blocks * pt.subs)}
                );
            }
            if (cands.size() > kNamed) ++intra_over;
            std::sort(cands.begin(), cands.end(), [](const part_plan::Range &x, const part_plan::Range &y) {
                return x.hi > y.hi;
            });
            ct.named = static_cast<uint32_t>(std::min<size_t>(cands.size(), kNamed));
            for (uint32_t r = 0; r < ct.named; ++r) {
                ct.ranges[r] = cands[r];
            }
            std::memcpy(plan + ctask_at, &ct, sizeof(ct));
            ctask_at += words_of<part_plan::ChainTask>();
        }
        std::memcpy(plan + ext_at, c.external.data(), c.external.size() * sizeof(int32_t));
        ext_at += static_cast<uint32_t>(c.external.size());
    }
    if (intra_over > 0 || below_violations > 0) {
        LOG_ERROR(
            "Partition plan is inconsistent (%u chain tasks name more than %u producers in their chain, %u producers "
            "sit above a dependent); running the graph without partitions",
            intra_over, kNamed, below_violations
        );
        return false;
    }
    fanin_pool_cursor = plan_at + static_cast<int32_t>(words);
    plan_first = plan_at;
    plan_words = words;
    LOG_INFO(
        "Partition plan: %u partitions, %u groups, %u words; narrow tasks %s (level waves %" PRIu64 " dealt, %" PRIu64
        " stamped); subpartition span max %u mean %.0f (%u over the %u-id look-ahead); map max %u per subpartition; "
        "%u global ids; %u chains, %u chain edges from another chain",
        parts, kGroups, words, deal ? "dealt" : "stamped", dealt_waves, stamp_waves, span_max,
        static_cast<double>(span_sum) / static_cast<double>(subs.size()), span_over, part_plan::kLookahead, map_max,
        gids, static_cast<uint32_t>(chains.size()), cross_chain
    );
    return true;
}
