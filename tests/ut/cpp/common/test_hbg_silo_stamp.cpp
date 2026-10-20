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
/**
 * Silo stamping, decided by the host at submit once the graph declares silos:
 *
 *   TASK_SILO_NONE    nothing declared, or a task the silos cannot feed ahead
 *   TASK_SILO_SPREAD  at least TASK_SILO_SPREAD_MIN_BLOCKS blocks: one piece per silo
 *   a hint            anything narrower: its latest stamped producer's, else the
 *                     next one in turn
 *
 * and, with them, the partitions, the partition plan the controllers run them from,
 * and the writer of the element a dispatch predicate reads.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "utils/device_arena.h"
#include "scheduler/scheduler.h"
#include "host_build_graph/orchestrator.h"
#include "host_build_graph/partition_plan.h"
#include "host_build_graph/shared_memory.h"
#include "host_build_graph/task_id.h"

class HbgSiloStampTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    SharedMemoryHandle *sm_handle = nullptr;
    OrchestratorState orch{};
    SchedulerState sched{};
    SchedulerLayout sched_layout{};
    std::vector<char> gm_heap;
    std::vector<TensorCreateInfo> create_infos;

    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(1 << 16);
        create_infos.reserve(16);
        sched_layout = SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);
        ASSERT_TRUE(sched.init_data_from_layout(sched_layout, runtime_arena, sm_handle->sm_base));
        sched.wire_arena_pointers(sched_layout, runtime_arena);
        sched.seed_queue_slots();
        ASSERT_TRUE(orch.init(sm_handle->sm_base, gm_heap.data(), gm_heap.size(), CHIP_DEFAULT_GRAPH_TASKS));
        orch.begin_scope();
    }

    void TearDown() override {
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }

    // One AIV task of `blocks` blocks with an output tensor, depending on `deps`.
    TaskId submit(int16_t blocks, const TaskId *deps = nullptr, uint32_t dep_count = 0) {
        uint32_t shape[] = {16};
        CoreTaskArgs args;
        create_infos.emplace_back(shape, 1, DataType::FLOAT32);
        args.add_output(create_infos.back());
        args.launch_spec.set_block_num(blocks);
        if (dep_count > 0) args.set_dependencies(deps, dep_count);
        MixedKernels mixed{};
        mixed.aiv0_kernel_id = 0;
        TaskOutputTensors out = orch.submit_task(mixed, args);
        EXPECT_TRUE(out.task_id().is_valid());
        return out.task_id();
    }

    uint8_t silo_of(TaskId id) { return sm_handle->header->tasks.get_task_by_task_id(id.local_id()).silo; }

    // The group a narrow task runs in, under the plan's group count.
    uint32_t group_of(TaskId id) { return static_cast<uint32_t>(silo_of(id) - TASK_SILO_FIRST) % part_plan::kGroups; }

    const uint32_t *plan() const {
        return reinterpret_cast<const uint32_t *>(orch.fanin_pool + orch.partition_plan_first());
    }

    const part_plan::SubPartition &sub(uint32_t part, uint32_t g) const {
        const auto &h = *part_plan::at<part_plan::Header>(plan(), 0);
        return *part_plan::at<part_plan::SubPartition>(
            plan(), h.sub_offset + (part * part_plan::kGroups + g) * part_plan::words_of<part_plan::SubPartition>()
        );
    }

    // The item of `id` in a subpartition, or nullptr.
    const part_plan::Item *item_of(const part_plan::SubPartition &sp, TaskId id) const {
        for (uint32_t i = 0; i < sp.items; ++i) {
            const auto *item =
                part_plan::at<part_plan::Item>(plan(), sp.item_offset + i * part_plan::words_of<part_plan::Item>());
            if (item->task == id.local_id()) return item;
        }
        return nullptr;
    }
};

TEST_F(HbgSiloStampTest, UndeclaredGraphStampsNothing) {
    const TaskId wide = submit(8);
    const TaskId narrow = submit(1, &wide, 1);
    EXPECT_EQ(silo_of(wide), TASK_SILO_NONE);
    EXPECT_EQ(silo_of(narrow), TASK_SILO_NONE);
    EXPECT_EQ(orch.silo_tasks(), 0);
}

TEST_F(HbgSiloStampTest, WideTaskIsSpreadAndNarrowerOneIsNot) {
    orch.declare_silos();
    const TaskId wide = submit(TASK_SILO_SPREAD_MIN_BLOCKS);
    const TaskId narrow = submit(TASK_SILO_SPREAD_MIN_BLOCKS - 1);
    EXPECT_EQ(silo_of(wide), TASK_SILO_SPREAD);
    EXPECT_GE(silo_of(narrow), TASK_SILO_FIRST);
    EXPECT_EQ(orch.silo_tasks(), 2);
}

TEST_F(HbgSiloStampTest, NarrowRootsTakeHintsInTurn) {
    orch.declare_silos();
    const TaskId a = submit(1);
    const TaskId b = submit(1);
    const TaskId c = submit(1);
    EXPECT_EQ(silo_of(a), TASK_SILO_FIRST);
    EXPECT_EQ(silo_of(b), TASK_SILO_FIRST + 1);
    EXPECT_EQ(silo_of(c), TASK_SILO_FIRST + 2);
}

TEST_F(HbgSiloStampTest, NarrowConsumerFollowsItsLatestStampedProducer) {
    orch.declare_silos();
    const TaskId a = submit(1);
    const TaskId b = submit(1);
    const TaskId both[] = {a, b};
    const TaskId c = submit(1, both, 2);
    EXPECT_EQ(silo_of(c), silo_of(b));
}

TEST_F(HbgSiloStampTest, ConsumerOfOnlySpreadProducersTakesTheNextHint) {
    orch.declare_silos();
    const TaskId root = submit(1);
    const TaskId wide = submit(8, &root, 1);
    const TaskId tail = submit(1, &wide, 1);
    EXPECT_EQ(silo_of(wide), TASK_SILO_SPREAD);
    EXPECT_EQ(silo_of(tail), TASK_SILO_FIRST + 1);
}

// Partitions: cut once the graph is complete, each boundary moved within a quarter
// partition to where the fewest dependencies cross, every task stamped with its
// partition, and a plan built for the controllers.
TEST_F(HbgSiloStampTest, PartitionsTieBreakTowardTheTargetSize) {
    orch.declare_partitions(4);
    // Two chains of four, the second fed by the last task of the first: every cut
    // near 4 crosses one dependency, so the boundary stays at 4.
    TaskId prev = submit(1);
    for (int i = 0; i < 3; ++i)
        prev = submit(1, &prev, 1);
    TaskId c = submit(1, &prev, 1);
    for (int i = 0; i < 3; ++i)
        c = submit(1, &c, 1);
    orch.finish_partitions();
    EXPECT_EQ(orch.partitions(), 2);
    auto &tasks = sm_handle->header->tasks;
    for (int32_t id = 0; id < 4; ++id)
        EXPECT_EQ(tasks.get_task_by_task_id(id).part, 0);
    for (int32_t id = 4; id < 8; ++id)
        EXPECT_EQ(tasks.get_task_by_task_id(id).part, 1);
    ASSERT_GE(orch.partition_plan_first(), 0);
    const auto &h = *part_plan::at<part_plan::Header>(plan(), 0);
    EXPECT_EQ(h.words, orch.partition_plan_words());
    EXPECT_EQ(h.groups, part_plan::kGroups);
    EXPECT_EQ(h.partitions, 2U);
}

// A task with a producer in an earlier partition names none of it: its threshold is
// its group's share of the partition starting, which the group's watermark reaches
// once every earlier share has finished.
TEST_F(HbgSiloStampTest, PlanHoldsAnEarlierPartitionsConsumerAtTheBoundary) {
    orch.declare_partitions(4);
    std::vector<TaskId> chain{submit(1)};
    for (int i = 0; i < 7; ++i)
        chain.push_back(submit(1, &chain.back(), 1));
    orch.finish_partitions();
    ASSERT_EQ(orch.partitions(), 2);
    const uint32_t g = group_of(chain[0]);
    const part_plan::SubPartition &first = sub(0, g);
    const part_plan::SubPartition &second = sub(1, g);
    EXPECT_EQ(first.base, 0U);
    EXPECT_EQ(first.span, 4U);
    EXPECT_EQ(second.base, 4U);
    const part_plan::Item *head = item_of(second, chain[4]);
    ASSERT_NE(head, nullptr);
    EXPECT_EQ(head->named, 0U);
    EXPECT_EQ(head->threshold, second.base);
    EXPECT_NE(head->flags & part_plan::kFlagBarrier, 0U);
    const part_plan::Item *next = item_of(second, chain[5]);
    ASSERT_NE(next, nullptr);
    ASSERT_EQ(next->named, 1U);
    EXPECT_EQ(next->ranges[0].lo, head->local);
    EXPECT_EQ(next->ranges[0].hi, head->local + 1);
    EXPECT_EQ(next->flags & part_plan::kFlagBarrier, 0U);
}

TEST_F(HbgSiloStampTest, PartitionBoundaryMovesToTheNarrowerCut) {
    orch.declare_partitions(4);
    // Tasks 1 and 2 both feed task 4, so cuts at 3 and 4 cross three and two
    // dependencies; after task 4 only its edge to task 5 crosses.
    const TaskId t0 = submit(1);
    const TaskId t1 = submit(1, &t0, 1);
    const TaskId t2 = submit(1, &t1, 1);
    submit(1, &t1, 1);
    const TaskId into4[] = {t1, t2};
    const TaskId t4 = submit(1, into4, 2);
    const TaskId t5 = submit(1, &t4, 1);
    const TaskId t6 = submit(1, &t5, 1);
    submit(1, &t6, 1);
    orch.finish_partitions();
    auto &tasks = sm_handle->header->tasks;
    EXPECT_EQ(orch.partitions(), 2);
    EXPECT_EQ(tasks.get_task_by_task_id(4).part, 0);
    EXPECT_EQ(tasks.get_task_by_task_id(5).part, 1);
}

// Within a partition, tasks are ordered by dependency level, not submission order:
// a root submitted late still precedes a consumer submitted early.
TEST_F(HbgSiloStampTest, PartitionIndicesFollowDependencyLevel) {
    orch.declare_partitions(100);
    const TaskId a = submit(1);
    const TaskId b = submit(1, &a, 1);
    const TaskId c = submit(1);
    const TaskId d = submit(2, &b, 1);
    orch.finish_partitions();
    auto &tasks = sm_handle->header->tasks;
    EXPECT_EQ(orch.partitions(), 1);
    EXPECT_EQ(tasks.get_task_by_task_id(a.local_id()).part_index, 0);
    EXPECT_EQ(tasks.get_task_by_task_id(c.local_id()).part_index, 1);
    EXPECT_EQ(tasks.get_task_by_task_id(b.local_id()).part_index, 2);
    EXPECT_EQ(tasks.get_task_by_task_id(d.local_id()).part_index, 3);
}

// A dependency-only task takes no place, and its consumer sits one level past the
// dummy's producers rather than two.
TEST_F(HbgSiloStampTest, DummyTakesNoIndexAndPassesItsLevelThrough) {
    orch.declare_partitions(100);
    const TaskId a = submit(1);
    CoreTaskArgs args;
    args.set_dependencies(&a, 1);
    const TaskId dummy = orch.submit_dummy_task(args).task_id();
    ASSERT_TRUE(dummy.is_valid());
    const TaskId b = submit(1, &dummy, 1);
    const TaskId c = submit(1, &a, 1);
    const TaskId root = submit(1);
    orch.finish_partitions();
    auto &tasks = sm_handle->header->tasks;
    EXPECT_EQ(tasks.get_task_by_task_id(dummy.local_id()).part_index, TASK_PART_INDEX_NONE);
    EXPECT_EQ(tasks.get_task_by_task_id(a.local_id()).part_index, 0);
    EXPECT_EQ(tasks.get_task_by_task_id(root.local_id()).part_index, 1);
    EXPECT_EQ(tasks.get_task_by_task_id(b.local_id()).part_index, 2);
    EXPECT_EQ(tasks.get_task_by_task_id(c.local_id()).part_index, 3);
}

// A producer split across the groups stands, in the group of a consumer run whole,
// as one proxy just below the consumer, mapped from the producer's global id and
// waiting for all four of its pieces; every piece announces that id.
TEST_F(HbgSiloStampTest, PlanProxiesAProducerRunElsewhere) {
    orch.declare_partitions(100);
    const TaskId wide = submit(TASK_PART_SPREAD_MIN_BLOCKS);
    const TaskId narrow = submit(1, &wide, 1);
    orch.finish_partitions();
    ASSERT_EQ(orch.partitions(), 1);
    const uint32_t home = group_of(narrow);
    const part_plan::SubPartition &sp = sub(0, home);
    ASSERT_EQ(sp.map, 1U);
    const auto &m = *part_plan::at<part_plan::MapEntry>(plan(), sp.map_offset);
    EXPECT_EQ(m.expected, part_plan::kGroups);
    const part_plan::Item *consumer = item_of(sp, narrow);
    ASSERT_NE(consumer, nullptr);
    ASSERT_EQ(consumer->named, 1U);
    EXPECT_EQ(consumer->ranges[0].lo, m.local);
    EXPECT_EQ(consumer->ranges[0].hi, m.local + 1);
    EXPECT_EQ(m.local + 1, consumer->local);
    for (uint32_t g = 0; g < part_plan::kGroups; ++g) {
        const part_plan::SubPartition &share = sub(0, g);
        const part_plan::Item *piece = item_of(share, wide);
        ASSERT_NE(piece, nullptr);
        EXPECT_EQ(
            piece->block_hi - piece->block_lo, static_cast<uint32_t>(TASK_PART_SPREAD_MIN_BLOCKS) / part_plan::kGroups
        );
        EXPECT_EQ(piece->gid, m.gid);
        EXPECT_NE(piece->flags & part_plan::kFlagAnnounce, 0U);
        if (g != home) EXPECT_EQ(share.map, 0U);
    }
}

// With partitions, a supertask is split across the silos from
// TASK_PART_SPREAD_MIN_BLOCKS blocks; a narrower one runs whole in one silo.
TEST_F(HbgSiloStampTest, PartitionsSpreadFromTheirOwnThreshold) {
    orch.declare_partitions(100);
    const TaskId narrow = submit(TASK_PART_SPREAD_MIN_BLOCKS - 1);
    const TaskId wide = submit(TASK_PART_SPREAD_MIN_BLOCKS);
    EXPECT_GE(silo_of(narrow), TASK_SILO_FIRST);
    EXPECT_EQ(silo_of(wide), TASK_SILO_SPREAD);
}

// The task a predicate's value waits on is the last one to write the element it
// reads -- here the writer of the half holding it, not the later writer of the other
// half, and not one of the predicated task's own producers.
TEST_F(HbgSiloStampTest, PredicateNamesTheLastWriterOfItsElement) {
    MixedKernels mixed{};
    mixed.aiv0_kernel_id = 0;
    uint32_t shape[] = {16};
    CoreTaskArgs make;
    create_infos.emplace_back(shape, 1, DataType::INT32);
    make.add_output(create_infos.back());
    TaskOutputTensors made = orch.submit_task(mixed, make);
    const simpler::hbg::Tensor &count = made.get_ref(0);

    uint32_t half[] = {8};
    uint32_t lo_at[] = {0};
    uint32_t hi_at[] = {8};
    const simpler::hbg::Tensor lo = count.view(half, lo_at);
    const simpler::hbg::Tensor hi = count.view(half, hi_at);
    CoreTaskArgs write_lo;
    write_lo.add_inout(lo);
    const TaskId lo_writer = orch.submit_task(mixed, write_lo).task_id();
    CoreTaskArgs write_hi;
    write_hi.add_inout(hi);
    orch.submit_task(mixed, write_hi);

    CoreTaskPredicate pred;
    pred.operand.tensor = &count;
    pred.operand.ndims = 1;
    pred.operand.indices[0] = 3;
    pred.op = PredicateOp::GT;
    pred.target = 0;
    CoreTaskArgs gated;
    create_infos.emplace_back(shape, 1, DataType::FLOAT32);
    gated.add_output(create_infos.back());
    gated.set_predicate(pred);
    const TaskId p = orch.submit_task(mixed, gated).task_id();
    ASSERT_TRUE(p.is_valid());
    const TaskId plain = submit(1);

    SharedMemoryTaskHeader &tasks = sm_handle->header->tasks;
    EXPECT_EQ(tasks.get_task_by_task_id(p.local_id()).predicate_writer, lo_writer.local_id());
    EXPECT_EQ(tasks.get_task_by_task_id(plain.local_id()).predicate_writer, -1);
}

// Predicated tasks run whole in one group form a chain when they share a predicate
// and depend on each other: the chain is decided once, at its first task's home
// slot, and if it runs, it goes whole to one group, its tasks naming each other
// relative to its start and its producers outside it listed for the manager.
TEST_F(HbgSiloStampTest, PlanChainsConditionalTasksThatShareAPredicate) {
    orch.declare_partitions(100);
    MixedKernels mixed{};
    mixed.aiv0_kernel_id = 0;
    uint32_t shape[] = {16};
    CoreTaskArgs make;
    create_infos.emplace_back(shape, 1, DataType::INT32);
    make.add_output(create_infos.back());
    TaskOutputTensors made = orch.submit_task(mixed, make);
    const TaskId writer = made.task_id();
    const simpler::hbg::Tensor &count = made.get_ref(0);
    CoreTaskPredicate pred;
    pred.operand.tensor = &count;
    pred.operand.ndims = 1;
    pred.operand.indices[0] = 0;
    pred.op = PredicateOp::GT;
    pred.target = 0;
    auto gated = [&](const TaskId *deps, uint32_t n) {
        CoreTaskArgs args;
        create_infos.emplace_back(shape, 1, DataType::FLOAT32);
        args.add_output(create_infos.back());
        args.set_predicate(pred);
        if (n > 0) args.set_dependencies(deps, n);
        return orch.submit_task(mixed, args).task_id();
    };
    const TaskId first = gated(&writer, 1);
    const TaskId second = gated(&first, 1);
    ASSERT_TRUE(second.is_valid());
    orch.finish_partitions();
    ASSERT_EQ(orch.partitions(), 1);
    const auto &h = *part_plan::at<part_plan::Header>(plan(), 0);
    ASSERT_EQ(h.chains, 1U);
    const auto &c = *part_plan::at<part_plan::Chain>(plan(), h.chain_offset);
    EXPECT_EQ(c.tasks, 2U);
    EXPECT_EQ(c.ids, 2U);
    EXPECT_EQ(c.owner, group_of(first));
    ASSERT_EQ(c.external, 1U);
    EXPECT_EQ(*reinterpret_cast<const int32_t *>(plan() + c.ext_offset), writer.local_id());
    const auto &later =
        *part_plan::at<part_plan::ChainTask>(plan(), c.task_offset + part_plan::words_of<part_plan::ChainTask>());
    EXPECT_EQ(later.task, second.local_id());
    ASSERT_EQ(later.named, 1U);
    EXPECT_EQ(later.ranges[0].lo, 0U);
    EXPECT_EQ(later.ranges[0].hi, 1U);
    const part_plan::SubPartition &sp = sub(0, group_of(first));
    const part_plan::Item *slot = item_of(sp, first);
    ASSERT_NE(slot, nullptr);
    EXPECT_EQ(slot->kind, static_cast<uint32_t>(part_plan::kSlot));
    EXPECT_NE(slot->flags & part_plan::kFlagDecides, 0U);
    EXPECT_EQ(sp.dyn_count, c.ids);
}
