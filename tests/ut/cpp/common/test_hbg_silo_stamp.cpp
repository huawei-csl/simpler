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
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "utils/device_arena.h"
#include "scheduler/scheduler.h"
#include "host_build_graph/orchestrator.h"
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
// partition and each partition's first task with how many of its tasks complete on
// the device.
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
    EXPECT_EQ(tasks.get_task_by_task_id(0).part_tasks, 4);
    EXPECT_EQ(tasks.get_task_by_task_id(4).part_tasks, 4);
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
    EXPECT_EQ(tasks.get_task_by_task_id(0).part_tasks, 5);
    EXPECT_EQ(tasks.get_task_by_task_id(5).part_tasks, 3);
}
