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

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <memory>
#include <thread>

#include "host_build_graph/shared_memory.h"
#include "scheduler/scheduler_async_progress.h"
#include "utils/device_arena.h"

namespace {

class HbgAsyncProgress : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    SharedMemoryHandle *sm_handle{nullptr};
    SchedulerState sched{};
    std::unique_ptr<AICoreCompletionMailbox> mailbox;
    std::unique_ptr<ChipTaskStorage[]> tasks;
    uint32_t ready_counter{1};

    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        const SchedulerLayout layout = SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);
        ASSERT_TRUE(sched.init_data_from_layout(layout, runtime_arena, sm_handle->sm_base));
        sched.wire_arena_pointers(layout, runtime_arena);
        sched.seed_queue_slots();
        mailbox = std::make_unique<AICoreCompletionMailbox>();
        mailbox->init_empty();
        tasks = std::make_unique<ChipTaskStorage[]>(MAX_ASYNC_WAITS);
        for (int32_t task = 0; task < MAX_ASYNC_WAITS; ++task) {
            tasks[task].task.task_id = TaskId::make_global(task);
            tasks[task].slot.task_kind = TaskKind::KERNEL;
            tasks[task].slot.active_mask = LaneMask::aiv0();
            tasks[task].slot.total_required_subtasks = 1;
            tasks[task].slot.logical_block_num = 1;
        }
    }

    void TearDown() override {
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }

    bool push_condition(int32_t task, uint64_t address) {
        return mailbox->try_push_condition(
            tasks[task].task.task_id, address, /*backend_cookie=*/0, /*expected_value=*/1, COMPLETION_ENGINE_ROCE,
            COMPLETION_TYPE_COUNTER
        );
    }

    bool push_condition(int32_t task) { return push_condition(task, reinterpret_cast<uint64_t>(&ready_counter)); }

    bool push_done(int32_t task) {
        return mailbox->try_push_normal_done(tasks[task].task.task_id, reinterpret_cast<uint64_t>(&tasks[task].slot));
    }

    AsyncPollResult poll(int32_t thread_idx = 0) {
        return sched.async_wait_list.poll_and_complete<false>(
            mailbox.get(), &sched
#if SIMPLER_SCHED_PROFILING
            ,
            thread_idx
#endif
        );
    }

    void fill_deferred_wave() {
        // 59 complete task records, four condition-only records, and five
        // conditions for the final task occupy exactly the 4096-message ring.
        static_assert(
            59 * (MAX_COMPLETIONS_PER_TASK + 1) + 4 * MAX_COMPLETIONS_PER_TASK + 5 == AICORE_COMPLETION_MAILBOX_CAPACITY
        );
        static_assert(MAX_ASYNC_WAITS == 64);
        for (int32_t task = 0; task < 63; ++task) {
            for (int32_t condition = 0; condition < MAX_COMPLETIONS_PER_TASK; ++condition) {
                ASSERT_TRUE(push_condition(task));
            }
            if (task < 59) ASSERT_TRUE(push_done(task));
        }
        for (int32_t condition = 0; condition < 5; ++condition)
            ASSERT_TRUE(push_condition(63));
        ASSERT_EQ(mailbox->head.load() - mailbox->tail.load(), AICORE_COMPLETION_MAILBOX_CAPACITY);
    }
};

TEST_F(HbgAsyncProgress, FullMailboxPublicationDrainsWithoutADedicatedConsumer) {
    fill_deferred_wave();
    ASSERT_FALSE(push_condition(63));
    int attempts = 0;
    int retry_budget = 3;
    const auto assisted = simpler::hbg::publish_async_with_progress(
        sched, *mailbox, 0,
        [&] {
            ++attempts;
            return push_condition(63);
        },
        [&] {
            return retry_budget-- > 0;
        }
    );

    ASSERT_TRUE(assisted.published);
    EXPECT_EQ(attempts, 2);
    EXPECT_EQ(assisted.progress.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(assisted.progress.completed, 59);
    EXPECT_EQ(assisted.progress.resolved, 59);
    EXPECT_EQ(sched.async_wait_list.count, 5);
    EXPECT_EQ(sched.async_wait_list.mpsc_skipped_count.load(), 1u);
    for (int32_t task = 0; task < MAX_ASYNC_WAITS; ++task) {
        EXPECT_EQ(sm_handle->header->tasks.is_completed(task), task < 59);
    }

    for (int32_t condition = 6; condition < MAX_COMPLETIONS_PER_TASK; ++condition)
        ASSERT_TRUE(push_condition(63));
    for (int32_t task = 59; task < MAX_ASYNC_WAITS; ++task)
        ASSERT_TRUE(push_done(task));
    const auto tail = poll();
    EXPECT_EQ(tail.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(tail.completed, 5);
    EXPECT_EQ(tail.resolved, 5);
    EXPECT_EQ(assisted.progress.completed + tail.completed, MAX_ASYNC_WAITS);
    EXPECT_EQ(sched.async_wait_list.count, 0);
    EXPECT_FALSE(mailbox->has_pending());
    for (int32_t task = 0; task < MAX_ASYNC_WAITS; ++task)
        EXPECT_TRUE(sm_handle->header->tasks.is_completed(task));
    const auto empty = poll();
    EXPECT_EQ(empty.completed, 0);
    EXPECT_EQ(empty.resolved, 0);
}

TEST_F(HbgAsyncProgress, FailedAssistanceDoesNotPublishOrRetry) {
    fill_deferred_wave();
    // The final task's conditions are visited first by the reverse wait-list scan.
    mailbox->entries[AICORE_COMPLETION_MAILBOX_CAPACITY - 1].addr = 0;
    int attempts = 0;
    const auto assisted = simpler::hbg::publish_async_with_progress(
        sched, *mailbox, 0,
        [&] {
            ++attempts;
            return push_condition(63);
        },
        [] {
            return true;
        }
    );

    EXPECT_FALSE(assisted.published);
    EXPECT_EQ(attempts, 1);
    EXPECT_EQ(assisted.progress.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(assisted.progress.completed, 0);
    EXPECT_EQ(assisted.progress.resolved, 0);
    EXPECT_EQ(sched.async_wait_list.busy.load(), 0);
    EXPECT_FALSE(mailbox->has_pending());
    for (int32_t task = 0; task < MAX_ASYNC_WAITS; ++task)
        EXPECT_FALSE(sm_handle->header->tasks.is_completed(task));
}

TEST_F(HbgAsyncProgress, CompletedTaskIsStillCountedWhenLaterRegistrationFails) {
    ASSERT_TRUE(push_done(0));
    for (int32_t condition = 0; condition <= MAX_COMPLETIONS_PER_TASK; ++condition)
        ASSERT_TRUE(push_condition(1));

    const auto result = poll();
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
    EXPECT_EQ(result.completed, 1);
    EXPECT_EQ(result.resolved, 1);
    EXPECT_TRUE(sm_handle->header->tasks.is_completed(0));
    EXPECT_FALSE(sm_handle->header->tasks.is_completed(1));
    EXPECT_EQ(sched.async_wait_list.busy.load(), 0);
}

TEST_F(HbgAsyncProgress, StoppedRunDoesNotAttemptPublication) {
    int attempts = 0;
    const auto stopped = simpler::hbg::publish_async_with_progress(
        sched, *mailbox, 0,
        [&] {
            ++attempts;
            return push_condition(0);
        },
        [] {
            return false;
        }
    );

    EXPECT_FALSE(stopped.published);
    EXPECT_EQ(attempts, 0);
    EXPECT_EQ(stopped.progress.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(stopped.progress.completed, 0);
    EXPECT_FALSE(mailbox->has_pending());
}

TEST_F(HbgAsyncProgress, BusyConsumerLeavesMessagesForTheLockOwner) {
    ASSERT_TRUE(push_condition(0));
    ASSERT_TRUE(push_done(0));
    ASSERT_TRUE(sched.async_wait_list.try_lock());

    AsyncPollResult rejected{};
    std::thread contender([&] {
        rejected = poll(1);
    });
    contender.join();
    EXPECT_EQ(rejected.completed, 0);
    EXPECT_EQ(rejected.resolved, 0);
    EXPECT_EQ(rejected.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(mailbox->tail.load(), 0u);
    EXPECT_EQ(sched.async_wait_list.count, 0);
    sched.async_wait_list.unlock();

    const auto owner = poll();
    EXPECT_EQ(owner.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(owner.completed, 1);
    EXPECT_EQ(owner.resolved, 1);
    EXPECT_TRUE(sm_handle->header->tasks.is_completed(0));
    EXPECT_EQ(poll().completed, 0);
}

TEST_F(HbgAsyncProgress, GraphChildrenCountOnlyTheirFinalOuterStreamCompletion) {
    auto children = std::make_unique<ChipTaskStorage[]>(2);
    std::array<std::atomic<ChipTaskState>, 2> states{};
    GraphExecution execution{};
    execution.definition_offset = static_cast<uint32_t>(GRAPH_DEFINITION_OBJECT_ALIGN);
    execution.task_storage = children.get();
    execution.task_states = states.data();
    execution.task_count = 2;
    execution.remaining_tasks.store(2);
    execution.outer_slot = &tasks[0].slot;
    graph_execution_set_phase(execution, GraphExecutionPhase::MATERIALIZED);
    tasks[0].slot.task_kind = TaskKind::GRAPH;
    tasks[0].slot.graph_context = &execution;
    for (int32_t child = 0; child < 2; ++child) {
        states[child].store(CHIP_TASK_PENDING);
        children[child].task.task_id = TaskId::make_global(child + 1);
        children[child].slot.task_kind = TaskKind::KERNEL;
        children[child].slot.graph_context = &execution;
        children[child].slot.sub_task_local_id = child;
    }

    ASSERT_TRUE(mailbox->try_push_normal_done(children[0].task.task_id, reinterpret_cast<uint64_t>(&children[0].slot)));
    const auto first = poll();
    EXPECT_EQ(first.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(first.completed, 0);
    EXPECT_EQ(first.resolved, 1);
    EXPECT_EQ(execution.remaining_tasks.load(), 1);
    EXPECT_FALSE(sm_handle->header->tasks.is_completed(0));

    ASSERT_TRUE(mailbox->try_push_normal_done(children[1].task.task_id, reinterpret_cast<uint64_t>(&children[1].slot)));
    const auto last = poll();
    EXPECT_EQ(last.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(last.completed, 1);
    EXPECT_EQ(last.resolved, 1);
    EXPECT_EQ(execution.remaining_tasks.load(), 0);
    EXPECT_EQ(execution.retired_tasks.load(), 2);
    EXPECT_EQ(graph_execution_phase(execution), GraphExecutionPhase::COMPLETED);
    EXPECT_TRUE(sm_handle->header->tasks.is_completed(0));
    EXPECT_EQ(states[0].load(), CHIP_TASK_COMPLETED);
    EXPECT_EQ(states[1].load(), CHIP_TASK_COMPLETED);
    EXPECT_EQ(poll().completed, 0);
}

}  // namespace
