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

#include "scheduler.h"
#include "spin_hint.h"

namespace simpler::hbg {

struct AsyncPublicationResult {
    AsyncPollResult progress{};
    bool published{false};
};

// Every FIN handler is also a possible consumer. A full mailbox must permit
// that handler to drain under the wait-list lock before it retries publication.
template <typename TryPublish, typename KeepRunning>
AsyncPublicationResult publish_async_with_progress(
    SchedulerState &sched, AICoreCompletionMailbox &mailbox, [[maybe_unused]] int32_t thread_idx,
    TryPublish &&try_publish, KeepRunning &&keep_running
) {
    AsyncPublicationResult result;
    while (keep_running()) {
        if (try_publish()) {
            result.published = true;
            return result;
        }
        sched.async_wait_list.mpsc_skipped_count.fetch_add(1, std::memory_order_relaxed);
        AsyncPollResult progress = sched.async_wait_list.poll_and_complete<false>(
            &mailbox, &sched
#if SIMPLER_SCHED_PROFILING
            ,
            thread_idx
#endif
        );
        result.progress.completed += progress.completed;
        result.progress.resolved += progress.resolved;
        if (progress.error_code != SIMPLER_ERROR_NONE) {
            result.progress.error_code = progress.error_code;
            result.progress.failed_slot_state = progress.failed_slot_state;
            return result;
        }
        SPIN_WAIT_HINT();
    }
    return result;
}

}  // namespace simpler::hbg
