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
 * @file tracr_aicore_layout.h
 * @brief The AICore TraCR wire contract: record layout, sentinels, sizes.
 *
 * Split out of `tracr_aicore_emit.h` because **the host needs these too** and
 * that header cannot be included host-side: it pulls in `aicore/aicore.h` and
 * `inner_kernel.h` for the device intrinsics, neither of which exists in a host
 * translation unit.
 *
 * So: constants shared by whoever writes the records and whoever decodes them
 * live here and depend on nothing; the emit functions stay next door.
 */

#pragma once

#include <cstdint>

// TraCR's EVENTID_RESET: closes the span currently open on the channel.
constexpr uint32_t kTracrEventReset = 0xFFFFu;
// TraCR's EVENTID_FLOW_START / EVENTID_FLOW_END: the two endpoints of a causal
// arrow. Each attaches to whatever span is open on its channel at that instant,
// and carries the flow id in extraId rather than an event type.
constexpr uint32_t kTracrEventFlowStart = 0xFFFEu;
constexpr uint32_t kTracrEventFlowEnd = 0xFFFDu;
// TraCR's "no extra information" sentinel (UINT32_MAX).
constexpr uint32_t kTracrExtraNone = 0xFFFFFFFFu;
// get_sys_cnt_aicore() counts at PLATFORM_PROF_SYS_CNT_FREQ (50 MHz); TraCR
// timestamps are nanoseconds.
constexpr int64_t kTracrNsPerTick = 20;

constexpr int kTracrHeaderWords = 2;
constexpr int kTracrWordsPerPayload = 2;

// Capacity value that silently disables recording, without counting a drop.
//
// One buffer has one writer. When several workers run the same kernel source
// against a shared buffer -- SPMD blocks, or a vector kernel split across a
// block's two AIV sub-cores -- each of them would read-modify-write the same
// count word with no atomic, losing records and corrupting the count. The
// designated writer passes a real capacity and every other worker passes this,
// so the guard costs one comparison and no buffer traffic.
//
// Marker filtering is not a loss for communication: an SPMD collective must
// already confine its notify/wait to one block, or the peer's counter is
// incremented once per block and the barrier releases early. For an SPMD
// *compute* kernel the per-block variation is the signal, and that wants a
// per-core buffer instead -- see D2 in the project docs.
constexpr int kTracrDisabled = -1;

/**
 * int64 words the host reserves per AICore, and the record capacity that
 * implies once the `[count][dropped]` header is taken out.
 *
 * Per **core**, not per chip, and that is the point: the host allocates
 * `PLATFORM_MAX_CORES * kTracrAicoreWordsPerCore` and each core writes only
 * `base + block_idx * kTracrAicoreWordsPerCore`. Cores never share a buffer, so
 * the count word needs no atomics and concurrent kernels cannot corrupt each
 * other's records -- a property of the addressing, not of how many cores happen
 * to be recording today.
 *
 * 512 words is 4 KB per core, so a full 72-core chip reserves 288 KB. Overflow
 * drops and counts rather than wrapping (see `tracr_aicore_emit`), so the cost
 * of this being small is losing a long run's tail, against every record being a
 * GM store on the critical path of whatever is being measured.
 */
constexpr int kTracrAicoreWordsPerCore = 512;
