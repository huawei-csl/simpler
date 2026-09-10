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
 * @file tracr_aicore_emit.h
 * @brief AICore-side TraCR marker emission into a GM buffer.
 *
 * A TraCR `.bts` trace file is a flat array of 16-byte Payloads and carries no
 * header, so nothing downstream needs to know who produced the bytes: the
 * AICore writes the records and the host serializes them as a lane. That is why
 * no part of TraCR itself is compiled for the core — this header defines the
 * whole device-side contract.
 *
 * Payload wire layout (little-endian, matching TraCR::Payload exactly):
 *
 *     bytes 0-1   channelId : uint16
 *     bytes 2-3   eventId   : uint16
 *     bytes 4-7   extraId   : uint32
 *     bytes 8-15  timestamp : uint64, nanoseconds
 *
 * written as two int64 words so the kernel needs no struct definition.
 *
 * Buffer layout:
 *
 *     word 0      count    - records written
 *     word 1      dropped  - records refused because the buffer was full
 *     word 2..    payloads, two words each
 *
 * The count is explicit rather than implied by a zero terminator: the buffer is
 * carried by an OUTPUT_EXISTING tensor, so a host-side zero fill is never staged
 * to the device and untouched words hold whatever GM held before.
 *
 * `tracr_aicore_reset` must run before the first emit on a buffer, and zeroes
 * only the header — untouched payload words are never read, because `count`
 * bounds them.
 *
 * Overflow drops the record and counts it. It never wraps: a wrapped ring is
 * rotated rather than time-ordered, and tracr_process requires every `.bts` to
 * be sorted by timestamp and never sorts one itself, so a wrap would silently
 * produce mis-ordered output and corrupt sync anchors.
 *
 * Every entry point is defined unconditionally but compiles to nothing without
 * `-DENABLE_TRACR`, so a kernel may call these unguarded and a production build
 * pays neither a GM store nor a clock read. This mirrors the contract
 * orchestration codegen already relies on for TraCR's own macros.
 *
 * No cache maintenance here. The buffer has a single reader, on the host, after
 * the kernel has completed, so ordering comes from the task-completion path
 * that already precedes the copy back. A design where the AICPU drains buffers
 * while a core keeps writing does need per-record `dcci`+`dsb`, as
 * chip_swimlane_aicore_commit_task_record issues.
 */

#pragma once

#include <cstdint>

#include "aicore/aicore.h"
#include "inner_kernel.h"

#include "aicore/tracr_aicore_layout.h"

/** Payload capacity of a buffer of `words` int64 words. */
__aicore__ __attribute__((always_inline)) inline int tracr_aicore_capacity(int words) {
    return (words - kTracrHeaderWords) / kTracrWordsPerPayload;
}

/** Open a buffer for writing. Required before the first emit. No-op for a
 *  worker that is not the designated writer. */
__aicore__ __attribute__((always_inline)) inline void tracr_aicore_reset(__gm__ int64_t *buf, int capacity) {
#ifndef ENABLE_TRACR
    (void)buf;
    (void)capacity;
#else
    if (capacity < 0) {
        return;
    }
    buf[0] = 0;
    buf[1] = 0;
#endif
}

/**
 * Append one marker. `channel` is a placeholder the host overwrites: the index
 * of a lane in `channel_names` depends on the run's core count, so it is
 * resolved by name at serialization time rather than assumed here.
 */
__aicore__ __attribute__((always_inline)) inline void tracr_aicore_emit(
    __gm__ int64_t *buf, int capacity, uint32_t channel, uint32_t event, uint32_t extra, uint64_t ticks
) {
#ifndef ENABLE_TRACR
    (void)buf;
    (void)capacity;
    (void)channel;
    (void)event;
    (void)extra;
    (void)ticks;
#else
    if (capacity < 0) {
        return;
    }
    int64_t n = buf[0];
    if (n >= static_cast<int64_t>(capacity)) {
        buf[1] = buf[1] + 1;
        return;
    }
    uint64_t word0 = (channel & 0xFFFFull) | ((event & 0xFFFFull) << 16) | (static_cast<uint64_t>(extra) << 32);
    int base = kTracrHeaderWords + static_cast<int>(n) * kTracrWordsPerPayload;
    buf[base] = static_cast<int64_t>(word0);
    buf[base + 1] = static_cast<int64_t>(ticks * static_cast<uint64_t>(kTracrNsPerTick));
    buf[0] = n + 1;
#endif
}

/** Open a span on `channel` with marker type `event`. */
__aicore__ __attribute__((always_inline)) inline void tracr_aicore_mark_set(
    __gm__ int64_t *buf, int capacity, uint32_t channel, uint32_t event, uint32_t extra
) {
    tracr_aicore_emit(buf, capacity, channel, event, extra, get_sys_cnt_aicore());
}

/** Close the span currently open on `channel`. */
__aicore__ __attribute__((always_inline)) inline void tracr_aicore_mark_reset(
    __gm__ int64_t *buf, int capacity, uint32_t channel
) {
    tracr_aicore_emit(buf, capacity, channel, kTracrEventReset, kTracrExtraNone, get_sys_cnt_aicore());
}

/**
 * Flow id for one message, packed so both endpoints derive it independently:
 * the sender calls (me, peer, seq) and the receiver (peer, me, seq) -- same
 * function, mirrored arguments, no handshake.
 *
 * A flow id must be unique per message and shared by both endpoints.
 * tracr_process pairs starts to ends *by index within one id*, so a reused id
 * does not warn: it mispairs silently, and one dropped record shifts every
 * later arrow on that id. `seq` is what keeps ids unique across rounds, and it
 * must be a value both sides already know -- a round or invocation index
 * qualifies, anything data-dependent does not.
 *
 * Ranks are global, not device ids: device ids are node-local, so two nodes
 * would collide. 8 bits each leaves 16 bits of sequence per ordered pair.
 */
__aicore__ __attribute__((always_inline)) inline uint32_t tracr_aicore_flow_id(
    uint32_t src_rank, uint32_t dst_rank, uint32_t seq
) {
    return ((src_rank & 0xFFu) << 24) | ((dst_rank & 0xFFu) << 16) | (seq & 0xFFFFu);
}

/** Tail of a causal arrow, on the sending side. */
__aicore__ __attribute__((always_inline)) inline void tracr_aicore_flow_start(
    __gm__ int64_t *buf, int capacity, uint32_t channel, uint32_t flow_id
) {
    tracr_aicore_emit(buf, capacity, channel, kTracrEventFlowStart, flow_id, get_sys_cnt_aicore());
}

/** Head of a causal arrow, on the receiving side. */
__aicore__ __attribute__((always_inline)) inline void tracr_aicore_flow_end(
    __gm__ int64_t *buf, int capacity, uint32_t channel, uint32_t flow_id
) {
    tracr_aicore_emit(buf, capacity, channel, kTracrEventFlowEnd, flow_id, get_sys_cnt_aicore());
}
