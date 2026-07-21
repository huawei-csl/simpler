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

#include <atomic>

#include "common/core_type.h"
#include "utils/device_arena.h"
#include "pto_async_wait.h"
#include "pto_ring_buffer.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"

// Forward declaration so this header can compile under both AICPU and host
// builds. The actual definition is provided by aicpu/device_time.cpp (AICPU)
// or a weak stub in pto_runtime2.h (host). Used only for sub-phase profiling.
uint64_t get_sys_cnt_aicpu();

// A ready-queue slot is a single atomic cell (Egorushkin atomic_queue design):
// the element pointer itself is the synchronization variable. An empty cell
// holds nullptr; a producer CAS-writes the pointer, a consumer exchange-reads
// it back to nullptr. No separate sequence field is needed.
struct PTO2ReadyQueueSlot {
    std::atomic<PTO2TaskSlotState *> cell;
};

// Number of CoreType values eligible for local dispatch (AIC=0, AIV=1)
static constexpr int PTO2_LOCAL_DISPATCH_TYPE_NUM = 2;

struct PTO2LocalReadyBuffer {
    PTO2TaskSlotState **slot_states = nullptr;
    int count = 0;
    int capacity = 0;

    void reset(PTO2TaskSlotState **buf, int cap) {
        slot_states = buf;
        count = 0;
        capacity = cap;
    }

    bool try_push(PTO2TaskSlotState *s) {
        if (slot_states && count < capacity) {
            slot_states[count++] = s;
            return true;
        }
        return false;
    }

    PTO2TaskSlotState *pop() { return (count > 0) ? slot_states[--count] : nullptr; }
};

// Lock-free MPMC ready queue using Maxim Egorushkin's atomic_queue design
// (as packaged in huawei-csl/queues MPMC_AtomicRingBuffer): each slot's element
// pointer is itself the sync variable, and consecutive logical positions are
// reindexed onto physically distant slots (stride reindex_mul, coprime to
// capacity) so concurrent producers/consumers rarely touch the same cache line.
// The push/pop cursors are packed two-per-atomic with a cached view of the
// opposite cursor: head_cached_tail = {head:hi32, cachedTail:lo32},
// tail_cached_head = {tail:hi32, cachedHead:lo32}. capacity/mask/reindex_mul
// are runtime values (arena-backed storage), not template constants.
struct alignas(64) PTO2ReadyQueue {
    PTO2ReadyQueueSlot *slots;
    uint64_t capacity;
    uint64_t mask;         // capacity - 1
    uint32_t reindex_mul;  // stride mapping logical position -> physical slot
    char _pad0[64 - 28];   // Pad to own cache line

    alignas(64) std::atomic<uint64_t> tail_cached_head;  // {tail:hi32, cachedHead:lo32}
    char _pad1[64 - sizeof(std::atomic<uint64_t>)];

    alignas(64) std::atomic<uint64_t> head_cached_tail;  // {head:hi32, cachedTail:lo32}
    char _pad2[64 - sizeof(std::atomic<uint64_t>)];

    static constexpr uint32_t hi32(uint64_t v) { return static_cast<uint32_t>(v >> 32u); }
    static constexpr uint32_t lo32(uint64_t v) { return static_cast<uint32_t>(v); }
    static constexpr uint64_t pack(uint32_t hi, uint32_t lo) {
        return (static_cast<uint64_t>(hi) << 32u) | static_cast<uint64_t>(lo);
    }
    // Stride coprime to a power-of-two capacity: smallest odd >= slots-per-line.
    static uint32_t calc_reindex_mul(uint64_t cap) {
        constexpr uint32_t line_bytes = 64u;
        uint32_t smallest =
            static_cast<uint32_t>((line_bytes + sizeof(PTO2TaskSlotState *) - 1) / sizeof(PTO2TaskSlotState *));
        if ((smallest & 1u) == 0u) ++smallest;
        if (smallest >= cap) smallest = 1u;
        return smallest;
    }
    // Overflow-safe because capacity divides 2^32: (pos*mul) mod capacity is
    // preserved through the 32-bit wrap.
    uint32_t phys_index(uint32_t position) const {
        return (position * reindex_mul) & static_cast<uint32_t>(mask);
    }

    uint64_t size() {
        const uint32_t head = hi32(head_cached_tail.load(std::memory_order_relaxed));
        const uint32_t tail = hi32(tail_cached_head.load(std::memory_order_relaxed));
        return static_cast<uint32_t>(head - tail);  // unsigned modular occupancy
    }

    // No-op: positions increase monotonically and every slot is exchanged back
    // to nullptr by its last pop, so run N+1 resumes from run N's cursors with
    // every slot already empty — no per-slot reset needed across arena reuse.
    void reset_for_reuse() {}

    // A reserved-but-not-yet-written slot (producer advanced the cursor, its
    // store not landed) is a transient the consumer briefly spins on; on the
    // AICPU each thread owns a core and completes its store without preemption,
    // so the spin is short and bounded.
    PTO2TaskSlotState *do_pop(uint32_t position) {
        std::atomic<PTO2TaskSlotState *> &e = slots[position].cell;
        PTO2TaskSlotState *val = e.exchange(nullptr, std::memory_order_acquire);
        while (val == nullptr) {
            do {
                SPIN_WAIT_HINT();
            } while (e.load(std::memory_order_relaxed) == nullptr);
            val = e.exchange(nullptr, std::memory_order_acquire);
        }
        return val;
    }

    void do_push(PTO2TaskSlotState *value, uint32_t position) {
        std::atomic<PTO2TaskSlotState *> &e = slots[position].cell;
        PTO2TaskSlotState *empty = nullptr;
        while (!e.compare_exchange_weak(empty, value, std::memory_order_release, std::memory_order_relaxed)) {
            empty = nullptr;
            do {
                SPIN_WAIT_HINT();
            } while (e.load(std::memory_order_relaxed) != nullptr);
        }
    }

    bool push(PTO2TaskSlotState *value) {
        uint64_t hct = head_cached_tail.load(std::memory_order_relaxed);
        uint32_t head;
        uint32_t tail;
        do {
            head = hi32(hct);
            tail = lo32(hct);
            const uint32_t head_trail = head - static_cast<uint32_t>(capacity);
            if (head_trail == tail && head_trail == (tail = hi32(tail_cached_head.load(std::memory_order_relaxed))))
                return false;  // Queue full
        } while (!head_cached_tail.compare_exchange_weak(
            hct, pack(head + 1u, tail), std::memory_order_relaxed, std::memory_order_relaxed
        ));
        do_push(value, phys_index(head));
        return true;
    }

    // Push all `count` items; blocks (spins) only if the queue lacks room, which
    // for the ready queue's large capacity vs. per-donation batch never occurs.
    void push_batch(PTO2TaskSlotState **items, int count) {
        if (count <= 0) return;
        const uint32_t amount = static_cast<uint32_t>(count);

        uint64_t hct = head_cached_tail.load(std::memory_order_relaxed);
        uint32_t head;
        uint32_t tail;
        do {
            head = hi32(hct);
            tail = lo32(hct);
            const uint32_t head_trail = head - static_cast<uint32_t>(capacity);
            uint32_t avail = tail - head_trail;
            if (avail < amount) {
                tail = hi32(tail_cached_head.load(std::memory_order_relaxed));
                avail = tail - head_trail;
                while (avail < amount) {
                    SPIN_WAIT_HINT();
                    tail = hi32(tail_cached_head.load(std::memory_order_relaxed));
                    avail = tail - head_trail;
                }
            }
        } while (!head_cached_tail.compare_exchange_weak(
            hct, pack(head + amount, tail), std::memory_order_relaxed, std::memory_order_relaxed
        ));

        for (uint32_t i = 0; i < amount; i++)
            do_push(items[i], phys_index(head + i));
    }

    PTO2TaskSlotState *pop() {
        uint64_t tch = tail_cached_head.load(std::memory_order_relaxed);
        uint32_t tail;
        uint32_t head;
        do {
            tail = hi32(tch);
            head = lo32(tch);
            if (tail == head && tail == (head = hi32(head_cached_tail.load(std::memory_order_relaxed))))
                return nullptr;  // Queue empty
        } while (!tail_cached_head.compare_exchange_weak(
            tch, pack(tail + 1u, head), std::memory_order_relaxed, std::memory_order_relaxed
        ));
        return do_pop(phys_index(tail));
    }

    // Reserve up to max_count contiguous positions with one CAS, then drain
    // them. Returns actual number popped (may be less than max_count).
    int pop_batch(PTO2TaskSlotState **out, int max_count) {
        const uint32_t want = static_cast<uint32_t>(max_count);
        uint64_t tch = tail_cached_head.load(std::memory_order_relaxed);
        uint32_t tail;
        uint32_t head;
        uint32_t to_pop;
        do {
            tail = hi32(tch);
            head = lo32(tch);
            to_pop = head - tail;
            if (to_pop < want) {
                head = hi32(head_cached_tail.load(std::memory_order_relaxed));
                to_pop = head - tail;
                if (to_pop == 0u) return 0;
            }
            to_pop = std::min(to_pop, want);
        } while (!tail_cached_head.compare_exchange_weak(
            tch, pack(tail + to_pop, head), std::memory_order_relaxed, std::memory_order_relaxed
        ));

        for (uint32_t i = 0; i < to_pop; i++)
            out[i] = do_pop(phys_index(tail + i));
        return static_cast<int>(to_pop);
    }
};

inline size_t ready_queue_reserve_layout(DeviceArena &arena, uint64_t capacity) {
    return arena.reserve(capacity * sizeof(PTO2ReadyQueueSlot), PTO2_ALIGN_SIZE);
}
inline bool
ready_queue_init_data_from_layout(PTO2ReadyQueue *queue, DeviceArena &arena, size_t slots_off, uint64_t capacity) {
    // Address the slots region for data writes without storing the pointer in
    // queue->slots — that field is set by ready_queue_wire_arena_pointers.
    auto *slots_arena = static_cast<PTO2ReadyQueueSlot *>(arena.region_ptr(slots_off));
    queue->capacity = capacity;
    queue->mask = capacity - 1;
    queue->reindex_mul = PTO2ReadyQueue::calc_reindex_mul(capacity);
    queue->tail_cached_head.store(0, std::memory_order_relaxed);
    queue->head_cached_tail.store(0, std::memory_order_relaxed);

    for (uint64_t i = 0; i < capacity; i++)
        slots_arena[i].cell.store(nullptr, std::memory_order_relaxed);

    return true;
}
// Stores queue->slots = arena.region_ptr(slots_off). Idempotent.
inline void ready_queue_wire_arena_pointers(PTO2ReadyQueue *queue, DeviceArena &arena, size_t slots_off) {
    queue->slots = static_cast<PTO2ReadyQueueSlot *>(arena.region_ptr(slots_off));
}
inline void ready_queue_destroy(PTO2ReadyQueue *queue) {
    // Arena owns the slots[] buffer; just forget the pointer.
    queue->slots = nullptr;
}

struct alignas(64) PTO2SpscQueue {
    // --- Producer cache lines (orchestrator thread) ---
    alignas(64) std::atomic<uint64_t> head_{0};
    alignas(64) uint64_t tail_cached_{0};

    // --- Consumer cache lines (scheduler thread 0) ---
    alignas(64) std::atomic<uint64_t> tail_{0};
    alignas(64) uint64_t head_cached_{0};

    // --- Shared Cacheline (read only) with mask and data ptr (immutable after init) ---
    alignas(64) PTO2TaskSlotState **buffer_{nullptr};
    uint64_t mask_{0};

    // Padding to exactly 5 cache lines
    char padding[64 - sizeof(PTO2TaskSlotState **) - sizeof(uint64_t)];

    static size_t reserve_layout(DeviceArena &arena, uint64_t capacity) {
        return arena.reserve(capacity * sizeof(PTO2TaskSlotState *), PTO2_ALIGN_SIZE);
    }

    bool init_data_from_layout(DeviceArena &arena, size_t buffer_off, uint64_t capacity) {
        if (capacity == 0 || (capacity & (capacity - 1)) != 0) return false;
        auto *buf = static_cast<PTO2TaskSlotState **>(arena.region_ptr(buffer_off));
        // calloc'd-equivalent: zero the slot pointers so spurious early pops
        // observe nullptr.
        for (uint64_t i = 0; i < capacity; i++)
            buf[i] = nullptr;
        mask_ = capacity - 1;
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
        tail_cached_ = 0;
        head_cached_ = 0;
        return true;
    }

    // Wire the arena-internal pointer. Called by both host (with host arena)
    // and AICPU (with device arena attached to the prebuilt image).
    void wire_arena_pointers(DeviceArena &arena, size_t buffer_off) {
        buffer_ = static_cast<PTO2TaskSlotState **>(arena.region_ptr(buffer_off));
    }

    void reset_for_reuse() {
        uint64_t h = head_.load(std::memory_order_relaxed);
        tail_.store(h, std::memory_order_relaxed);
        tail_cached_ = h;
        head_cached_ = h;
    }

    // Arena owns the buffer; here we only forget our pointer.
    void destroy() { buffer_ = nullptr; }

    bool push(PTO2TaskSlotState *item) {
        uint64_t h = head_.load(std::memory_order_relaxed);
        uint64_t next_h = h + 1;
        if (next_h - tail_cached_ > mask_) {
            tail_cached_ = tail_.load(std::memory_order_acquire);
            if (next_h - tail_cached_ > mask_) return false;
        }
        buffer_[h & mask_] = item;
        head_.store(next_h, std::memory_order_release);
        return true;
    }

    // Pop up to max_count items (consumer only). Returns actual count.
    int pop_batch(PTO2TaskSlotState **out, int max_count) {
        uint64_t t = tail_.load(std::memory_order_relaxed);
        uint64_t avail = head_cached_ - t;
        if (avail < static_cast<uint64_t>(max_count)) {
            head_cached_ = head_.load(std::memory_order_acquire);
            avail = head_cached_ - t;
            if (avail == 0) return 0;
        }
        int count = (avail < static_cast<uint64_t>(max_count)) ? static_cast<int>(avail) : max_count;
        for (int i = 0; i < count; i++)
            out[i] = buffer_[(t + i) & mask_];
        tail_.store(t + count, std::memory_order_release);
        return count;
    }

    // Approximate size (used for backoff decisions, not exact).
    uint64_t size() const {
        uint64_t h = head_.load(std::memory_order_acquire);
        uint64_t t = tail_.load(std::memory_order_acquire);
        return h - t;
    }
};

static_assert(sizeof(PTO2SpscQueue) == 5 * 64, "PTO2SpscQueue must be exactly 5 cache lines (320B)");
// =============================================================================

struct CompletionStats {
    int32_t fanout_edges;       // Number of fanout edges traversed (notify consumers)
    int32_t tasks_enqueued;     // Number of consumers that became READY
    int32_t fanin_edges;        // Number of fanin edges traversed (release producers)
    bool mixed_task_completed;  // True only when this callback completed a mixed task
};

struct PTO2SchedulerLayout {
    size_t off_ready_queue_slots[PTO2_NUM_RESOURCE_SHAPES];
    size_t off_dummy_ready_queue_slots;
    size_t off_pending_spsc_buffer;
    uint64_t ready_queue_capacity;
    uint64_t spsc_capacity;
};

struct PTO2SchedulerState {
    // Shared memory access
    PTO2SharedMemoryHeader *sm_header;

    // Per-ring state
    struct alignas(64) RingSchedState {
        PTO2SharedMemoryRingHeader *ring;
        int32_t last_task_alive;
        std::atomic<int32_t> advance_lock;  // multi-thread CAS

        bool init_data_from_layout(void *sm_dev_base, int32_t ring_id) {
            ring = pto2_sm_layout::ring_header_addr(sm_dev_base, ring_id);
            last_task_alive = 0;
            advance_lock.store(0, std::memory_order_relaxed);
            return true;
        }

        void destroy() { ring = nullptr; }

        void sync_to_sm() { ring->fc.last_task_alive.store(last_task_alive, std::memory_order_release); }

        void advance_ring_pointers() {
            const int32_t watermark = ring->completed_watermark.load(std::memory_order_acquire);
            int32_t old_last_task_alive = last_task_alive;

            // Retire any slot at the tail whose last consumer is at or below
            // the global completed watermark — i.e. every consumer of this
            // producer has reached COMPLETED. Implies this slot itself is
            // COMPLETED because the seed value of last_consumer_local_id is
            // the slot's own local_id.
            while (last_task_alive <= watermark) {
                PTO2TaskSlotState &slot_state = ring->get_slot_state_by_task_id(last_task_alive);
                if (watermark < slot_state.last_consumer_local_id) break;
                last_task_alive++;
            }

            for (int32_t id = old_last_task_alive; id < last_task_alive; id++)
                ring->get_slot_state_by_task_id(id).reset_for_reuse();

            sync_to_sm();
        }
    } ring_sched_states[PTO2_MAX_RING_DEPTH];

    // Ready queues remain global (scheduling is ring-agnostic)
    PTO2ReadyQueue ready_queues[PTO2_NUM_RESOURCE_SHAPES];

    // Dependency-only tasks (active_mask is empty, shape == DUMMY). Drained by
    // the dispatch loop and completed inline -- never goes to AICore.
    PTO2ReadyQueue dummy_ready_queue;

    // Thread 0 exclusive: bounded SPSC drain → classify → route. The
    // orchestrator pushes slot_states into the SPSC queue; thread 0 drains
    // a batch per scheduler iter, classifies each task's fanin state, and
    // routes terminally — either to a ready queue (all fanins met) or onto
    // a producer's wake_list (first unmet). No intermediate FIFO: each
    // drained task is classified once, never re-queued. The wake-list-only
    // redesign made classify_fanin_state's decision terminal, so the
    // previously-needed pending FIFO became dead weight on the critical
    // path.
    struct alignas(64) PendingState {
        static constexpr int BACKOFF_LIMIT = 32;
        static constexpr int DRAIN_BATCH = 30;

        // --- Thread 0 exclusive ---
        int backoff_counter{0};
        PTO2TaskSlotState *drain_buf[DRAIN_BATCH];

        // --- SPSC queue: orchestrator (push) ↔ thread 0 (pop) ---
        PTO2SpscQueue queue;

        // --- Orchestrator write, thread 0 read ---
        alignas(64) std::atomic<bool> orch_needs_drain{false};
    } wiring;

    alignas(64) AsyncWaitList async_wait_list;

    void push_ready_routed(PTO2TaskSlotState *slot_state) {
        PTO2ResourceShape shape = slot_state->active_mask.to_shape();
        if (shape == PTO2ResourceShape::DUMMY) dummy_ready_queue.push(slot_state);
        else ready_queues[static_cast<int32_t>(shape)].push(slot_state);
    }

    bool fanin_satisfied(PTO2TaskSlotState *s) const {
        const PTO2TaskPayload &p = *s->payload;
        for (int32_t i = 0; i < p.fanin_count; i++) {
            const auto &prod_ring = *ring_sched_states[p.fanin_ring_ids[i]].ring;
            if (prod_ring.completion_flags[p.fanin_local_ids[i] & prod_ring.task_window_mask].load(
                    std::memory_order_acquire
                ) == 0)
                return false;
        }
        return true;
    }

    // First-unmet classification used by the wiring-queue drain and the
    // wake_list rescan. Returns:
    //   -1: all fanins met (route directly to ready)
    //   ≥0: index of the first unmet fanin (register on its producer's
    //       wake list). Decision is terminal — tasks are never re-queued
    //       for polling; rescans happen lazily on producer completion via
    //       on_mixed_task_complete's wake_list drain.
    int classify_fanin_state(PTO2TaskSlotState *s) const {
        const PTO2TaskPayload &p = *s->payload;
        for (int32_t i = 0; i < p.fanin_count; i++) {
            const auto &prod_ring = *ring_sched_states[p.fanin_ring_ids[i]].ring;
            if (prod_ring.completion_flags[p.fanin_local_ids[i] & prod_ring.task_window_mask].load(
                    std::memory_order_acquire
                ) == 0) {
                return i;
            }
        }
        return -1;
    }

    // (e) Register `consumer` on `producer`'s wake list. If the producer has
    // already completed (head == WAKE_LIST_SENTINEL) we must NOT assume the
    // consumer is ready: classify_fanin_state() short-circuits at the FIRST
    // unmet fanin, so the producer handed to us is only "an" unmet fanin, not
    // necessarily the last one. A producer that completes in the window between
    // that classify and this call would otherwise let us push a consumer whose
    // *later* fanins are still pending — a premature dispatch that lets a
    // dependent task run against a not-yet-produced input (e.g. the
    // paged_attention online-softmax UP chain: UP_bn dispatched before UP_{bn-1}
    // finishes writing the shared accumulator -> concurrent RMW -> wrong query).
    // On the SENTINEL path we re-classify against ALL fanins and only route to
    // ready when every fanin is satisfied; otherwise we re-target the next
    // still-unmet producer and retry. Monotonic completion_flags guarantee
    // termination.
    void register_wake(PTO2TaskSlotState *producer, PTO2TaskSlotState *consumer) {
        while (true) {
            PTO2TaskSlotState *expected = producer->wake_list_head.load(std::memory_order_relaxed);
            while (expected != WAKE_LIST_SENTINEL) {
                consumer->next_in_wake_list = expected;
                if (producer->wake_list_head.compare_exchange_weak(
                        expected, consumer, std::memory_order_acq_rel, std::memory_order_relaxed
                    )) {
                    return;  // registered on a still-pending producer
                }
                // CAS failed: expected reloaded; retry (may now be SENTINEL).
            }
            // Producer completed. Re-check every fanin before committing.
            int32_t state = classify_fanin_state(consumer);
            if (state < 0) {
                push_ready_routed(consumer);  // all fanins now satisfied
                return;
            }
            // Re-target the next still-unmet producer and retry.
            const PTO2TaskPayload &p = *consumer->payload;
            producer =
                &ring_sched_states[p.fanin_ring_ids[state]].ring->get_slot_state_by_task_id(p.fanin_local_ids[state]);
        }
    }

    // Thread 0 entry point: drain a bounded batch from the orchestrator's
    // SPSC queue, then classify+route each drained task terminally. Returns
    // the count of routed tasks (also the drained count — each drained task
    // is classified once and never re-queued).
    //
    // Sub-phase timing pointers (optional). If non-null, cumulative cycle/
    // iteration counters for Stage 1 (SPSC drain) and Stage 2 (classify+route)
    // are accumulated into them.
    int drain_wiring_queue(bool force_drain = false) {
        // Stage 1: drain SPSC → drain_buf
        int drained = wiring.queue.pop_batch(wiring.drain_buf, PendingState::DRAIN_BATCH);

        // Backoff when nothing to do and orchestrator isn't pressing
        if (drained == 0) {
            if (!force_drain && !wiring.orch_needs_drain.load(std::memory_order_acquire) &&
                wiring.backoff_counter < PendingState::BACKOFF_LIMIT) {
                wiring.backoff_counter++;
                return 0;
            }
        }
        wiring.backoff_counter = 0;

        // Stage 2: classify + route each drained task in-line. Each task's
        // state is "all met → ready_queue" or "first unmet → register on that
        // producer's wake_list". Tasks are scanned exactly once here;
        // re-scans on producer completion happen via on_mixed_task_complete's
        // wake_list drain.
        for (int i = 0; i < drained; i++) {
            PTO2TaskSlotState *s = wiring.drain_buf[i];
            int state = classify_fanin_state(s);
            if (state < 0) {
                push_ready_routed(s);
            } else {
                // Producer is in fanin_ring_ids[state] (may differ from
                // the consumer's ring under multi-ring fanin). When the
                // producer completes, its wake_list drain rescans this
                // consumer and either pushes to ready or re-registers on
                // the next unmet producer.
                int32_t prod_local = s->payload->fanin_local_ids[state];
                uint8_t prod_ring = s->payload->fanin_ring_ids[state];
                auto &ring = *ring_sched_states[prod_ring].ring;
                PTO2TaskSlotState *producer = &ring.get_slot_state_by_task_id(prod_local);
                register_wake(producer, s);
            }
        }

        return drained;
    }

    int get_ready_tasks_batch(
        PTO2ResourceShape shape, PTO2LocalReadyBuffer &local_buf, PTO2TaskSlotState **out, int max_count
    ) {
        int count = 0;
        while (count < max_count && local_buf.count > 0)
            out[count++] = local_buf.slot_states[--local_buf.count];
        int remaining = max_count - count;
        if (remaining > 0) count += ready_queues[static_cast<int32_t>(shape)].pop_batch(out + count, remaining);
        return count;
    }

    bool on_subtask_complete(PTO2TaskSlotState &slot_state) {
        // Relaxed fetch_add: completed_subtasks is a pure counter with no
        // other observers piggybacking state through it. The only readers
        // are this fetch_add itself (per-subtask) and reset_for_reuse's
        // relaxed init. Real publication of the producer's completion to
        // consumer threads happens downstream in on_mixed_task_complete via
        // completion_flag.store(release) + wake_list_head.exchange(acq_rel)
        // — those are the AICPU↔AICPU sync edges. The producer→consumer
        // GM data ordering is handled by AICore-side cache coherence
        // independent of this counter's ordering.
        int16_t prev = slot_state.completed_subtasks.fetch_add(1, std::memory_order_relaxed);
        return (prev + 1) == slot_state.total_required_subtasks;
    }

    // Publish this slot as COMPLETED, then advance the per-ring monotonic
    // completed_watermark — the highest local_id W such that every task
    // 0..W has reached COMPLETED. Reclamation in advance_ring_pointers gates
    // on watermark >= producer.last_consumer_local_id, so no consumer→producer
    // notification edge is needed.
    void on_mixed_task_complete(PTO2TaskSlotState &slot_state) {
        // (m) Skip slot_state.task_state.store here; completion_flags below is
        // the single source of truth. Saves one atomic release store per task.
        const int32_t my_id = static_cast<int32_t>(slot_state.task->task_id.local());
        int32_t ring_id = slot_state.ring_id;
        auto &rss = ring_sched_states[ring_id];
        auto &ring = *rss.ring;

        // Publish to the polling-fast completion array. Release ordering
        // makes the producer's output writes visible to consumers that
        // acquire-load this byte in fanin_satisfied.
        ring.completion_flags[my_id & ring.task_window_mask].store(1, std::memory_order_release);

        // Drain the wake list. Each consumer registered on this slot was
        // waiting on at least one unmet fanin (this one). After
        // completion_flag is set above, atomic-exchange wake_list_head to
        // SENTINEL (refusing any future registrations) and process each
        // waiter: rescan its fanin, route to ready_queue if all met, else
        // re-register on the new first-unmet producer. Ordering:
        // completion_flag is set BEFORE the exchange, so any consumer that
        // races a registration against our exchange and observes a SENTINEL
        // during retry will see completion_flag=1 and either rescan-and-route
        // or self-register on the next unmet.
        PTO2TaskSlotState *waiter = slot_state.wake_list_head.exchange(WAKE_LIST_SENTINEL, std::memory_order_acq_rel);
        while (waiter != nullptr && waiter != WAKE_LIST_SENTINEL) {
            PTO2TaskSlotState *next = waiter->next_in_wake_list;
            // next_in_wake_list left as-is: every re-registration via
            // register_wake() overwrites the field before the CAS publishes
            // the consumer, and reset_for_reuse() clears it on slot reuse.
            // No reader between here and the next overwrite/reset.
            // Fast path: single-fanin waiters were waiting on *us* (the only
            // possible fanin). No rescan needed — push straight to ready.
            // Saves one classify_fanin_state call (a byte read in
            // completion_flags) per waiter. Skips the cache-miss-prone
            // multi-ring lookup for the common chain-task case where each
            // task has exactly one predecessor.
            if (waiter->payload->fanin_count == 1) {
                push_ready_routed(waiter);
                waiter = next;
                continue;
            }
            int state = classify_fanin_state(waiter);
            if (state < 0) {
                push_ready_routed(waiter);
            } else {
                // Still some fanin unmet — re-register on the new first
                // unmet producer's wake list.
                int32_t prod_local = waiter->payload->fanin_local_ids[state];
                uint8_t prod_ring = waiter->payload->fanin_ring_ids[state];
                auto &prod_ring_hdr = *ring_sched_states[prod_ring].ring;
                PTO2TaskSlotState *producer = &prod_ring_hdr.get_slot_state_by_task_id(prod_local);
                register_wake(producer, waiter);
            }
            waiter = next;
        }

        // CAS-advance the watermark, bounded by my_id (which we know is
        // published since we just completed it). If a forward task we observe
        // as COMPLETED is also published, but a gap remains, we stop — the
        // task filling the gap will resume the walk when it completes.
        int32_t w = ring.completed_watermark.load(std::memory_order_acquire);
        while (w < my_id) {
            int32_t next = w + 1;
            if (ring.completion_flags[next & ring.task_window_mask].load(std::memory_order_acquire) == 0) break;
            if (ring.completed_watermark.compare_exchange_weak(
                    w, next, std::memory_order_acq_rel, std::memory_order_acquire
                )) {
                w = next;
            }
        }

        // Try to retire slots whose last consumer has reached COMPLETED.
        // Gate the try-lock + advance walk on a lag threshold: most
        // completions advance the watermark by 1 slot; firing the try-lock
        // per completion costs ~10-30 ns × ~65K completions × N threads of
        // wasted CAS attempts. With the gate, the try-lock fires ~32× less
        // often. Empirically 32 is the sweet spot — bigger thresholds let
        // the allocator stall more often waiting for reclamation. The lag
        // read of last_task_alive is non-atomic but monotonic and only used
        // as a hint — stale-but-OK.
        if (w - rss.last_task_alive >= 32) {
            int32_t expected_lock = 0;
            if (rss.advance_lock.compare_exchange_strong(
                    expected_lock, 1, std::memory_order_acquire, std::memory_order_relaxed
                )) {
                rss.advance_ring_pointers();
                rss.advance_lock.store(0, std::memory_order_release);
            }
        }
    }

    // === Cold-path API ===

    static PTO2SchedulerLayout reserve_layout(DeviceArena &arena, int32_t /*dep_pool_capacity*/) {
        PTO2SchedulerLayout layout{};
        layout.ready_queue_capacity = PTO2_READY_QUEUE_SIZE;
        layout.spsc_capacity = PTO2_WRIRING_QUEUE_SIZE;

        for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++)
            layout.off_ready_queue_slots[i] = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
        layout.off_dummy_ready_queue_slots = ready_queue_reserve_layout(arena, PTO2_READY_QUEUE_SIZE);
        layout.off_pending_spsc_buffer = PTO2SpscQueue::reserve_layout(arena, PTO2_WRIRING_QUEUE_SIZE);
        return layout;
    }

    bool init_data_from_layout(const PTO2SchedulerLayout &layout, DeviceArena &arena, void *sm_dev_base) {
        PTO2SchedulerState *sched = this;
        sched->sm_header = reinterpret_cast<PTO2SharedMemoryHeader *>(sm_dev_base);

        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
            if (!sched->ring_sched_states[r].init_data_from_layout(sm_dev_base, r)) return false;

        for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++)
            if (!ready_queue_init_data_from_layout(
                    &sched->ready_queues[i], arena, layout.off_ready_queue_slots[i], layout.ready_queue_capacity
                ))
                return false;
        if (!ready_queue_init_data_from_layout(
                &sched->dummy_ready_queue, arena, layout.off_dummy_ready_queue_slots, layout.ready_queue_capacity
            ))
            return false;

        if (!sched->wiring.queue.init_data_from_layout(arena, layout.off_pending_spsc_buffer, layout.spsc_capacity))
            return false;

        sched->wiring.backoff_counter = 0;

        return true;
    }

    void wire_arena_pointers(const PTO2SchedulerLayout &layout, DeviceArena &arena) {
        PTO2SchedulerState *sched = this;
        for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++)
            ready_queue_wire_arena_pointers(&sched->ready_queues[i], arena, layout.off_ready_queue_slots[i]);
        ready_queue_wire_arena_pointers(&sched->dummy_ready_queue, arena, layout.off_dummy_ready_queue_slots);
        sched->wiring.queue.wire_arena_pointers(arena, layout.off_pending_spsc_buffer);
    }

    // Forget per-region pointers; arena owns the backing memory.
    void destroy() {
        PTO2SchedulerState *sched = this;
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
            sched->ring_sched_states[r].destroy();
        sched->wiring.queue.destroy();
        for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++)
            ready_queue_destroy(&sched->ready_queues[i]);
        ready_queue_destroy(&sched->dummy_ready_queue);
    }

    // Surgical reset for arena reuse (#1234): resets per-run mutable state
    // without redoing the O(ready_queue_capacity) buffer-zeroing that
    // init_data_from_layout does. Ring pointer is re-set from sm_dev_base
    // since we can't rely on the previous run's value being valid across
    // arena reuse.
    void reset_for_reuse(void *sm_dev_base) {
        PTO2SchedulerState *sched = this;
        sched->sm_header = reinterpret_cast<PTO2SharedMemoryHeader *>(sm_dev_base);
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            sched->ring_sched_states[r].ring = pto2_sm_layout::ring_header_addr(sm_dev_base, r);
            sched->ring_sched_states[r].last_task_alive = 0;
            sched->ring_sched_states[r].advance_lock.store(0, std::memory_order_relaxed);
        }
        for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++)
            sched->ready_queues[i].reset_for_reuse();
        sched->dummy_ready_queue.reset_for_reuse();
        sched->wiring.queue.reset_for_reuse();
        sched->wiring.backoff_counter = 0;
        sched->wiring.orch_needs_drain.store(false, std::memory_order_relaxed);
        sched->async_wait_list.reset_for_reuse();
    }
};

// Scheduler cold-path API is declared as PTO2SchedulerState member functions.
// See init()/destroy() below the struct definition.

inline bool
AsyncWaitList::try_inline_complete_locked(AsyncWaitList::DrainCompletionSink &sink, PTO2TaskSlotState &slot_state) {
    sink.sched->on_mixed_task_complete(slot_state);
    sink.inline_completed++;
    return true;
}

template <bool Profiling>
inline AsyncPollResult
AsyncWaitList::poll_and_complete(AICoreCompletionMailbox *aicore_mailbox, PTO2SchedulerState *sched) {
    AsyncPollResult result;
    if (!try_lock()) return result;

    AsyncWaitList::DrainCompletionSink sink{};
    sink.sched = sched;

    int32_t drain_err = PTO2_ERROR_NONE;
    drain_aicore_completion_mailbox_locked(aicore_mailbox, sink, drain_err);
    if (drain_err != PTO2_ERROR_NONE) {
        result.error_code = drain_err;
        unlock();
        return result;
    }
    result.completed += sink.inline_completed;

    for (int32_t i = count - 1; i >= 0; --i) {
        AsyncWaitEntry &entry = entries[i];
        uintptr_t last_invalidated_counter_line = static_cast<uintptr_t>(-1);
        for (int32_t c = 0; c < entry.condition_count; c++) {
            CompletionCondition &cond = entry.conditions[c];
            if (cond.satisfied) continue;
            if (cond.completion_type == COMPLETION_TYPE_COUNTER && cond.counter_addr != nullptr) {
                uintptr_t counter_line = mailbox_cache_line(cond.counter_addr);
                if (counter_line != last_invalidated_counter_line) {
                    cache_invalidate_range(reinterpret_cast<const void *>(counter_line), sizeof(uint32_t));
                    last_invalidated_counter_line = counter_line;
                }
            }
            CompletionPollResult poll = cond.test();
            if (poll.state == CompletionPollState::FAILED) {
                result.error_code = poll.error_code;
                result.failed_slot_state = entry.slot_state;
                unlock();
                return result;
            }
            if (poll.state == CompletionPollState::READY) {
                cond.satisfied = true;
                cond.retire();
                entry.waiting_completion_count--;
            }
        }

        if (entry.normal_done && entry.waiting_completion_count <= 0) {
            sched->on_mixed_task_complete(*entry.slot_state);
            result.completed++;

            int32_t last = count - 1;
            if (i != last) entries[i] = entries[last];
            count = last;
        }
    }

    unlock();
    return result;
}
