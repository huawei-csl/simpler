#pragma once

#include <stdint.h>

#include "scheduler/scheduler_types.h"

// =============================================================================
// GqIndexSpace: one manager's GroupQueue index space.
//
// A manager assigns every dispatch a monotone position in its own space and
// learns of completions as a contiguous prefix -- the queue's watermark -- rather
// than by asking each core. Recovering task identity from that prefix is the
// manager's job, and this holds what it needs: what each position was dispatched
// for, how far the prefix has reached, and how much of it the manager has
// consumed.
//
// Every field is private to one scheduler thread, which is load-bearing rather
// than incidental: translation work placed on shared state would be divided by
// nothing as thread count rises.
//
// WINDOW bounds in-flight dispatches, not task count: position i and i+WINDOW
// share a slot, and room() keeps them from ever being live together, so the
// window slides over an unbounded position space.
// =============================================================================
class alignas(64) GqIndexSpace {
  public:
    static constexpr uint32_t WINDOW = 4096;

    // What the manager must recover from a reported position to finish the
    // dispatch: the block's task, which of its subtasks ran, and the tag the
    // dispatch carried.
    struct Owner {
        ChipTaskSlotState *slot;
        int32_t reg_task_id;
        SubtaskSlot subslot;
        int32_t core_id;  // reported to completion for its swimlane/tracking only
        // The tracker offset this dispatch took out of the thread's core budget,
        // or -1 when it took none. A grouped entry takes none: the controller
        // holds it until its producers retire and places it itself, so there is
        // nothing for the manager to give back.
        int32_t capacity_token;
    };

    void init(uint32_t capacity) {
        push_index_ = 0;
        watermark_ = 0;
        retired_cursor_ = 0;
        capacity_ = (capacity == 0 || capacity > WINDOW) ? WINDOW : capacity;
        early_n_ = 0;
        for (uint32_t i = 0; i < WINDOW / 64; ++i) early_[i] = 0;
    }

    // How many further dispatches this manager may hand out before a position has
    // to retire. Callers size their pop by this, so assign()'s refusal is the
    // backstop rather than the normal path.
    // A slot is held from submit until this manager *knows* the entry finished --
    // by the watermark covering it, or by an ahead-of-watermark notification. An
    // entry the controller has already finished but not yet reported still holds
    // its slot, which is what the hardware bound actually is.
    uint32_t room() const {
        const uint32_t held = static_cast<uint32_t>(push_index_ - watermark_) - early_n_;
        return capacity_ > held ? capacity_ - held : 0;
    }
    uint64_t push_index() const { return push_index_; }
    // Positions handed out that this manager has not yet retired. Under the
    // grouping contract this, not the core tracker, is what says work is still in
    // flight: a grouped entry is placed by the controller and takes no core here.
    uint64_t outstanding() const { return push_index_ - retired_cursor_; }
    uint64_t watermark() const { return watermark_; }

    // Take a position without yet knowing what will occupy it. Opening a group
    // reserves one per member, ascending, so every in-group producer already has
    // a position by the time a consumer names it -- which is what lets the
    // controller hold the edge instead of the manager.
    uint64_t reserve() {
        if (room() == 0) return UINT64_MAX;
        const uint64_t idx = push_index_++;
        owner_[idx & (WINDOW - 1)] = Owner{nullptr, 0, SubtaskSlot::AIC, -1, -1};
        return idx;
    }

    void set_owner(uint64_t idx, const Owner &owner) { owner_[idx & (WINDOW - 1)] = owner; }

    // Give back the position taken last, when what it was taken for could not be
    // submitted. Retirement is by contiguous prefix, so a position left reserved
    // and empty halts the prefix at itself and nothing behind it ever retires.
    void unreserve(uint64_t idx) {
        if (push_index_ > 0 && idx == push_index_ - 1) {
            --push_index_;
            owner_[idx & (WINDOW - 1)] = Owner{nullptr, 0, SubtaskSlot::AIC, -1, -1};
        }
    }

    uint64_t assign(const Owner &owner) {
        if (room() == 0) return UINT64_MAX;
        const uint64_t idx = push_index_++;
        owner_[idx & (WINDOW - 1)] = owner;
        return idx;
    }

    Owner &owner_of(uint64_t idx) { return owner_[idx & (WINDOW - 1)]; }

    // The queue reports a contiguous prefix plus the positions finished ahead of
    // it. Record the ahead ones; the prefix is what actually retires.
    void note_early(uint64_t idx) {
        if (idx < watermark_ || idx - watermark_ >= WINDOW) return;
        uint64_t &word = early_[(idx & (WINDOW - 1)) >> 6];
        const uint64_t bit = 1ULL << (idx & 63);
        if ((word & bit) == 0) {
            word |= bit;
            ++early_n_;
        }
    }

    void advance_watermark(uint64_t wm) {
        // Positions the prefix now covers were already counted as known-finished if
        // they arrived as ahead notifications; drop them from that tally as they
        // pass, so a slot is never released twice.
        while (watermark_ < wm) {
            uint64_t &word = early_[(watermark_ & (WINDOW - 1)) >> 6];
            const uint64_t bit = 1ULL << (watermark_ & 63);
            if ((word & bit) != 0) {
                word &= ~bit;
                --early_n_;
            }
            ++watermark_;
        }
    }

    // Positions the manager has not yet turned into completions. Retiring is the
    // manager's own bookkeeping: the queue has already freed the core.
    bool next_retired(uint64_t *out) {
        if (retired_cursor_ >= watermark_) return false;
        *out = retired_cursor_++;
        return true;
    }

  private:
    uint64_t push_index_{0};
    uint64_t watermark_{0};
    uint64_t retired_cursor_{0};
    uint32_t capacity_{WINDOW};
    Owner owner_[WINDOW]{};
    uint64_t early_[WINDOW / 64]{};
    // Positions known finished ahead of the watermark, i.e. slots already released.
    uint32_t early_n_ = 0;
};
