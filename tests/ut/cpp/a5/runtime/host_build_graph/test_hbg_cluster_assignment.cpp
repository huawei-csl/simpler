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

#include "scheduler/scheduler_cluster_assignment.h"

namespace {

using simpler::hbg::ClusterDies;
using simpler::hbg::ClusterPlacement;
using simpler::hbg::plan_cluster_assignment;

struct Placement {
    ClusterDies clusters[36]{};
    int32_t owners[36]{};
    int32_t counts[5]{};

    Placement() {
        for (int32_t c = 0; c < 36; ++c) {
            const auto die = static_cast<uint8_t>(c / 18);
            clusters[c] = {die, die, die};
        }
    }

    void expect_round_robin(int32_t threads) {
        int32_t census[5]{};
        for (int32_t c = 0; c < 36; ++c) {
            EXPECT_EQ(owners[c], c % threads);
            ASSERT_GE(owners[c], 0);
            ASSERT_LT(owners[c], threads);
            ++census[owners[c]];
        }
        for (int32_t t = 0; t < threads; ++t)
            EXPECT_EQ(counts[t], census[t]);
    }
};

TEST(HbgClusterAssignment, FiveThreadsKeepBothRequestedSetsRoundRobin) {
    const int32_t cpus[2][5] = {{1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}};
    const int8_t dies[2][5] = {{0, 0, 0, 1, 1}, {0, 0, 1, 1, 1}};
    for (int32_t config = 0; config < 2; ++config) {
        Placement p;
        EXPECT_EQ(
            plan_cluster_assignment(p.clusters, 36, cpus[config], dies[config], 5, p.owners, p.counts),
            ClusterPlacement::RoundRobin
        );
        p.expect_round_robin(5);
        EXPECT_EQ(p.counts[0], 8);
        for (int32_t t = 1; t < 5; ++t)
            EXPECT_EQ(p.counts[t], 7);
    }
}

TEST(HbgClusterAssignment, BalancedActualParticipantsOwnNineLocalClustersEach) {
    // Binding order and logical CPU numbering need not follow physical dies.
    const int32_t cpus[4] = {5, 3, 4, 2};
    const int8_t dies[4] = {1, 0, 1, 0};
    Placement p;
    ASSERT_EQ(plan_cluster_assignment(p.clusters, 36, cpus, dies, 4, p.owners, p.counts), ClusterPlacement::DieLocal);
    int32_t seen[4]{};
    for (int32_t c = 0; c < 36; ++c) {
        ASSERT_GE(p.owners[c], 0);
        ASSERT_LT(p.owners[c], 4);
        EXPECT_EQ(dies[p.owners[c]], p.clusters[c].aic);
        ++seen[p.owners[c]];
    }
    for (int32_t t = 0; t < 4; ++t) {
        EXPECT_EQ(seen[t], 9);
        EXPECT_EQ(p.counts[t], 9);
    }
    EXPECT_EQ(p.owners[0], 1);
    EXPECT_EQ(p.owners[1], 3);
    EXPECT_EQ(p.owners[18], 0);
    EXPECT_EQ(p.owners[19], 2);
}

TEST(HbgClusterAssignment, UnknownDuplicateOrUnbalancedActualParticipantsFallBack) {
    const int32_t cpus[3][4] = {{-1, 3, 4, 5}, {2, 3, 4, 4}, {2, 3, 4, 5}};
    const int8_t dies[3][4] = {{-1, 0, 1, 1}, {0, 0, 1, 1}, {0, 0, 0, 1}};
    for (int32_t config = 0; config < 3; ++config) {
        Placement p;
        EXPECT_EQ(
            plan_cluster_assignment(p.clusters, 36, cpus[config], dies[config], 4, p.owners, p.counts),
            ClusterPlacement::RoundRobin
        );
        p.expect_round_robin(4);
    }
}

TEST(HbgClusterAssignment, IncoherentOrMissingClusterDiesFallBackWithoutLosingCoverage) {
    const int32_t cpus[4] = {2, 3, 4, 5};
    const int8_t dies[4] = {0, 0, 1, 1};
    for (int32_t config = 0; config < 4; ++config) {
        Placement p;
        if (config == 0) p.clusters[7].aiv0 = 1;
        if (config == 1) p.clusters[7].aiv1 = 1;
        if (config == 2) p.clusters[7] = {255, 255, 255};
        if (config == 3)
            for (auto &cluster : p.clusters)
                cluster = {0, 0, 0};
        EXPECT_EQ(
            plan_cluster_assignment(p.clusters, 36, cpus, dies, 4, p.owners, p.counts), ClusterPlacement::RoundRobin
        );
        p.expect_round_robin(4);
    }
}

TEST(HbgClusterAssignment, RejectsInvalidCountsBeforeReadingInputs) {
    Placement p;
    EXPECT_EQ(
        plan_cluster_assignment(p.clusters, 37, nullptr, nullptr, 4, p.owners, p.counts), ClusterPlacement::Invalid
    );
    EXPECT_EQ(
        plan_cluster_assignment(p.clusters, 36, nullptr, nullptr, 6, p.owners, p.counts), ClusterPlacement::Invalid
    );
    EXPECT_EQ(
        plan_cluster_assignment(p.clusters, 36, nullptr, nullptr, 0, p.owners, p.counts), ClusterPlacement::Invalid
    );
}

}  // namespace
