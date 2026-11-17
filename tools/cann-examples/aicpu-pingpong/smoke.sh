#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Smoke-run this tool on one device. tools/build_cann_examples.sh --run calls it
# after building, with $1 = device id.
#
# A smoke only has to prove the launch and measurement path works, so it uses a
# short iteration count and a single rep; the full matrix is the by-hand run in
# README.md.
set -euo pipefail
DEV="${1:?device id required}"
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH must be set}"
: "${SIMPLER_DISPATCHER_SO:?set SIMPLER_DISPATCHER_SO to build/lib/<arch>/dispatcher/libsimpler_aicpu_dispatcher.so}"
if [ ! -f "$SIMPLER_DISPATCHER_SO" ]; then
    echo "dispatcher SO not found at $SIMPLER_DISPATCHER_SO — run pip install first" >&2
    exit 1
fi

export SIMPLER_AICPU_PINGPONG_SO="$HERE/device/build/libaicpu_pingpong.so"
# device_id, launch_count (0 = discovered pool), iters, reps, warmup
exec "$HERE/host/build/aicpu_pingpong" "$DEV" 0 200 1 50
