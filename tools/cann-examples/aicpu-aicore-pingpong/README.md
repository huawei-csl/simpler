# aicpu-aicore-pingpong

Measures GM hand-off latency between any two cores on an a5 chip: AICPU↔AICore
and AICore↔AICore. Three standalone programs — an x86 or aarch64 **host**
launcher, an aarch64 **AICPU** `.so`, and an **AICore** `.o` — with no `simpler`
runtime involved. The companion [`../aicpu-pingpong/`](../aicpu-pingpong/)
covers AICPU↔AICPU; its counter lives in the device SO's BSS, so its numbers
are not comparable with the GM hand-offs measured here.

Findings from both chips it has run on are in
[`../a5-latency-findings/`](../a5-latency-findings/README.md).

## Modes

| Mode | What runs | Needs the AICPU `.so`? |
| ---- | --------- | ---------------------- |
| `0` census | every block writes one line and returns | no |
| `1` measure | AICPU↔AICore, one (cpu, slot) pair per round | **yes** |
| `2` probe | checks that an AICore `st_dev` becomes visible in GM | no |
| `3` core2core | AICore↔AICore, one (slot, slot) pair per round | no |

**Start with census on any new part.** It cannot block — every block writes one
line and returns — and it prints the occupied slot ids, which is the part's real
topology.

## Build

`ASCEND_HOME_PATH` must point at a CANN install holding `bin/ccec`, `lib64/`
and `tools/hcc/`. The host half builds for whatever the host is; the AICPU half
always cross-compiles to aarch64.

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/<cann-version>
CROSS=$ASCEND_HOME_PATH/tools/hcc/bin/aarch64-target-linux-gnu

# AICore kernel — dav-c310 is a5 (both 950DT and 950PR); a2/a3 is dav-c220
cmake -S device-aicore -B device-aicore/build -DCCE_AICORE_ARCH=dav-c310-cube
cmake --build device-aicore/build -j

# AICPU driver — only needed for mode 1
cmake -S device-aicpu -B device-aicpu/build \
    -DCMAKE_C_COMPILER=$CROSS-gcc -DCMAKE_CXX_COMPILER=$CROSS-g++ \
    -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64
cmake --build device-aicpu/build -j

# Host launcher
cmake -S host -B host/build
cmake --build host/build -j
```

`libascendcl` pulls in `libascend_dump.so`, which resolves `drvHdc*` against
`libascend_hal.so` from the **driver** package rather than the toolkit. The
default search path is `/usr/local/Ascend/driver/lib64/driver`; pass
`-DASCEND_DRIVER_LIB_DIR=...` if the driver lives elsewhere.

## Run

Always through `task-submit` on a shared box
([`.claude/rules/running-onboard.md`](../../../.claude/rules/running-onboard.md)),
with the device log redirected to the run's own directory.

```bash
export PP_PARTNER_O=$PWD/device-aicore/build/aicpu_aicore_pingpong_partner.o
LOGDIR=$PWD/out/ascend; mkdir -p "$LOGDIR"
export ASCEND_PROCESS_LOG_PATH="$LOGDIR"

# census
task-submit --device auto --device-num 1 \
    --env PP_PARTNER_O --env ASCEND_PROCESS_LOG_PATH \
    --run "$PWD/host/build/launch_pp_census \$TASK_DEVICE 0"

# AICore <-> AICore, every pair
PP_ARENAS=8 PP_REPS=1 task-submit --device auto --device-num 1 --max-time 1500 \
    --env PP_PARTNER_O --env ASCEND_PROCESS_LOG_PATH --env PP_ARENAS --env PP_REPS \
    --run "$PWD/host/build/launch_pp_census \$TASK_DEVICE 3 0 0 255 2 99 2000 1000"

# AICPU <-> AICore (mode 1) also needs:
export SIMPLER_DISPATCHER_SO=<repo>/build/lib/a5/dispatcher/libsimpler_aicpu_dispatcher.so
export PP_DRIVER_SO=$PWD/device-aicpu/build/libpp_driver.so
PP_CPUS=1,2,3   # the cpu_ids ../aicpu-pingpong/ discovery reported
```

Arguments are
`<device_id> [mode] [blockDim] [slot_a] [slot_b] [cpu_a] [cpu_b] [iters] [warmup]`.
`blockDim` of `0` or absent means "ask the device"; pass it explicitly only to
reproduce an older run. `iters`/`warmup` change the floor, so a comparison
between two runs must use the same values.

Environment knobs: `PP_ARENAS` (separate `aclrtMalloc` arenas swept in one
launch), `PP_REPS`, `PP_SEED`, `PP_NSLOTS` (mode 1 sample size), `PP_CPUS`,
`PP_CLOCK_ITERS`, `PP_CHASE_STEPS`, `PP_CHASE_LINES_USED`.

Output is CSV on stdout: `C2C,` rows for mode 3, `CSV,` rows for mode 1, plus
`PROBE,` / `CPUPROBE,` rows carrying each core's clock and GM-read probe.

## Porting to another a5 part

The slot space is 108 wide on every a5 part — AIC `0..17` and AIV `18..53` on
die 0, the same again from 54 on die 1 — but a part with fewer clusters occupies
it **sparsely**, and a disabled cluster leaves a hole. The host therefore runs a
census before every sweep and schedules only the slots that answered, so
`PP_SLOT_DIE`, `PP_SLOT_IS_AIC` and `PP_SLOT_CLUSTER` stay correct unchanged.

`PP_MAX_BLOCKDIM` in [`shared/pingpong.h`](shared/pingpong.h) is the 950DT
value and is now only a fallback: the launcher queries
`ACL_DEVICE_INFO_AI_CORE_NUM`. Over-dispatching blockDim **stalls the stream
until the op-execute timeout with nothing in any log**, which is why the count
is queried rather than assumed.

Two parts measured so far:

| | cube / vector | occupied slots | note |
| --- | --- | --- | --- |
| `Ascend950DT` | 36 / 72 | 108, dense | |
| `Ascend950PR` | 28 / 56 | 84, hole at die 1 cluster 4 | 14 clusters per die |
