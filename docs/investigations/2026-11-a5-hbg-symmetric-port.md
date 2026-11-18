# Porting symmetric AICPU scheduling onto hbg-aicpu-perf

**Date:** 2026-11-17
**Verdict:** adapt the five-file algorithm from PR #2179 to the current A5
AICPU scheduler. Reject copying the complete old runtime or rewriting the
scheduler. The conversion and requested die-aware placement are implemented;
targeted checks, A5 builds, eight applicable simulator invocations and all nine
NPU1 workload runs pass. The C owner pattern matches die-local assignment;
physical core-to-die IDs were not exported for independent verification.

## Scope and source revisions

The desired behavior is that every AICPU scheduling thread owns AICore
clusters, polls completion, dispatches work, and resolves dependencies.
Symmetry describes thread responsibilities; shared queues and the deferred
completion consumer still need their existing synchronization.

| Source | Revision | Role |
| --- | --- | --- |
| `hbg-aicpu-perf` | `173ba8859126de29cf7adc43f25d19b9b883d348` | Implementation baseline |
| `hbg-aicpu-perf-symmetric` | Working changes on the same revision | Implemented symmetric scheduler and placement |
| `tracr-a5-hbg-symmetric` | `51dba3fc04ce94a882a8dbf2ff6fa72c313c905d` | Existing A5 reference |
| [PR #2179](https://github.com/hw-native-sys/simpler/pull/2179) | `fa08b104cd1502b41b96dfce531a271b28d0c83a` | Original a2a3 conversion |

The baseline forces A5 HBG through the AICPU scheduler. Its relevant executor
is `aicpu_legacy_executor.cpp`; `aicpu_executor.cpp` also contains the newer
resident-AICore route and must not be replaced with the old executor.

## Comparing the three options

This was the initial source investigation. The implementation below follows
the PR #2179 option, adapted to the current contracts.

| Option | Assessment | Decision |
| --- | --- | --- |
| Extract the old A5 symmetric implementation | Faithful PR #2179 port, with useful A5-specific reference code. Complete-file copying would overwrite newer contracts. | Use as a cross-check. |
| Adapt PR #2179 | Smallest clear description of the change. Existing queues, wake lists and atomics already provide most concurrency machinery. | Chosen implementation blueprint. |
| Reimplement from scratch | Would rediscover the same thread-role conversion while expanding the proof and testing burden. No demonstrated performance reason to replace the scheduler. | Reject for this change. |

The old A5 port did follow the original approach. Comparing the original HBG
files with the first symmetric copies shows identical cold-path changes;
completion/context changes differ only in deleted comment terminology, and
dispatch adds a few TraCR lines around the same conversion.

The large old diff is principally packaging:

| Change | Size | Necessary for symmetry here? |
| --- | ---: | --- |
| PR #2179, five implementation files | +173 / -391 | Algorithmic reference |
| Initial A5 conversion, excluding copied unchanged code | +183 / -399 | Algorithmic reference |
| Full A5 runtime addition, `be57273a8` | +4788 / -1 | No: separate runtime/build/test packaging |
| Runtime layout repair, `1859c1dc1` | +132 / -23 | No: forwarding headers and build layout |
| Die-aware assignment, `6522c7959` | +215 / -9 | Independent performance candidate |
| Affinity/four-thread extension, `51dba3fc0` | +59 / -0 | Independent performance candidate |

These are historical diff counts, not an estimate of the finished new patch.
The complete PR commit is +202/-412 including its test and documentation;
its PR body's +195 insertion count is stale. A path-remapped `git apply
--check` accepts the cold-path and completion hunks but rejects executor,
context and dispatch hunks. Even the accepted completion hunk requires
semantic adaptation to the current API.

## Implemented scheduler and placement

The existing A5 `host_build_graph` runtime is modified in place on the new
branch. The unchanged performance branch remains available for a later
baseline comparison. This change adds no runtime directory, behavior flag,
TraCR integration or historical compatibility path.

| File under `src/a5/runtime/host_build_graph/` | Implemented change |
| --- | --- |
| `aicpu/aicpu_legacy_executor.cpp` | All threads enter `resolve_and_dispatch`; boot, shutdown and terminal reporting remain intact. |
| `runtime/scheduler/scheduler_cold_path.cpp` | Clusters belong to all scheduling threads; dedicated SPSC completion-queue allocation is removed. |
| `runtime/scheduler/scheduler_context.h` | Dedicated resolver index/entry and `CompletedTaskQueue` storage are removed; observed CPU/die metadata supports placement. |
| `runtime/scheduler/scheduler_completion.cpp` | Finished tasks resolve inline using the current `complete_task` outcome. Full-mailbox publication can help the consumer. |
| `runtime/scheduler/scheduler_dispatch.cpp` | Every scheduling thread services async and dummy work, accounting, errors and profiling; the dedicated resolver loop is removed. |

For five threads and the full 36-cluster census, ownership changes from
9/9/9/9 plus one resolver to 8/7/7/7/7. Each cluster retains its AIC plus
two AIVs. This reduces the maximum per-thread polling set from 27 cores to
24; performance is still unmeasured.

The conversion preserves current Graph image/materialization, captured
ready-slot, MIX/lane, early-dispatch publication, sync-start drain, and
terminal/teardown contracts.
In particular, `SchedulerState::complete_task` returns
`TaskCompletionOutcome`: Graph body completion need not advance the outer
stream count. Accounting uses `stream_tasks_completed` and propagates
`error_code`. Graph-root release stays inline as explained in
[the earlier drain investigation](2026-09-hbg-graph-release-vs-drain-reorder.md).

### Requested affinity comparison

The user subsequently requested all three placements on the new symmetric
branch, including die-aware assignment for the four-thread case. That extends
the original symmetry-only scope:

| Configuration | Requested CPUs | Scheduling threads | Placement with verified full census |
| --- | --- | ---: | --- |
| A | `[1,2,3,4,5]` | 5 | Round-robin, 8/7/7/7/7 clusters |
| B | `[2,3,4,5,6]` | 5 | Round-robin, 8/7/7/7/7 clusters |
| C | `[2,3,4,5]` | 4 | Die-local, 9 clusters each; CPUs 2/3 on die 0 and 4/5 on die 1 |

All selected threads schedule; the last CPU has no dedicated resolver role
even if a generic platform affinity log calls its slot `orch`. Existing
thread-count validation remains in place, and the benchmark explicitly
selects five or four threads through the current call configuration.

The placement helper uses the CPU actually observed by the affinity gate,
including report-order recovery, and physical-die metadata from the host
topology probe. It never infers the die from the logical CPU number alone.
Unknown or ambiguous topology, duplicate CPU observations, unequal scheduler
populations across dies, or incoherent cluster dies retain round-robin
ownership. Die-local assignment requires coherent AIC/AIV triples on both
dies; every cluster still has exactly one owner. The cold path checks the
result against each core tracker's capacity.

This requires an explicit packed CPU-to-die map in the current A5
`KernelArgs`: its size changes from 176 to 184 bytes, with the new map at
offset 172 and `force_simt_anchor` at 176. Host and device producers/consumers
use the updated declaration and layout assertions together. There is no old
layout fallback. Graph images and AICore compute kernels are unchanged;
the launch payload must be rebuilt consistently.

## Concurrency and profiling adaptations implemented

1. **Deferred-mailbox progress.** The 4096-message ring can fill before all
   FIN handlers reach loop-level polling: 64 tasks with 64 conditions and
   one normal-done message require 4160 messages. Both publication sites now
   use `publish_async_with_progress`, which helps drain under the existing
   consumer lock and retries. A deterministic test fills the real ring,
   verifies assistance retires 59 tasks, then retires the remaining five
   exactly once. Cancellation and failed assistance stop publication.
2. **Concurrent async inspection.** `AsyncWaitList::count` is now a relaxed
   atomic occupancy hint. Size/alignment and lock-free assertions preserve
   its layout; entry access remains under the existing try-lock. Synchronous
   scheduler passes avoid an unconditional contended lock attempt.
3. **Completion and errors.** Ordinary, deferred and dummy paths retain
   `completed_tasks_`, the profiling mirror and current Graph stream counts.
   A regression test first demonstrated that successful inline completions
   were lost from the returned count when a later mailbox registration
   failed; the poll result now retains them, and callers account for them
   before reporting the error. Completion failures use the existing
   `fail_scheduler` path, including error metadata and emergency shutdown.
4. **Phase attribution.** Empty async polls still compact into one bar.
   Symmetric lanes now flush that accumulator before every other emitted
   scheduler phase, including Drain, GraphPrepare, Dispatch and EarlyDispatch,
   so a later backdated poll bar cannot overlap preceding work. Current phase
   enums and queue snapshots are retained.

The original mailbox and unlocked-count findings concerned a naive symmetric
port, not observed failures of the unchanged performance branch. The new
mailbox tests exercise the production helper and `SchedulerState`; they do
not replace onboard validation.

## Validation and performance results

| Check | Status |
| --- | --- |
| Ten focused C++ test targets | Passed: async initialization/mailbox/progress, Graph activation, A5/a2a3 drain, onboard/sim affinity TLS, topology/map/layout, and cluster assignment |
| Nineteen Python capture-builder tests | Passed; the scheduler-phase fixture uses the current structured trace schema and preserves resident-runtime coverage |
| Six production syntax checks | Passed: completion and dispatch translation units at `(SIMPLER_DFX, SIMPLER_SCHED_PROFILING)` = `(0,0)`, `(1,0)`, `(1,1)` |
| `git diff --check` | Passed |
| Full A5 onboard and simulation builds | Passed |
| Eight applicable simulator invocations | Vector, scheduler phases, Graph sync-start and deferred Graph visibility passed with both four and five threads |
| Initial hardware phase fixture | Kernel/output checks passed; the corrected validator accepts 34 dispatches, 34 completions and one dummy across five streams |
| Hardware cluster census and timings | Passed: A/B/C placement captures and all nine workload runs completed on NPU 1; task `task_20261118_174418_485453` exited 0 |

Local evidence is retained under
`~/tmp/hbg-symmetric-benchmark-20261117/`, including
`focused-cpp-tests.log`, `scheduler-local-checks.txt` and `tests-manifest.json`.
These are targeted checks, not a full hardware suite. Two attempted
`defer_wait_shapes::mask_2_graph_instances` simulator runs failed because
their kernel explicitly rejects `resident_enabled != 1`. The AICPU path
uses zero. The supported `defer_wait_visibility::aiv_graph_take` replacement
passed with both thread counts; the original failures remain in the artifacts.

The initial hardware validator required optional `metadata.core_to_thread`,
which the capture omitted. Two failed 319,999-byte device-to-host copies in
the driver log match the full shared-memory refreshes used by metadata
collection and counter reconciliation. The metadata reader ignores its copy
result, so its initially zero header can leave ownership empty. Record
collection uses separate narrow reads; complete phase records survived.
The underlying driver `Get dst tseg failed` remains unresolved. This is
transfer-failure evidence, not proof of a scheduler publication race.

Full-width Graph/MIX/SPMD validation instead obtains scheduler ownership
from PMU CSV `thread_id`, the actual AICPU ready-queue owner. Same-run
terminal notes can provide physical core IDs, but the completed affinity
matrix did not export those notes. The checker joins PMU ownership to raw
records by run epoch and logical core, requires all 108 workers and 36 intact
clusters, and cross-checks raw ownership when present. Inner `DrainPrepare` and
`DrainPublish` records are contained within `Drain`; they are not overlapping
outer phases. For this sync-start fixture their counts, plus `Complete`,
must each be 324. Ordinary `Dispatch` counters are not an independent total
for work published through the drain path.

The authorized timing matrix completed on the symmetric branch: PA Case1,
Qwen B16/seq3500/40-layer, and the supplied DeepSeek CSA B4/S2 case, each with
configurations A, B and C. All runs used NPU 1 under one allocation, identical
toolchain/PTO pin and runtime binaries, and `--rounds 5`. The parser found five
real `chip.run` invocations per case, discarded the first, and averaged rounds
1–4 (`n=4`) using STrace `chip.run.runner_run.device_wall`.

| Workload | A `[1,2,3,4,5]` ms | B `[2,3,4,5,6]` ms | C `[2,3,4,5]` ms |
| --- | ---: | ---: | ---: |
| Paged attention Case1 | 53.784299 | 49.194323 | 45.296675 |
| Qwen3 B16/seq3500/40-layer | 31.081846 | 30.359390 | 28.170107 |
| DeepSeek CSA B4/S2/seed0 | 2.340691 | 1.897766 | 1.720761 |

All nine runs and the three placement captures passed. Baseline/final runtime
binary hashes match, the packaged affinity JSON was restored and verified, and
the supplied CSA fixture passed all seven outputs in each of its five rounds
for each affinity (105 output checks total). Paged attention and Qwen use
`--skip-golden`.

The placement captures confirm A/B round-robin cluster ownership
(8/7/7/7/7) and C's nine clusters per scheduler. C's owner pattern matches the
die-local assignment for CPUs 2/3 and 4/5. The runtime log records each
requested CPU list, but this run did not export the PMU terminal physical-core
IDs; the physical core-to-die map and actual per-thread CPU IDs therefore are
not independently proven by these artifacts. Treat C's result as a successful
die-local owner-pattern check with that observation limit, not as a direct
physical-core census. No production source was changed for the benchmark.

Local report and complete raw artifacts are under
`~/tmp/hbg-symmetric-benchmark-20261117/`; see `REPORT.md`, `summary.json`,
`rounds.csv`, and the per-affinity `validation/` and `results/` directories.
A new comparison against the unchanged performance branch remains outside
this affinity matrix.

Local worktrees: `~/src/simpler-hbg-aicpu-perf-symmetric` for this work and
`~/src/simpler-hbg-aicpu-perf` for the unchanged baseline; main remains in
`~/src/simpler`.
