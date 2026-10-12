# Where the GroupQueue loses: deepseek_v4_flash_decode

The only case that separates the two designs, and it separates them the wrong
way. Two dies, 43 layers, 368 kernels, ~29,000 in-graph tasks a round per rank.
Three arms interleaved in one lock, two independent sessions:

| session | silicon | M0 | M2 | M2 vs M0 | window |
| ------- | ------- | -- | -- | -------- | ------ |
| 2026-10-09, n=12 | 36.187 ms | 36.638 ms | 38.384 ms | **+4.8 %** | +3.4 % |
| 2026-10-12, n=18 | 35.740 ms | 36.710 ms | 37.911 ms | **+3.3 %** | +2.7 % |

Compute issued agrees between the arms to 0.5 %, every round completes every
task, and every duration comes from the calibration. M0 reads +1.2 % and +2.7 %
against silicon.

**The case is critical-path bound, which is the opposite of what the GroupQueue
is for.** Its cube cores are **3.3 % busy** -- 1.21 ms per cube core against a
36.6 ms round -- and the whole device carries ~44 ms of core time a round across
72 cores, under 2 % utilisation. The wall is 43 layers of serial chain at ~842 us
each, against ~28 us of cube work per core per layer. paged_attention's -48 %
came from feeding idle cores faster where parallel work was abundant; here the
work is a chain, so there is no throughput to recover.

**And the mechanism that would shorten a chain never engages: `deps_named=0`.**
The controller resolves no dependency, so the GroupQueue adds its indirection to
every hop without taking the AICPU out of any -- about 0.8 us across roughly
1,300 serial hops a round, which is the measured delta. M2 also polls 5.3x as
often as M0 (28.8k against 5.5k per thread per round) and carries a 6.1 ms
ledger against 0.32 ms.

**Grouping is structurally unreachable here, not merely undeclared.**
`OrchestratorState::begin_group` / `end_group` back-fill a group over a range of
**host-side task ids** (`task_allocator.active_count()`). This case's
`host_build_graph` orchestration is the Graph form -- one GRAPH task per layer,
expanded by the device Scheduler -- so its in-graph tasks are created on device
and never pass through that range. The same holds for qwen's Graph orchestration
above. **No graph-execution case can declare a group today**, which is why the
GroupQueue's central mechanism is dormant on both.

Two routes to a measurable improvement, in order of leverage:

- **Let a Definition declare its tasks as a group**, so the Scheduler tags them
  at expansion. A Definition is already bounded, internally connected and
  replayed -- the shape a group wants. This is the only route that reaches any
  graph-execution case.
- **A flat `host_build_graph` orchestration for this case**, which would make
  grouping declarable today. The existing flat source is the
  `tensormap_and_ringbuffer` one and does not compile against the
  `host_build_graph` orchestration API (`simpler::tmr::` types), so it needs a
  port or a regeneration.

Reducing M2's per-hop cost is worth doing on its own terms but cannot turn a
latency-bound case into a win.

**Two caveats on the absolute numbers, neither of which explains the delta**
(both arms share them). The per-rank asymmetry silicon shows -- 32,273 against
28,871 AICore tasks a round -- is lost, because the 26 predicated MoE tile tasks
key on `recv_count_out`, which `dispatch_meta` writes on device and the model
never computes; both simulated ranks then do identical work. And a comm kernel's
cross-die wait is baked into its calibrated duration, so it is charged but does
not respond to rank drift -- which is exactly what a scheduler change produces.
Until both are addressed this case is a **scheduler benchmark, not a fidelity
reference**.

## Reproducing it

Method and the discipline the numbers depend on are in
[../running.md](../running.md); the simulated baseline's own error against
silicon is in [fidelity.md](fidelity.md). Three things are specific to this case:

- **Two dies.** `task-submit --device auto --device-num 2`, and the run passes
  both of `$TASK_PHYS_DEVICE` to the driver.
- **The driver needs one line.** Its `_RUNTIME_ENV` table has entries for
  `tensormap_and_ringbuffer` and `host_build_graph` but not `group_queue`, which
  is `host_build_graph`-derived and takes the same (empty) ring sizing. Without
  it the run raises `KeyError: 'group_queue'`.
- **One calibration covers both ranks.** `deps.json` cannot name the func_id of a
  task the device expands, so it comes from the swimlane's recorded per-task
  kernel ids (`asim_gen_calib.py` prefers them; pass `-` for the dep-gen round).
  The two ranks' records are merged into one table, because the per-rank split is
  which kernels fired rather than how long they take.
