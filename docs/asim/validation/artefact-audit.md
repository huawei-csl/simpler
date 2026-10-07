# Is an M0-vs-M2 delta an artefact? (audit, 2026-09-17)

> **Resolved 2026-10-07.** The audit's conclusion -- that the fix is virtual time
> rather than a faster simulator -- is what now runs: each simulated device keeps
> a per-thread ledger of its work beyond the latencies it models and runs that
> thread's cores on real time less it, so the phase recorder's removal of the
> same ledger is exact ([DESIGN.md](../DESIGN.md) §3). Its reading that M2's deltas
> were **lower** bounds was wrong: the overrun it describes as inflating M2's
> window had already been subtracted, cores' compute included, which made the
> deltas over-statements. The text below is the audit as written.

The M2 wins are large enough that the burden is on the simulator to show it is
not handing them out. Everything below was checked with the two arms running the
same graph back to back on one locked card.

**Symmetric, and therefore not a source of advantage:**

| property | M0 (`asim`) | M2 (`asimgq`) |
| -------- | ----------- | ------------- |
| compute issued/round, PA | 91,128 us | 91,051 us (-0.08 %) |
| compute issued/round, qwen | 746,555 us | 746,519 us (-0.005 %) |
| tasks completed | 65,792 / 11,087 | 65,792 / 11,087 |
| cores | 72, from `get_worker_count()` | 72, same call |
| compute draw | table + bounded Irwin-Hall | same code, same calib |
| slots per core | 2 (`active` + `pushed`) | 2 (`outstanding 0..2`) |
| submit -> work starts | 1008 ns | 1038 ns |

M2's dispatch is the longer of the two: it pays the die crossing to reach its
controller *and* the controller-to-core link, where M0 crosses once.

**The one large asymmetry runs against M2.** M0's modelled status read (195 ns)
exceeds what the simulator costs to execute one, so it never overshoots and its
window is model-faithful: overrun is 0-4 us per thread per round. M2's modelled
poll (5-10 ns) is *below* its own cost, so nearly every poll overshoots and the
excess lands in the measured window -- 15,211 us per thread per round on qwen,
about 65 % of it. The published deltas are therefore **lower bounds**.

**Cheapening the simulator does not fix it, because the poll count is
demand-driven.** Cutting the poll from 136 ns to 117 ns raised the count from
104k to 114k and left wall-per-poll at 210 -> 206 ns; only ~120 ns of that is the
simulator, the rest being the scheduler's own loop. Since `asimgq` spins to a
real-time deadline, a faithful window needs modelled latency >= the real
per-poll cost (~200 ns). At 10 ns it is 20x under, so the loop outruns the model
and the window measures the host. The fix is virtual time -- advancing a model
clock instead of spinning -- not a faster simulator.

**What this licenses.** Rankings and the direction of an M0-vs-M2 delta are
sound. M2's absolute `device_wall` is not: it is a floor set by simulator plus
scheduler cost, which is also why qwen barely moves between a 5 ns and a 10 ns
read model -- both sit under the floor. Only a 30 ns *per word* read (~300 ns on
a ten-deep scan) rises above it and moves the number.

**Measurement discipline this established.** Three consecutive runs of one
binary agree to 0.55 %; the same build an hour later differs ~3 %, with the card
held throughout. Compare arms back to back inside one submission. The
`[GQ_WORK]` counters themselves cost ~1.3 % of `device_wall`, so an instrumented
run is its own baseline.
