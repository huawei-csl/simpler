# M2 campaign — the ready group queue (2026-09-14)

What the task-grouping contract is worth, measured as **`device_wall`** against an
M0 of `host_build_graph` + simulated cores (`a2a3asim`). Method and fidelity:
[../validation.md](../validation.md). The earlier campaign reads the scheduling
window against a `scan_and_claim` M0, so its numbers are not comparable with
these: [m2-scheduling-window.md](m2-scheduling-window.md).

A separate campaign from the one above, and not comparable to it: that one reads
the scheduling window against a `scan_and_claim` M0, this one reads `device_wall`
against an M0 of `host_build_graph` + simulated cores (`a2a3asim`). Both arms
here are `a2a3asimgq`; what changes between them is the grouping contract.

**The structure.** The shared ready queue holds *groups*, not tasks. A group's
dependencies are on sink tasks only, so a group no external edge reaches is ready
from the start; a scheduler thread takes one and feeds the whole of it to its
controller, internal edges included. paged_attention Case1 annotates as
`group = local_id / 257`: 65 792 tasks in 256 components of exactly 257,
contiguous in local id, with zero edges between them.

**Positions are taken and filled in the same step.** A group is fed in ascending
task order, and each entry takes its position immediately before being submitted.
Ascending order is what lets a controller hold an internal edge at all — a
consumer can only name producers that already hold positions — and it does not
depend on the order the per-shape rows drain. Taking and filling together is what
keeps the manager retiring: it harvests completions as a contiguous prefix, so a
position reserved now and submitted later halts that prefix at itself and nothing
behind it ever retires. An earlier build reserved a whole group up front and
measured `push=3840, watermark=0` — every thread wedged at the first hole.

**Hardware limits.** 32 task slots per controller, 4 dependency comparators per
slot. A slot is occupied from submit until the *manager learns* the task finished,
by watermark or ahead-notification — an entry the controller has finished but not
yet reported still holds its slot. A task with more unmet producers than there are
comparators cannot be expressed, so the manager holds it, and with it the rest of
its group, which is fed in order.

| config | PA `device_wall` | vs M0 |
| ------ | ---------------- | ----- |
| M0 (`a2a3asim`) | 21.907 ms | — |
| M2, grouping off | 18.200 ms | −16.9 % |
| M2 ready group queue, unlimited | 7.794 ms | −64.4 % |
| **M2 ready group queue, 32 slots / 4 deps** | **9.87 ms** | **−55 %** |

The 32/4 figure moved from 9.453 ms as the queue was made correct on a graph of
chained groups (see qwen below): removing a whole-graph pre-scan from seeding took
it to 8.575 ms, covering the position table properly put it back to 9.35 ms, and
handing groups out concurrently costs the rest.

**Same work, verified.** A skip-golden run proves only that nothing deadlocked,
so work equivalence is measured rather than assumed: compute issued by the
controller, dispatches charged, and tasks completed, all at `--rounds 1` so no
counter's reset semantics can be mistaken for a difference.

| | compute/round | dispatches | completed |
| - | ------------- | ---------- | --------- |
| M0 | 97 030.0 µs | 63 701 | 65 792 / 65 792 |
| M2 grouping off | 97 005.7 µs | 62 923 | 65 792 / 65 792 |
| M2 group queue, 32/4 | 96 982.1 µs | 63 898 | 65 792 / 65 792 |

Compute agrees to −0.06 %. The mechanism shows up in the balance rather than the
total: per-thread compute spread is 5.6 % ungrouped and 0.1 % grouped, which is
the same "fill the idle cores" effect the zero-latency balancer oracle found.

**The comparator count is the parameter that decides the design.**

| comparators | PA `device_wall` | stalls forced |
| ----------- | ---------------- | ------------- |
| 1 | 28.254 ms | 293 843 |
| 2 | 22.549 ms | 195 025 |
| 3 | 9.690 ms | 0 |
| 4 | 9.453 ms | 0 |

The cliff is between 2 and 3, not a gentle curve: at 1 comparator the group queue
is *slower than not grouping at all* (18.200 ms) and slower than M0, because a
task that cannot express its producers holds everything behind it in its group's
feed order, and the ungrouped path pays no such penalty — it can dispatch any
ready task from anywhere. PA needs 3; 4 buys margin. The slot count is the milder
knob: 32 versus unlimited costs 21 %.

## qwen decode — where the contract stops working

paged_attention is the shape the contract is built for. qwen decode is not, and
running it is what says which of the two the structure needs.

Its graph has to be expanded first: the shipped orchestration submits each of the
40 decoder layers as one Graph the device Scheduler expands, so dep-gen records 45
outer nodes rather than tasks. Expanded (`decode_fwd_layers_expanded.cpp`) it is
**11,085 tasks, 23,601 edges, all forward — and ONE weakly-connected component**,
against paged_attention's 256 disjoint ones. Its critical path is 33 levels, so
the parallelism is there; the grouping is what fails to use it.

| group size | qwen `device_wall` | internal edges |
| ---------- | ------------------ | -------------- |
| ungrouped | **31.93 ms** | — |
| 277 (one layer) | 91.5 ms | 84.4 % |
| 64 | 75.1 ms | 32.2 % |
| 32 | 62.7 ms | ~20 % |
| 16 | 51.9 ms | ~12 % |

**Grouping loses at every size, and gets better as it does less.** Performance
improves monotonically as internal-edge capture collapses from 84 % to 12 %, which
says none of the gain comes from the controller resolving edges — only from
shrinking the unit of serialisation. Extrapolated, the best group size is 1, which
is not grouping.

The mechanism is the inverse of the one that wins on paged_attention. Per-thread
compute spread, ungrouped against grouped:

| case | ungrouped | grouped | |
| ---- | --------- | ------- | - |
| paged_attention | 5.6 % | **0.1 %** | grouping balances |
| qwen decode (G=16) | 0.5 % | **35.2 %** | grouping unbalances |

A group belongs to one thread, and a thread's controller owns 18 of the 72 cores.
When groups are interchangeable that pins work harmlessly and buys locality; when
they are chained it pins the machine to a quarter of itself. Compute issued is
identical either way (+0.03 %, 11,087 of 11,087 tasks completing), so this is
scheduling, not work.

**The discriminator is whether groups are independent of each other** — not group
size, not edge density, not how many comparators the controller has. The
comparator limit never fires on paged_attention and fires in the hundreds on qwen,
but even a controller with no limits would not fix a graph whose groups cannot run
at the same time.

## What this result does not cover

- **Nothing is numerically verified.** Every figure here is `--skip-golden`.
- **PA is the friendly case**: zero cross-group edges, so the controller resolves
  *every* edge. A group any external edge reaches still arrives task by task —
  the sink-completion counter that would make it general is not built.
- **MIX** tasks are fed as separate cube/vector entries rather than one 3-part
  package entry, which changes their placement.
- The bookkeeping core id handed to `complete_slot_task` is the thread's first
  core, not the controller's actual placement. Inert here (aSim writes no deferred
  slabs) but wrong for swimlane attribution; `queue_entry_core()` exists to fix it.
- **qwen does not run this yet** — see `KNOWN_ISSUES.md`.
