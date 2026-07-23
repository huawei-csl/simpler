#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Statistically compare two `tools/benchmark_rounds.sh`-teed output files.

`benchmark_rounds.sh` tees one file per side (baseline / current), each
containing, per example, the per-round table emitted by
`strace_timing --rounds-table` (a `Round  Host (us)  Device (us) ...` header,
one row per round, terminated by an `Avg <Metric>:` line). This tool re-parses
the raw per-round rows (not just the `Avg` summary line), drops the top/bottom
`--trim-pct` of rounds per side per metric (interference from other processes
on a shared box shows up as high-side outliers), and reports Welch's t-test
(unequal-variance, two-sided) + Cohen's d for each example/metric so a
regression can be called "real" vs "within noise" instead of eyeballing a
delta percentage.

Pure Python, no scipy dependency (scipy is not a project dependency — see
[[venv-isolation]]). The regularized incomplete beta function backing the
t-distribution CDF is the standard continued-fraction algorithm (Numerical
Recipes §6.4); cross-checked against `scipy.stats.ttest_ind` for correctness,
see `docs/dfx/` for the benchmark skill this tool backs.

Usage:
    python -m simpler_setup.tools.compare_benchmark_runs \\
        tmp/benchmark_baseline_*.txt tmp/benchmark_current_*.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from math import exp, lgamma, log, sqrt

# ---------------------------------------------------------------------------
# Pure-Python Welch's t-test (no scipy dependency).
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction evaluation for the incomplete beta function (NR §6.4)."""
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log1m_x = log(1.0 - x) if x < 1.0 else float("-inf")
    ln_bt = lgamma(a + b) - lgamma(a) - lgamma(b) + a * (0.0 if x == 0 else log(x)) + b * log1m_x
    bt = exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def welch_t_test(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Welch's two-sample, two-sided t-test. Returns (t, df, p_value)."""
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1) if na > 1 else 0.0
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1) if nb > 1 else 0.0
    se2 = va / na + vb / nb
    if se2 == 0.0:
        return 0.0, float(na + nb - 2), 1.0
    t = (ma - mb) / sqrt(se2)
    df = se2 * se2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = _betai(df / 2.0, 0.5, df / (df + t * t))
    return t, df, p


def cohens_d(a: list[float], b: list[float]) -> float:
    """Pooled-SD standardized effect size."""
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1) if na > 1 else 0.0
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1) if nb > 1 else 0.0
    pooled_sd = sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) if na + nb > 2 else 0.0
    if pooled_sd == 0.0:
        return 0.0
    return (ma - mb) / pooled_sd


def trim(xs: list[float], pct: float) -> list[float]:
    """Drop the top/bottom `pct`% of sorted values from each tail."""
    xs = sorted(xs)
    n = len(xs)
    k = int(n * pct / 100.0)
    if k <= 0 or n - 2 * k < 1:
        return xs
    return xs[k : n - k]


# ---------------------------------------------------------------------------
# Parsing benchmark_rounds.sh-teed output.
# ---------------------------------------------------------------------------

_BANNER_RE = re.compile(r"^={10,}$")
_SUBSECTION_RE = re.compile(r"^\s*----\s*(.+?)\s*----\s*$")
_HEADER_METRIC_RE = re.compile(r"(Host|Device|Effective|Orch|Sched) \(us\)")
_ROW_RE = re.compile(r"^\s*(\d+)((?:\s+-?[0-9.]+){1,5})\s*$")


@dataclass
class Example:
    label: str
    metrics: dict  # metric name -> list[float] (raw, one per round)


def parse_rounds_file(path: str) -> list[Example]:
    """Parse a benchmark_rounds.sh-teed file into per-example/case round samples."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    examples: list[Example] = []
    example_name = None
    subsection = None
    columns: list[str] | None = None
    in_table = False

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip("\n")
        if _BANNER_RE.match(line.strip()):
            # Banner-sandwiched title: "====", "  <name>", "====".
            if i + 1 < n and not _BANNER_RE.match(lines[i + 1].strip()):
                candidate = lines[i + 1].strip()
                if i + 2 < n and _BANNER_RE.match(lines[i + 2].strip()):
                    example_name = candidate
                    subsection = None
                    columns = None
                    in_table = False
                    i += 3
                    continue
            i += 1
            continue

        stripped = line.strip()
        if stripped and set(stripped) <= {"-"}:
            # Table header/data separator (all-dashes line).
            if columns is not None:
                in_table = True
            i += 1
            continue

        sub_match = _SUBSECTION_RE.match(line)
        if sub_match:
            subsection = sub_match.group(1)
            columns = None
            in_table = False
            i += 1
            continue

        if columns is None:
            found = _HEADER_METRIC_RE.findall(line)
            if found:
                columns = found
                in_table = False  # the dash separator line (above) starts rows
                i += 1
                continue

        if in_table:
            row_match = _ROW_RE.match(line)
            if row_match:
                if example_name is None:
                    i += 1
                    continue
                label = example_name if subsection is None else f"{example_name} [{subsection}]"
                ex = next((e for e in examples if e.label == label), None)
                if ex is None:
                    ex = Example(label=label, metrics={c: [] for c in columns})
                    examples.append(ex)
                values = row_match.group(2).split()
                for col, val in zip(columns, values):
                    ex.metrics.setdefault(col, []).append(float(val))
                i += 1
                continue
            # First non-row line after the table (an "Avg ..." line, blank, etc)
            # ends this table; keep columns so a same-named re-entrant table
            # (shouldn't normally happen) still appends, but stop consuming rows.
            in_table = False
            columns = None
            i += 1
            continue

        i += 1

    return examples


# ---------------------------------------------------------------------------
# Comparison + reporting.
# ---------------------------------------------------------------------------

_METRIC_ORDER = ("Host", "Device", "Effective", "Orch", "Sched")
_HEADLINE_METRIC = "Effective"


def compare(baseline: list[Example], current: list[Example], trim_pct: float, alpha: float) -> str:
    base_by_label = {e.label: e for e in baseline}
    cur_by_label = {e.label: e for e in current}
    labels = [label for label in base_by_label if label in cur_by_label]

    out = []
    improved = regressed = noise = 0

    for label in labels:
        be, ce = base_by_label[label], cur_by_label[label]
        metrics_here = [m for m in _METRIC_ORDER if m in be.metrics and m in ce.metrics]
        if not metrics_here:
            continue
        out.append(f"\n{label}")
        out.append("-" * len(label))
        for metric in metrics_here:
            a_raw, b_raw = be.metrics[metric], ce.metrics[metric]
            if not a_raw or not b_raw:
                continue
            a, b = trim(a_raw, trim_pct), trim(b_raw, trim_pct)
            ma, mb = sum(a) / len(a), sum(b) / len(b)
            delta = mb - ma
            pct = (delta / ma * 100.0) if ma else 0.0
            t, df, p = welch_t_test(a, b)
            d = cohens_d(a, b)
            sig = p < alpha
            verdict = ("REGRESSION" if delta > 0 else "IMPROVEMENT") if sig else "noise"
            if metric == _HEADLINE_METRIC:
                if sig and delta > 0:
                    regressed += 1
                elif sig and delta < 0:
                    improved += 1
                else:
                    noise += 1
            out.append(
                f"  {metric:<10} base={ma:>10.1f}us  cur={mb:>10.1f}us  "
                f"Δ={delta:>+9.1f}us ({pct:>+6.2f}%)  "
                f"p={p:.4f}  d={d:>+.2f}  n=({len(a)},{len(b)})  [{verdict}]"
            )

    summary = (
        f"\nOverall ({_HEADLINE_METRIC}, alpha={alpha}): "
        f"{improved} improved, {regressed} regressed, {noise} within noise "
        f"(of {improved + regressed + noise} examples)"
    )
    return "\n".join(out) + "\n" + summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", help="benchmark_rounds.sh-teed output file (baseline side)")
    parser.add_argument("current", help="benchmark_rounds.sh-teed output file (current side)")
    parser.add_argument(
        "--trim-pct",
        type=float,
        default=5.0,
        help="Percent of rounds dropped from each tail per side per metric before comparing (default: 5.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for the Welch's t-test p-value (default: 0.05)",
    )
    args = parser.parse_args()

    baseline = parse_rounds_file(args.baseline)
    current = parse_rounds_file(args.current)
    if not baseline or not current:
        print("No per-round tables found in one or both files (was --rounds-table output teed?).", file=sys.stderr)
        return 1

    print(compare(baseline, current, args.trim_pct, args.alpha))
    return 0


if __name__ == "__main__":
    sys.exit(main())
