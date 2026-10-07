# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""M0 component and latency diagram, light theme, as a standalone SVG."""

from pathlib import Path
from xml.sax.saxutils import escape as esc

W, H = 1600, 1180
INK, MUTED, LINE = "#1F2328", "#57606A", "#3A3F45"
REAL_F, REAL_S = "#EEF3F9", "#2F5D8A"
SIM_F, SIM_S = "#E8F4F1", "#1B6F69"
DISP, COMP = "#2F5D8A", "#9A3412"
FONT = "Helvetica, Arial, sans-serif"
out = []


def t(x, y, s, size=15, weight="normal", fill=INK, anchor="start", italic=False):
    st = ' font-style="italic"' if italic else ""
    out.append(
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}"{st}>{esc(s)}</text>'
    )


def rect(x, y, w, h, fill, stroke, sw=2, rx=10, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"'
        f' fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
    )


def pill(x_right, y_mid, label, stroke):
    w = 16 + 8.4 * len(label)
    out.append(
        f'<rect x="{x_right - w}" y="{y_mid - 12}" width="{w}" height="24"'
        f' rx="12" fill="#FFFFFF" stroke="{stroke}" stroke-width="1.5"/>'
    )
    t(x_right - w / 2, y_mid + 4, label, 12, "bold", stroke, "middle")


def arrow(x1, y1, x2, y2, color, sw=2.5):
    out.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}"'
        f' stroke-width="{sw}" marker-end="url(#ah-{color[1:]})"/>'
    )


out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
out.append("<defs>")
for c in (DISP, COMP, LINE):
    out.append(
        f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
    )
out.append("</defs>")
out.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

# header
t(60, 62, "M0: what aSim simulates, and the latencies it charges", 30, "bold")
t(
    60,
    94,
    "a2a3asim + host_build_graph. Where the two cases differ, values read paged_attention Case1 / qwen3-14B decode.",
    16,
    fill=MUTED,
)

# ---- AICPU (real) ----
ax, ay, aw, ah = 60, 140, 400, 444
rect(ax, ay, aw, ah, REAL_F, REAL_S)
t(ax + 20, ay + 36, "AICPU package", 21, "bold", REAL_S)
pill(ax + aw - 18, ay, "REAL", REAL_S)
t(ax + 20, ay + 66, "host_build_graph scheduler: real code,", 15)
t(ax + 20, ay + 87, "running at real AICPU speed", 15)
for i in range(4):
    bx = ax + 20 + (i % 2) * 185
    by = ay + 110 + (i // 2) * 84
    rect(bx, by, 175, 72, "#FFFFFF", REAL_S, 1.5, 8)
    t(bx + 12, by + 26, f"sched thread {i}", 15, "bold")
    t(bx + 12, by + 50, "6 packages = 18 cores", 14, fill=MUTED)
rect(ax + 20, ay + 290, aw - 40, 40, "#FFFFFF", REAL_S, 1.5, 8)
t(ax + aw / 2, ay + 316, "shared global ready queue (atomics)", 15, anchor="middle")
t(ax + 20, ay + 368, "Measured on paged_attention: ~1.15 us of", 14, fill=MUTED)
t(ax + 20, ay + 388, "scheduler software per task, per thread", 14, fill=MUTED)
t(ax + 20, ay + 408, "(~0.95 us with transport made free)", 14, fill=MUTED)

# ---- register window (simulated) ----
rx_, ry, rw, rh = 600, 190, 300, 320
rect(rx_, ry, rw, rh, SIM_F, SIM_S)
t(rx_ + 18, ry + 34, "Per-core registers", 20, "bold", SIM_S)
pill(rx_ + rw - 18, ry, "SIMULATED", SIM_S)
rect(rx_ + 18, ry + 60, rw - 36, 64, "#FFFFFF", SIM_S, 1.5, 8)
t(rx_ + 32, ry + 86, "DATA_MAIN_BASE", 15, "bold")
t(rx_ + 32, ry + 108, "task in", 14, fill=MUTED)
rect(rx_ + 18, ry + 140, rw - 36, 64, "#FFFFFF", SIM_S, 1.5, 8)
t(rx_ + 32, ry + 166, "COND", 15, "bold")
t(rx_ + 32, ry + 188, "ACK / FIN out", 14, fill=MUTED)
t(rx_ + 18, ry + 234, "asim_push and asim_read_status", 14, fill=MUTED)
t(rx_ + 18, ry + 254, "replace the MMIO store and nGnRE", 14, fill=MUTED)
t(rx_ + 18, ry + 274, "load; the COND word is bit-identical.", 14, fill=MUTED)

# ---- cores (simulated) ----
cx, cy, cw, chh = 1040, 140, 500, 444
rect(cx, cy, cw, chh, SIM_F, SIM_S)
t(cx + 20, cy + 36, "Compute cores", 21, "bold", SIM_S)
pill(cx + cw - 18, cy, "SIMULATED", SIM_S)
t(cx + 20, cy + 64, "24 packages x (1 AIC + 2 AIV) = 72 cores", 15)
# one package
px, py = cx + 20, cy + 86
rect(px, py, cw - 40, 120, "#FFFFFF", SIM_S, 1.5, 8, dash="5,4")
t(px + 12, py + 22, "one package", 13, fill=MUTED, italic=True)
for i, name in enumerate(("AIC", "AIV0", "AIV1")):
    bx = px + 14 + i * 148
    rect(bx, py + 34, 136, 74, SIM_F, SIM_S, 1.5, 8)
    t(bx + 68, py + 60, name, 16, "bold", anchor="middle")
    t(bx + 68, py + 84, "2-deep intake:", 13, fill=MUTED, anchor="middle")
    t(bx + 68, py + 100, "active + pushed", 13, fill=MUTED, anchor="middle")
# lifecycle chain
ly = cy + 262
states = ["landed", "noticed", "running", "FIN"]
sx = [cx + 16, cx + 142, cx + 268, cx + 394]
for x, s in zip(sx, states):
    rect(x, ly - 24, 84, 36, "#FFFFFF", SIM_S, 1.5, 18)
    t(x + 42, ly, s, 14, "bold", anchor="middle")
labels = ["entry poll 318 ns", "pick-up 491 / 581", "compute"]
for i, lab in enumerate(labels):
    x1, x2 = sx[i] + 84, sx[i + 1]
    arrow(x1 + 2, ly - 6, x2 - 3, ly - 6, LINE, 1.8)
    t((x1 + x2) / 2, ly + 30, lab, 13, anchor="middle")
t(cx + 20, ly + 66, "Compute is drawn per kernel from the calibration:", 14)
t(cx + 20, ly + 88, "paged_attention 4 kernels, 1.20-1.64 us;", 14, fill=MUTED)
t(cx + 20, ly + 108, "qwen 36 kernels, 0.08-270 us, median 19.5 us.", 14, fill=MUTED)
t(cx + 20, ly + 136, "A pushed task waiting in the intake costs 0 ns:", 14)
t(cx + 20, ly + 156, "it starts as the active one ends.", 14, fill=MUTED)

# ---- arrows between boxes ----
yd, yc = 300, 445
arrow(ax + aw + 4, yd, rx_ - 6, yd, DISP)
t((ax + aw + rx_) / 2, yd - 30, "push", 15, "bold", DISP, "middle")
t((ax + aw + rx_) / 2, yd + 22, "5 ns AICPU", 13, fill=DISP, anchor="middle")
t((ax + aw + rx_) / 2, yd + 38, "occupancy", 13, fill=DISP, anchor="middle")
arrow(rx_ + rw + 4, yd, cx - 6, yd, DISP)
t((rx_ + rw + cx) / 2, yd - 30, "arrives", 15, "bold", DISP, "middle")
t((rx_ + rw + cx) / 2, yd - 12, "+310 ns", 13, fill=DISP, anchor="middle")
arrow(cx - 4, yc, rx_ + rw + 6, yc, COMP)
t((rx_ + rw + cx) / 2, yc + 24, "FIN visible", 15, "bold", COMP, "middle")
t((rx_ + rw + cx) / 2, yc + 42, "after 80 / 140 ns", 13, fill=COMP, anchor="middle")
arrow(rx_ - 4, yc, ax + aw + 6, yc, COMP)
t((ax + aw + rx_) / 2, yc + 24, "COND read", 15, "bold", COMP, "middle")
t((ax + aw + rx_) / 2, yc + 42, "195 ns per core,", 13, fill=COMP, anchor="middle")
t((ax + aw + rx_) / 2, yc + 58, "strictly serial", 13, fill=COMP, anchor="middle")
t((ax + aw + rx_) / 2, yc + 76, "(raw LDR: 92 ns)", 12, fill=MUTED, anchor="middle")

# ---- proportional lifecycle bars ----
by0 = 640
t(60, by0, "Where one task's time goes, idle core, median kernel", 22, "bold")
t(
    60,
    by0 + 28,
    "Each bar is 100 % of the span from the push to the scheduler"
    " reading the FIN; segments are to scale within each bar.",
    15,
    fill=MUTED,
)
segs = [
    ("push", 5, 5, "#2F5D8A"),
    ("arrival", 310, 310, "#7FA3C7"),
    ("entry poll", 318, 318, "#A9C3DC"),
    ("pick-up", 491, 581, "#C9D9E8"),
    ("compute", 1514, 19478, "#5E9C76"),
    ("FIN notice", 80, 140, "#E3A587"),
    ("COND read", 195, 195, "#9A3412"),
]
bx0, bw = 250, 1290
for row, (case, col) in enumerate((("paged_attention", 1), ("qwen 40L", 2))):
    yb = by0 + 66 + row * 92
    total = sum(s[col] for s in segs)
    over = total - segs[4][col]
    t(60, yb + 30, case, 17, "bold")
    x = bx0
    for name, a, b, color in segs:
        w = bw * (a if col == 1 else b) / total
        out.append(
            f'<rect x="{x:.2f}" y="{yb}" width="{max(w, 0.6):.2f}"'
            f' height="44" fill="{color}" stroke="#FFFFFF" stroke-width="1"/>'
        )
        v = a if col == 1 else b
        if len(f"{name} {v:,}") * 13 * 0.62 < w - 8:
            fc = "#FFFFFF" if name in ("compute", "push", "COND read") else INK
            t(x + w / 2, yb + 27, f"{name} {v:,}", 13, "bold", fc, "middle")
        x += w
    out.append(f'<rect x="{bx0}" y="{yb}" width="{bw}" height="44" fill="none" stroke="{LINE}" stroke-width="1.5"/>')
    t(
        bx0 + bw,
        yb + 66,
        f"{total:,} ns in all; everything but compute is {over / total * 100:.0f} % of it",
        14,
        "bold",
        INK,
        "end",
    )

# legend table
ty = by0 + 262
t(60, ty, "segment", 14, "bold", MUTED)
t(250, ty, "paged_attention", 14, "bold", MUTED)
t(400, ty, "qwen", 14, "bold", MUTED)
t(520, ty, "what it is", 14, "bold", MUTED)
desc = {
    "push": "AICPU posts DATA_MAIN_BASE; the only AICPU occupancy on dispatch",
    "arrival": "the store crosses the die to the core",
    "entry poll": "the core's own spin loop notices the task",
    "pick-up": "dcci + ack before the kernel body",
    "compute": "calibrated kernel duration (median shown)",
    "FIN notice": "the core's FIN becomes visible to the AICPU",
    "COND read": "AICPU reads that core's status; one core per read",
}
for i, (name, a, b, color) in enumerate(segs):
    yy = ty + 26 + i * 22
    out.append(f'<rect x="60" y="{yy - 12}" width="14" height="14" fill="{color}"/>')
    t(84, yy, name, 14)
    t(250, yy, f"{a:,} ns", 14)
    t(400, yy, f"{b:,} ns", 14)
    t(520, yy, desc[name], 14, fill=MUTED)

# footer
fy = H - 28
t(
    60,
    fy - 22,
    "Latencies from microbenchmarks and in-situ calibration"
    " (docs/asim/calibration.md); read = 195 ns is the model's one fitted parameter.",
    13,
    fill=MUTED,
)
t(
    60,
    fy,
    "M0 against real silicon, device_wall: +2.5 to +3.6 % on paged_attention,"
    " -4.2 to -4.4 % on qwen (two sessions, 2026-10-07, model clock).",
    13,
    fill=MUTED,
)
out.append("</svg>")
Path(__file__).with_name("m0-latencies.svg").write_text("\n".join(out) + "\n")
print("written")
