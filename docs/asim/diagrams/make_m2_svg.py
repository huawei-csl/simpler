# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""M2 (GroupQueue) component and latency diagram, light theme, standalone SVG."""

from pathlib import Path
from xml.sax.saxutils import escape as esc

W, H = 1600, 1310
INK, MUTED, LINE = "#1F2328", "#57606A", "#3A3F45"
REAL_F, REAL_S = "#EEF3F9", "#2F5D8A"
SIM_F, SIM_S = "#E8F4F1", "#1B6F69"
GQ_F, GQ_S = "#F3EFF9", "#5B3F8C"
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
for c in (DISP, COMP, LINE, GQ_S, SIM_S):
    out.append(
        f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
    )
out.append("</defs>")
out.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

t(60, 62, "M2: the GroupQueue aSim simulates, and the latencies it charges", 30, "bold")
t(
    60,
    94,
    "a2a3asimgq + group_queue. Where the two cases differ, values read paged_attention Case1 / qwen3-14B decode.",
    16,
    fill=MUTED,
)

# ---- AICPU (real managers + simulated status registers) ----
ax, ay, aw, ah = 60, 140, 420, 470
rect(ax, ay, aw, ah, REAL_F, REAL_S)
t(ax + 20, ay + 36, "AICPU package", 21, "bold", REAL_S)
pill(ax + aw - 18, ay, "REAL", REAL_S)
t(ax + 20, ay + 66, "group_queue managers: real code,", 15)
t(ax + 20, ay + 87, "running at real AICPU speed", 15)
for i in range(4):
    bx = ax + 20 + (i % 2) * 195
    by = ay + 108 + (i // 2) * 76
    rect(bx, by, 185, 64, "#FFFFFF", REAL_S, 1.5, 8)
    t(bx + 12, by + 25, f"manager {i}", 15, "bold")
    t(bx + 12, by + 47, f"owns queue {i} only", 14, fill=MUTED)
sx_, sy_ = ax + 20, ay + 278
rect(sx_, sy_, aw - 40, 74, SIM_F, SIM_S, 1.5, 8, dash="5,4")
t(sx_ + 12, sy_ + 25, "status register x4", 15, "bold", SIM_S)
t(sx_ + 12, sy_ + 47, "watermark + 32 look-ahead bits,", 14, fill=MUTED)
t(sx_ + 12, sy_ + 65, "one word per queue (simulated)", 14, fill=MUTED)
arrow(ax + aw / 2, sy_ - 2, ax + aw / 2, ay + 260, SIM_S, 2)
t(ax + aw / 2 + 10, ay + 270, "read: 10 ns, one word", 13, "bold", SIM_S)
t(ax + 20, ay + 380, "No shared ready queue, nothing atomic", 14, fill=MUTED)
t(ax + 20, ay + 400, "between managers.", 14, fill=MUTED)
t(ax + 20, ay + 428, "Measured on paged_attention: ~570 ns of", 14, fill=MUTED)
t(ax + 20, ay + 448, "manager wall per task, per thread", 14, fill=MUTED)

# ---- GroupQueue controller (simulated new hardware) ----
gx, gy, gw, gh = 600, 140, 400, 470
rect(gx, gy, gw, gh, GQ_F, GQ_S)
t(gx + 20, gy + 36, "GroupQueue controller x4", 21, "bold", GQ_S)
pill(gx + gw - 18, gy, "SIMULATED", GQ_S)
t(gx + 20, gy + 64, "one per manager, 6 packages each", 15)
rows = [
    ("ready rings", "cube, vector and mix, each pushed independently"),
    ("hold array", "32 slots x 4 dependency comparators"),
    ("watermark + look-ahead", "1,024 positions; 32 bits published"),
    ("placement", "idle core first, else behind a busy one"),
    ("steal", "cancel a pipelined task, 30 ns each way"),
]
for k, (head, sub) in enumerate(rows):
    ry = gy + 86 + k * 74
    rect(gx + 20, ry, gw - 40, 62, "#FFFFFF", GQ_S, 1.5, 8)
    t(gx + 34, ry + 25, head, 15, "bold")
    t(gx + 34, ry + 47, sub, 13, fill=MUTED)

# ---- cores (simulated, same model as M0) ----
cx, cy, cw, chh = 1120, 140, 420, 470
rect(cx, cy, cw, chh, SIM_F, SIM_S)
t(cx + 20, cy + 36, "Compute cores", 21, "bold", SIM_S)
pill(cx + cw - 18, cy, "SIMULATED", SIM_S)
t(cx + 20, cy + 64, "24 packages x (1 AIC + 2 AIV) = 72,", 15)
t(cx + 20, cy + 85, "18 per controller", 15)
px, py = cx + 20, cy + 104
rect(px, py, cw - 40, 116, "#FFFFFF", SIM_S, 1.5, 8, dash="5,4")
t(px + 12, py + 22, "one package", 13, fill=MUTED, italic=True)
for i, name in enumerate(("AIC", "AIV0", "AIV1")):
    bx = px + 12 + i * 122
    rect(bx, py + 32, 112, 72, SIM_F, SIM_S, 1.5, 8)
    t(bx + 56, py + 58, name, 16, "bold", anchor="middle")
    t(bx + 56, py + 80, "2-deep", 13, fill=MUTED, anchor="middle")
    t(bx + 56, py + 96, "intake", 13, fill=MUTED, anchor="middle")
ly = cy + 268
states = ["landed", "noticed", "running", "FIN"]
sx = [cx + 14, cx + 116, cx + 218, cx + 320]
for x, s in zip(sx, states):
    rect(x, ly - 24, 84, 36, "#FFFFFF", SIM_S, 1.5, 18)
    t(x + 42, ly, s, 14, "bold", anchor="middle")
for i, lab in enumerate(["poll 318 ns", "pick-up 491/581", "compute"]):
    x1, x2 = sx[i] + 84, sx[i + 1]
    arrow(x1 + 2, ly - 6, x2 - 3, ly - 6, LINE, 1.8)
    t((x1 + x2) / 2, ly + 30 + (16 if i == 1 else 0), lab, 13, anchor="middle")
t(cx + 20, ly + 80, "The same core model as M0: compute drawn", 14)
t(cx + 20, ly + 100, "per kernel from the calibration.", 14)
t(cx + 20, ly + 128, "Pick-up here is M0's calibrated ack less", 14, fill=MUTED)
t(cx + 20, ly + 148, "the 628 ns of AICPU-to-core travel a", 14, fill=MUTED)
t(cx + 20, ly + 168, "controller's adjacent push does not make.", 14, fill=MUTED)

# ---- arrows ----
yd, yc = 300, 500
mid1 = (ax + aw + gx) / 2
mid2 = (gx + gw + cx) / 2
arrow(ax + aw + 4, yd, gx - 6, yd, DISP)
t(mid1, yd - 30, "submit", 15, "bold", DISP, "middle")
t(mid1, yd - 12, "5 ns occupancy", 13, fill=DISP, anchor="middle")
t(mid1, yd + 24, "arrives +310 ns", 13, fill=DISP, anchor="middle")
arrow(gx + gw + 4, yd, cx - 6, yd, DISP)
t(mid2, yd - 30, "push", 15, "bold", DISP, "middle")
t(mid2, yd - 12, "30 ns link", 13, fill=DISP, anchor="middle")
t(mid2, yd + 24, "0 if pipelined", 13, fill=DISP, anchor="middle")
arrow(cx - 4, yc, gx + gw + 6, yc, COMP)
t(mid2, yc + 24, "FIN raised", 15, "bold", COMP, "middle")
t(mid2, yc + 42, "80 / 140 ns", 13, fill=COMP, anchor="middle")
t(mid2, yc + 58, "+ 30 ns link", 13, fill=COMP, anchor="middle")
arrow(gx - 4, yc, ax + aw + 6, yc, COMP)
t(mid1, yc + 24, "update", 15, "bold", COMP, "middle")
t(mid1, yc + 42, "80 / 140 ns", 13, fill=COMP, anchor="middle")
t(mid1, yc + 58, "across the die", 13, fill=COMP, anchor="middle")

# ---- section A: one task's span ----
by0 = 680
t(60, by0, "One task's span on an idle core, median kernel", 22, "bold")
t(
    60,
    by0 + 28,
    "Each bar is 100 % of the span from the submit to the manager"
    " reading the retirement; segments are to scale within each bar.",
    15,
    fill=MUTED,
)
segs = [
    ("submit", 5, 5, "#2F5D8A"),
    ("arrival", 310, 310, "#7FA3C7"),
    ("link", 30, 30, "#B9A6D6"),
    ("entry poll", 318, 318, "#A9C3DC"),
    ("pick-up", 491, 581, "#C9D9E8"),
    ("compute", 1514, 19478, "#5E9C76"),
    ("FIN raised", 80, 140, "#E3A587"),
    ("link ", 30, 30, "#B9A6D6"),
    ("update", 80, 140, "#D9825B"),
    ("status read", 10, 10, "#9A3412"),
]
m0_total = {1: 2913, 2: 21027}
bx0, bw = 250, 1290
for row, (case, col) in enumerate((("paged_attention", 1), ("qwen 40L", 2))):
    yb = by0 + 64 + row * 92
    total = sum(s[col] for s in segs)
    t(60, yb + 30, case, 17, "bold")
    x = bx0
    for name, a, b, color in segs:
        v = a if col == 1 else b
        w = bw * v / total
        out.append(
            f'<rect x="{x:.2f}" y="{yb}" width="{max(w, 0.6):.2f}"'
            f' height="44" fill="{color}" stroke="#FFFFFF" stroke-width="1"/>'
        )
        if len(f"{name.strip()} {v:,}") * 13 * 0.62 < w - 8:
            fc = "#FFFFFF" if name in ("compute", "submit", "status read") else INK
            t(x + w / 2, yb + 27, f"{name.strip()} {v:,}", 13, "bold", fc, "middle")
        x += w
    out.append(f'<rect x="{bx0}" y="{yb}" width="{bw}" height="44" fill="none" stroke="{LINE}" stroke-width="1.5"/>')
    t(bx0 + bw, yb + 66, f"{total:,} ns; M0's span is {m0_total[col]:,} ns", 14, "bold", INK, "end")
t(
    60,
    by0 + 268,
    "M2 does not win on a task's latency: its dispatch crosses one"
    " more link than M0's. It wins on what the AICPU pays per task.",
    15,
    "bold",
    GQ_S,
)

# ---- section B: legend (left) and AICPU cost per task (right) ----
ty = by0 + 316
t(60, ty, "segment", 14, "bold", MUTED)
t(210, ty, "PA", 14, "bold", MUTED)
t(290, ty, "qwen", 14, "bold", MUTED)
t(380, ty, "what it is", 14, "bold", MUTED)
desc = {
    "submit": "manager's posted write into its queue",
    "arrival": "the entry becomes legible to the controller",
    "link": "controller to core, one way",
    "entry poll": "the core's own loop notices the task",
    "pick-up": "dcci + ack before the kernel body",
    "compute": "calibrated kernel duration (median)",
    "FIN raised": "what raising a FIN costs the core",
    "link ": "core back to its controller",
    "update": "controller writes the AICPU-side register",
    "status read": "one word: watermark + look-ahead",
}
for i, (name, a, b, color) in enumerate(segs):
    yy = ty + 26 + i * 22
    out.append(f'<rect x="60" y="{yy - 12}" width="14" height="14" fill="{color}"/>')
    t(84, yy, name.strip(), 14)
    t(210, yy, f"{a:,}", 14)
    t(290, yy, f"{b:,}", 14)
    t(380, yy, desc[name], 14, fill=MUTED)

cx2, cw2 = 900, 640
t(cx2, ty, "What the AICPU pays per task", 17, "bold")
t(cx2, ty + 22, "paged_attention, wall per task per scheduler thread, measured", 13, fill=MUTED)
bars = [("M0", 1145, 255, REAL_S), ("M0, free transport", 950, 0, "#7FA3C7"), ("M2", 570, 16, GQ_S)]
scale = (cw2 - 170) / 1200
for i, (lab, v, tr, col) in enumerate(bars):
    yb = ty + 50 + i * 62
    t(cx2, yb + 24, lab, 14, "bold")
    x0 = cx2 + 160
    out.append(f'<rect x="{x0}" y="{yb}" width="{v * scale:.1f}" height="34" fill="{col}"/>')
    if tr:
        out.append(
            f'<rect x="{x0 + (v - tr) * scale:.1f}" y="{yb}" width="{max(tr * scale, 1.5):.1f}"'
            f' height="34" fill="#FFFFFF" fill-opacity="0.55" stroke="{col}" stroke-width="1"/>'
        )
    t(x0 + v * scale + 8, yb + 23, f"{v:,} ns", 14, "bold")
t(cx2, ty + 250, "Light end of each bar: modelled transport the AICPU is held for --", 13, fill=MUTED)
t(cx2, ty + 268, "M0 ~255 ns (a push + 1.3 core reads at 195 ns), M2 ~16 ns", 13, fill=MUTED)
t(cx2, ty + 286, "(a submit + 1.1 register reads at 10 ns).", 13, fill=MUTED)

# footer
fy = H - 28
t(
    60,
    fy - 22,
    "Latencies from the GroupQueue model (asimgq_core.h) and the shared"
    " calibration; all figures on the model clock, measured 2026-10-07.",
    13,
    fill=MUTED,
)
t(
    60,
    fy,
    "M2 against M0, device_wall: -47.9 % on paged_attention, +1.2"
    " % on qwen (parity: qwen is bound by its cube cores' compute).",
    13,
    fill=MUTED,
)
out.append("</svg>")
Path(__file__).with_name("m2-latencies.svg").write_text("\n".join(out) + "\n")
print("written")
