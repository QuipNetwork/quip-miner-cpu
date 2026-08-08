"""Render the kernel comparison as two stacked Tufte-style panels.

Panel A: cost. Mean wall-clock per sample against problem size.
Panel B: quality. Paired energy advantage over cpu-sa, with +/-1 standard
error. Paired because every kernel saw the identical 30 instances at each
size, which removes instance-to-instance variance from the comparison.

No gridlines, no plot frames, no legend. Series are labelled at their right
end. The Advantage2-System1 pivot is a single rule with its label in a strip
above panel A, clear of the data.
"""

import json
import statistics as st

rows = json.load(open("results.json"))["rows"]
sizes = sorted({r["nodes"] for r in rows})
by = {(r["kernel"], r["nodes"]): r for r in rows}

KERNELS = ["cpu-sa", "cpu-gibbs", "cpu-sb"]
COLOR = {"cpu-sa": "#555555", "cpu-gibbs": "#aaaaaa", "cpu-sb": "#1f6fb4"}
WIDTH = {"cpu-sa": 1.6, "cpu-gibbs": 1.6, "cpu-sb": 2.4}
PIVOT = 4577

# Core-adjusted cost: wall-clock times the cores the kernel occupies. That is
# what a miner pays when it fills the machine with models.
# Throughput one core delivers. Wall-clock alone rewards a kernel for spending
# more cores, so rank on this instead.
time_ms = {
    k: [1000.0 / (by[(k, n)]["mean_time_ms"] * by[(k, n)]["cores"]) for n in sizes]
    for k in KERNELS
}
# A row measured under heavy load is drawn hollow rather than dropped.
LOAD_LIMIT = 20.0
loaded = {k: [by[(k, n)]["load_avg"] > LOAD_LIMIT for n in sizes] for k in KERNELS}

rel, err = {}, {}
for k in ("cpu-gibbs", "cpu-sb"):
    means, ses = [], []
    for n in sizes:
        sa = by[("cpu-sa", n)]["per_sample_best_milli"]
        other = by[(k, n)]["per_sample_best_milli"]
        d = [(o - a) / abs(a) * 100 for a, o in zip(sa, other)]
        means.append(st.mean(d))
        ses.append(st.stdev(d) / len(d) ** 0.5)
    rel[k], err[k] = means, ses

# Geometry
W, ML, MR, MT = 940, 92, 138, 118
PH, GAP, MB = 250, 74, 62
H = MT + PH + GAP + PH + MB

x0, x1 = min(sizes), max(sizes)
def sx(n):
    return ML + (n - x0) / (x1 - x0) * (W - ML - MR)

A_TOP, A_BOT = MT, MT + PH
a_max = 3.7
def say(v):
    return A_BOT - v / a_max * PH

B_TOP = MT + PH + GAP
B_BOT = B_TOP + PH
b_lo, b_hi = -0.26, 0.50
def sby(v):
    return B_BOT - (v - b_lo) / (b_hi - b_lo) * PH

# Each right-edge label is a two-line stack (name over value), so it needs
# more vertical room than a single line of text.
LABEL_H = 30

def spread(items):
    """Nudge overlapping right-edge labels apart, preserving order."""
    items = sorted(items, key=lambda t: t[0])
    for i in range(1, len(items)):
        if items[i][0] - items[i - 1][0] < LABEL_H:
            items[i] = (items[i - 1][0] + LABEL_H, items[i][1], items[i][2])
    return items

s = []
add = s.append
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

add(f'<text x="{ML}" y="26" font-size="15" font-weight="600" fill="#111">'
    'CPU Ising kernels on the Advantage2-System1 topology</text>')
add(f'<text x="{ML}" y="44" font-size="11" fill="#666">'
    'Zero biases, couplings drawn from {-1, +1}. 16 reads x 1000 sweeps. 30 instances per size, shared across kernels.</text>')
add(f'<text x="{ML}" y="60" font-size="11" fill="#666">'
    'cpu-gibbs occupies 4 cores per sample, the others 1. Hollow markers were measured at a 1-minute load above 20.</text>')

# Pivot rule, label in the strip above panel A
add(f'<line x1="{sx(PIVOT):.1f}" y1="{A_TOP - 36}" x2="{sx(PIVOT):.1f}" y2="{B_BOT}" '
    'stroke="#d8b98a" stroke-width="1"/>')
add(f'<text x="{sx(PIVOT):.1f}" y="{A_TOP - 42}" font-size="10.5" fill="#9a7a45" '
    'text-anchor="middle">pivot: 4577 nodes, 41515 edges</text>')

# ---------- Panel A ----------
add(f'<text x="{ML}" y="{A_TOP - 16}" font-size="12" font-weight="600" fill="#111">'
    'Throughput: models per second per core, so higher is better</text>')
add(f'<line x1="{ML}" y1="{say(0)}" x2="{ML}" y2="{say(3.5)}" stroke="#111" stroke-width="1"/>')
for v in (0, 1.0, 2.0, 3.0, 3.5):
    add(f'<text x="{ML - 8}" y="{say(v) + 3.5:.1f}" font-size="10" fill="#444" '
        f'text-anchor="end">{v:.1f}</text>')
add(f'<text x="{ML - 8}" y="{say(3.5) - 12:.1f}" font-size="10" fill="#444" '
    'text-anchor="end">per core</text>')

labels_a = []
for k in KERNELS:
    pts = " ".join(f"{sx(n):.1f},{say(v):.1f}" for n, v in zip(sizes, time_ms[k]))
    add(f'<polyline points="{pts}" fill="none" stroke="{COLOR[k]}" '
        f'stroke-width="{WIDTH[k]}"/>')
    for n, v, hot in zip(sizes, time_ms[k], loaded[k]):
        if hot:
            add(f'<circle cx="{sx(n):.1f}" cy="{say(v):.1f}" r="3.2" fill="#fff" '
                f'stroke="{COLOR[k]}" stroke-width="1.4"/>')
        else:
            add(f'<circle cx="{sx(n):.1f}" cy="{say(v):.1f}" r="2.4" fill="{COLOR[k]}"/>')
    labels_a.append((say(time_ms[k][-1]), k, f"{time_ms[k][-1]:.2f}/s/core"))
for y, k, val in spread(labels_a):
    add(f'<text x="{sx(x1) + 10}" y="{y + 3.5:.1f}" font-size="11" fill="{COLOR[k]}" '
        f'font-weight="600">{k}</text>')
    add(f'<text x="{sx(x1) + 10}" y="{y + 16:.1f}" font-size="9.5" fill="#777">{val}</text>')

# ---------- Panel B ----------
add(f'<text x="{ML}" y="{B_TOP - 30}" font-size="12" font-weight="600" fill="#111">'
    'Quality: paired energy difference from cpu-sa on the same instances</text>')
add(f'<text x="{ML}" y="{B_TOP - 15}" font-size="10.5" fill="#666">'
    'below the line is lower energy, which is better. Whiskers are +/-1 standard error over 30 paired instances.</text>')

add(f'<line x1="{ML}" y1="{sby(b_hi)}" x2="{ML}" y2="{sby(b_lo)}" stroke="#111" stroke-width="1"/>')
for v in (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4):
    add(f'<text x="{ML - 8}" y="{sby(v) + 3.5:.1f}" font-size="10" fill="#444" '
        f'text-anchor="end">{v:+.1f}</text>')
add(f'<text x="{ML - 8}" y="{sby(b_hi) - 12:.1f}" font-size="10" fill="#444" '
    'text-anchor="end">%</text>')

# cpu-sa baseline at zero, labelled outside the plot
add(f'<line x1="{ML}" y1="{sby(0):.1f}" x2="{sx(x1):.1f}" y2="{sby(0):.1f}" '
    'stroke="#555" stroke-width="1" stroke-dasharray="4 3"/>')

labels_b = [(sby(0), "cpu-sa", "baseline")]
for k in ("cpu-gibbs", "cpu-sb"):
    pts = " ".join(f"{sx(n):.1f},{sby(v):.1f}" for n, v in zip(sizes, rel[k]))
    add(f'<polyline points="{pts}" fill="none" stroke="{COLOR[k]}" '
        f'stroke-width="{WIDTH[k]}"/>')
    for n, v, e in zip(sizes, rel[k], err[k]):
        add(f'<line x1="{sx(n):.1f}" y1="{sby(v - e):.1f}" x2="{sx(n):.1f}" '
            f'y2="{sby(v + e):.1f}" stroke="{COLOR[k]}" stroke-width="1"/>')
        add(f'<circle cx="{sx(n):.1f}" cy="{sby(v):.1f}" r="2.4" fill="{COLOR[k]}"/>')
    labels_b.append((sby(rel[k][-1]), k, f"{rel[k][-1]:+.2f}%"))
for y, k, val in spread(labels_b):
    add(f'<text x="{sx(x1) + 10}" y="{y + 3.5:.1f}" font-size="11" fill="{COLOR[k]}" '
        f'font-weight="600">{k}</text>')
    add(f'<text x="{sx(x1) + 10}" y="{y + 16:.1f}" font-size="9.5" fill="#777">{val}</text>')

# ---------- Shared x axis ----------
add(f'<line x1="{ML}" y1="{B_BOT + 1}" x2="{sx(x1):.1f}" y2="{B_BOT + 1}" '
    'stroke="#111" stroke-width="1"/>')
for n in sizes:
    e = by[("cpu-sa", n)]["edges"]
    add(f'<text x="{sx(n):.1f}" y="{B_BOT + 16}" font-size="10" fill="#444" '
        f'text-anchor="middle">{n}</text>')
    add(f'<text x="{sx(n):.1f}" y="{B_BOT + 28}" font-size="8.5" fill="#999" '
        f'text-anchor="middle">{e // 1000}k</text>')
add(f'<text x="{(ML + sx(x1)) / 2:.1f}" y="{B_BOT + 47}" font-size="11" fill="#333" '
    'text-anchor="middle">nodes (edges below)</text>')

add("</svg>")
open("kernel_comparison.svg", "w").write("\n".join(s))
print("wrote kernel_comparison.svg")
