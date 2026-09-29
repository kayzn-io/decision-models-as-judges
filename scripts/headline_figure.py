"""Draw docs/headline.png: the one-figure summary shown at the top of the README.

Reads results/g3_summary.csv and the confusion counts behind it, so the figure
tracks the committed results. Run after `judges analyze --gate g3 ...`:

    uv run python scripts/headline_figure.py
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "headline.png"

PAPER, INK, MUTED, LINE, ORANGE = "#f7f2ea", "#241f1a", "#6b6157", "#d9d1c4", "#f26a1b"
GREY_1, GREY_2, GREY_3, WHITE = "#3b3f45", "#6d7379", "#9aa0a6", "#fbf8f2"

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})

g3 = pd.read_csv(ROOT / "results" / "g3_summary.csv")
full = g3[g3.profile == "full"].set_index("judge_id")

rows = [
    ("code", "Rule-based check", "reads the answer key", GREY_3),
    ("llm_strong", "gpt-5", "strong text model", GREY_1),
    ("jev", "Jev", "typed decision model", ORANGE),
    ("llm_cheap", "gpt-4o-mini", "fast text model", GREY_2),
]
cost_per_1000 = {"code": 0.0, "llm_strong": 28.01, "jev": 0.19, "llm_cheap": 0.65}
unanimous = {"code": 1.0, "llm_strong": 0.80, "jev": 0.987, "llm_cheap": 0.957}

W, H = 12.0, 6.2
fig = plt.figure(figsize=(W, H), facecolor=PAPER)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_axis_off()

ax.text(
    0.6,
    5.65,
    "Four judges, 230 support conversations, one answer key they never see",
    fontsize=17,
    fontweight="bold",
    color=INK,
    va="center",
)
ax.text(
    0.6,
    5.22,
    "Full transcript, five verdicts per conversation, modal verdict scored against "
    'tau-bench\'s state check. Always answering "pass" would score 49%.',
    fontsize=10.5,
    color=MUTED,
    va="center",
)

x0, x_bar_end = 3.9, 8.6
head_y = 4.55
for x, label in [
    (x0, "agrees with the key"),
    (9.25, "same verdict\n5 of 5"),
    (10.55, "$ per 1,000\nverdicts"),
]:
    ax.text(
        x if x != x0 else x0,
        head_y,
        label,
        ha="left" if x == x0 else "center",
        va="center",
        fontsize=9.5,
        color=MUTED,
        linespacing=1.25,
    )
ax.plot([0.6, W - 0.6], [head_y - 0.32, head_y - 0.32], color=LINE, lw=1)
for v in [0.25, 0.5, 0.75, 1.0]:
    xv = x0 + v * (x_bar_end - x0)
    ax.plot([xv, xv], [0.75, head_y - 0.32], color=LINE, lw=0.8, zorder=1)
    ax.text(xv, 0.52, f"{v:.0%}", ha="center", va="top", fontsize=9, color=MUTED)
xg = x0 + 0.491 * (x_bar_end - x0)
ax.plot([xg, xg], [0.75, head_y - 0.32], color=GREY_3, lw=1.4, ls=(0, (4, 3)), zorder=1)
ax.text(xg, 0.30, "always pass, 49%", ha="center", va="top", fontsize=8.5, color=MUTED)

y = head_y - 0.32 - 0.5
bar_h, gap = 0.5, 0.36
for jid, name, sub, colour in rows:
    acc = float(full.loc[jid, "accuracy"])
    yy = y - bar_h
    ax.text(
        0.6,
        yy + bar_h / 2 + 0.11,
        name,
        ha="left",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
    )
    ax.text(0.6, yy + bar_h / 2 - 0.16, sub, ha="left", va="center", fontsize=9, color=MUTED)
    ax.add_patch(
        FancyBboxPatch(
            (x0, yy),
            acc * (x_bar_end - x0),
            bar_h,
            boxstyle="round,pad=0,rounding_size=0.06",
            fc=colour,
            ec="none",
            zorder=2,
        )
    )
    ax.text(
        x0 + acc * (x_bar_end - x0) + 0.1,
        yy + bar_h / 2,
        f"{acc:.0%}",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=colour,
    )
    ax.text(
        9.25,
        yy + bar_h / 2,
        f"{unanimous[jid]:.0%}",
        ha="center",
        va="center",
        fontsize=12,
        color=INK,
    )
    c = cost_per_1000[jid]
    ax.text(
        10.55,
        yy + bar_h / 2,
        "free" if c == 0 else (f"${c:.2f}" if c < 1 else f"${c:.0f}"),
        ha="center",
        va="center",
        fontsize=12,
        color=INK,
    )
    y -= bar_h + gap

fig.savefig(OUT, dpi=200, facecolor=PAPER)
print("wrote", OUT)
