"""Poster figures as inline SVG (vector, so resolution is never an issue at A1)."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3de"
COL = {"gpt4_1_mini": "#2a78d6", "claude_haiku_4_5": "#eb6834"}
NAME = {"gpt4_1_mini": "GPT-4.1 mini", "claude_haiku_4_5": "Claude Haiku 4.5"}
PNAME = {"simple": "simple prompt", "advanced": "objectivity prompt"}
MM = 1 / 25.4

plt.rcParams.update({
    "svg.fonttype": "none", "font.family": "Source Sans 3", "font.size": 24,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.2,
})


def _svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", transparent=True)
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg"):]


def dumbbell(groups, width_mm, height_mm=150):
    """groups: list of (title, desc) where desc maps (model, prompt) -> stats row."""
    slots = []                                     # (kind, payload)
    for title, desc in groups:
        slots.append(("head", title))
        for m in ["gpt4_1_mini", "claude_haiku_4_5"]:
            for p in ["simple", "advanced"]:
                if (m, p) in desc:
                    slots.append(("row", (m, p, desc[(m, p)])))
    fig, ax = plt.subplots(figsize=(width_mm * MM, height_mm * MM))
    fig.subplots_adjust(left=0.40, right=0.80, top=0.99, bottom=24 / height_mm)
    n = len(slots)
    ys = np.arange(n)[::-1].astype(float)
    ticks, labels, lows, highs = [], [], [], []
    for y, (kind, pl) in zip(ys, slots):
        if kind == "head":
            ax.text(-0.85, y - .05, pl, transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=24, fontweight="bold", color=INK)
            ax.axhline(y - .5, color=GRID, lw=1)
            continue
        m, p, r = pl
        c, j = r["d_ctrl"] * 100, r["d_judg"] * 100
        ax.plot([c, j], [y, y], color=GRID, lw=6, zorder=1, solid_capstyle="round")
        clo, chi = [v * 100 for v in r["d_ctrl_ci"]]
        ax.plot([clo, chi], [y + .17, y + .17], color=MUTED, lw=2.5)
        ax.scatter(c, y + .17, s=240, facecolor="white", edgecolor=MUTED, lw=3, zorder=3)
        lo, hi = [v * 100 for v in r["d_judg_ci"]]
        ax.plot([lo, hi], [y - .17, y - .17], color=COL[m], lw=2.5)
        ax.scatter(j, y - .17, s=280, color=COL[m], edgecolor="white", lw=2, zorder=3)
        star = "***" if r["eff_p"] < .001 else "**" if r["eff_p"] < .01 else "*" if r["eff_p"] < .05 else "n.s."
        ax.annotate(f"{r['eff'] * 100:+.0f} pp {star}".replace("-", "\u2212"), (1.03, y),
                    xycoords=("axes fraction", "data"), ha="left", va="center", fontsize=24, color=INK, fontweight="bold",
                    annotation_clip=False)
        ticks.append(y); labels.append(f"{NAME[m]} · {'simple' if p == 'simple' else 'objectivity'}")
        lows.append(min(lo, clo)); highs.append(max(hi, chi))
    ax.set_yticks(ticks, labels, fontsize=24, color=INK)
    ax.axvline(0, color=INK2, lw=1.4)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.set_xlim(min(min(lows) - 3, -30), max(max(highs) + 3, 10))
    ax.set_ylim(-.6, n - .4)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Δ P(true) after the swap, pp (95% CI)", fontsize=24)
    ax.spines["left"].set_visible(False)
    return _svg(fig)


def compass(reg, width_mm, height_mm):
    fig, ax = plt.subplots(figsize=(width_mm * MM, height_mm * MM))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
    pts = []
    for _, r in reg.iterrows():
        pass
    vals = []
    for m in ["gpt4_1_mini", "claude_haiku_4_5"]:
        for p, mk in [("simple", "o"), ("advanced", "s")]:
            e = reg[(reg.model == m) & (reg.prompt == p) & (reg.axis == "economic")]
            s = reg[(reg.model == m) & (reg.prompt == p) & (reg.axis == "social")]
            if e.empty or s.empty or "shift" not in reg or np.isnan(e["shift"].values[0]) or np.isnan(s["shift"].values[0]):
                continue
            x, y = e["shift"].values[0], s["shift"].values[0]
            xl, xh = e["shift_lo"].values[0], e["shift_hi"].values[0]
            yl, yh = s["shift_lo"].values[0], s["shift_hi"].values[0]
            if max(abs(xl), abs(xh), abs(yl), abs(yh)) > 5:   # non-converged fit
                continue
            pts.append((m, p, mk, x, y, xl, xh, yl, yh))
            vals += [xl, xh, yl, yh]
    lim = max(0.6, max(np.abs(vals)) * 1.12) if vals else 1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.fill_between([-lim, lim], -lim, lim, color="#f4f3ef", zorder=0)
    ax.axhline(0, color=INK2, lw=1.6, zorder=1); ax.axvline(0, color=INK2, lw=1.6, zorder=1)
    for a in np.arange(-2, 2.01, .25):
        if abs(a) < lim and abs(a) > 1e-9:
            ax.axhline(a, color=GRID, lw=.8, zorder=0); ax.axvline(a, color=GRID, lw=.8, zorder=0)
    kw = dict(fontsize=24, color=INK2, bbox=dict(facecolor="#f4f3ef", edgecolor="none", pad=1))
    ax.text(lim * .97, lim * .02, "right", ha="right", va="bottom", **kw)
    ax.text(-lim * .97, lim * .02, "left", ha="left", va="bottom", **kw)
    ax.text(lim * .02, lim * .97, "authoritarian", ha="left", va="top", **kw)
    ax.text(lim * .02, -lim * .97, "libertarian", ha="left", va="bottom", **kw)
    for m, p, mk, x, y, xl, xh, yl, yh in pts:
        ax.plot([xl, xh], [y, y], color=COL[m], lw=2.5, alpha=.7, zorder=2)
        ax.plot([x, x], [yl, yh], color=COL[m], lw=2.5, alpha=.7, zorder=2)
        ax.scatter(x, y, marker=mk, s=420, color=COL[m] if p == "simple" else "white",
                   edgecolor=COL[m], lw=3.5, zorder=3)
    return _svg(fig)
