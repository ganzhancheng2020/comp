"""Figures for the report (static, light surface; palette and mark specs from the dataviz reference instance)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tfb.data import load, network
from tfb.features import plateau_of

OUT = Path(__file__).resolve().parent
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3 (validated all-pairs)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.dpi": 200})


def fig_fd():
    p, l = "D7_I10_E", 5
    z = load(p, "train"); net = network(p)
    v, q = z["speed"][:, :, l].ravel(), z["flow"][:, :, l].ravel()
    m = np.isfinite(v) & np.isfinite(q)
    rng = np.random.default_rng(0); k = rng.choice(np.where(m)[0], 15000, replace=False)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.scatter(q[k] / net.capacity_vph[l], v[k] / net.free_speed_kmh[l], s=2, c=S1, alpha=0.25, linewidths=0)
    ax.set_xlabel("flow / capacity"); ax.set_ylabel("speed / free-flow speed")
    ax.set_title("Two flat plateaus", fontsize=8.5, color=INK, loc="left")
    ax.text(0.30, 0.92, "free flow", color=INK2, fontsize=7.5); ax.text(0.55, 0.40, "queued", color=INK2, fontsize=7.5)
    fig.tight_layout(); fig.savefig(OUT / "fig_fd.png"); plt.close(fig)


def fig_common_mode():
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ks = [1, 2, 3, 5, 8, 12, 20, 30]
    for p, c in (("D7_I10_E", S1), ("D12_I5_S", S2), ("D7_I405_S", S3)):
        v = load(p, "train")["speed"].astype(float); pl = plateau_of(v)
        e = np.where(v > 0.85 * pl, v - pl, np.nan)
        cs = []
        for k in ks:
            a, b = e[:, :, k:].ravel(), e[:, :, :-k].ravel(); mm = np.isfinite(a) & np.isfinite(b)
            cs.append(np.corrcoef(a[mm], b[mm])[0, 1])
        ax.plot(ks, cs, color=c, lw=1.6, marker="o", ms=4, label=p)
    ax.set_ylim(0, 0.4); ax.set_xlabel("distance between links (positions)"); ax.set_ylabel("corr. of free-flow speed noise")
    ax.set_title("Corridor-wide common-mode noise", fontsize=8.5, color=INK, loc="left")
    ax.legend(fontsize=7, loc="lower left"); fig.tight_layout(); fig.savefig(OUT / "fig_common_mode.png"); plt.close(fig)


def fig_ramp():
    from tfb.t2_events import queue_truth
    p = "D7_I405_S"
    z = load(p, "train"); net = network(p); Q, _ = queue_truth(p, z=z)
    ri = {r: i for i, r in enumerate(z["ramps"])}
    rf = z["ramp_flow"]; prof = np.nanmean(rf, 0)
    rows = []
    for l, s in enumerate(net.on_ramp_link_ids.fillna("")):
        for x in str(s).split(";"):
            if x in ri:
                a = rf[:, :, ri[x]] / np.maximum(prof[None, :, ri[x]], 1); q = Q[:, :, l]; mm = np.isfinite(a)
                if q[mm].sum() >= 500:
                    rows.append((l, np.nanmean(a[mm & ~q]), np.nanmean(a[mm & q])))
    rows = rows[:12]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    y = np.arange(len(rows))
    for i, (l, f, qd) in enumerate(rows):
        ax.plot([qd, f], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter([r[1] for r in rows], y, s=22, c=S1, label="mainline free", zorder=2, edgecolors=SURF, linewidths=1)
    ax.scatter([r[2] for r in rows], y, s=22, c=S2, label="mainline queued", zorder=3, edgecolors=SURF, linewidths=1)
    ax.set_yticks(y, [f"link {r[0]}" for r in rows], fontsize=6.5); ax.set_xlim(0, 1.4)
    ax.set_xlabel("on-ramp flow / time-of-day profile")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    ax.set_title("On-ramp flow when the link is queued", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "fig_ramp.png"); plt.close(fig)


def fig_learning_curve():
    rows = np.array([300, 600, 1200, 2400])
    sp = {"private": [1.584, 1.542, 1.497, 1.486], "validation": [1.524, 1.490, 1.451, 1.420]}
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for (s, vals), c in zip(sp.items(), (S1, S2)):
        ax.plot(rows, vals, color=c, lw=1.6, marker="o", ms=4, label=s)
    ax.set_xscale("log", base=2); ax.set_xticks(rows, ["300k", "600k", "1.2M", "≤2.4M"])
    ax.set_xlabel("train rows per panel"); ax.set_ylabel("speed RMSE, km/h")
    ax.set_title("Task 1 learning curve, out of scenario", fontsize=8.5, color=INK, loc="left")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "fig_learning.png"); plt.close(fig)


if __name__ == "__main__":
    fig_fd(); fig_common_mode(); fig_ramp(); fig_learning_curve()
    print("figures written")
