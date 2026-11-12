"""E4 report: accuracy vs optical dimension D (Figure 3) at the frozen placement D-E3.

    python analysis/e4_report.py --config configs/e4.yaml

Per map, D and regime: mean dev accuracy with a 95% paired-bootstrap CI. Saturation D = the smallest
D whose accuracy is within 1 pp of that map's best accuracy in the regime. Interpretation (protocol):
if the optical map saturates at a lower D than H it is more feature-efficient (fewer camera pixels);
if both saturate together they have equivalent capacity. Also: the optical map minus H at every D.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.plot_style import ENTITY_COLORS, GRID, INK, INK2, LABELS, SURFACE, style_axes  # noqa: E402
from optlm.stats import PairedBootstrap  # noqa: E402
from optlm.utils import load_config, read_json, reports_dir, runs_dir, write_json  # noqa: E402

MAP_ORDER = ["B", "C", "H", "F2-scatter", "F2-fourier"]


def build_report(cfg: dict) -> dict:
    exp = cfg.get("experiment", "E4")
    recs = {}
    for p in sorted(runs_dir(exp).glob("*.json")):
        r = read_json(p)
        m = r["metrics"]
        recs[(m["map"], m["D"], m["regime"])] = r
    dec = read_json(reports_dir() / cfg["cells_from"]) if "cells_from" in cfg else {}
    out = {"experiment": exp, "placement": dec, "regimes": {}}
    for reg in cfg["regimes"]:
        name = reg["name"]
        keys = [k for k in recs if k[2] == name]
        if not keys:
            continue
        C = {}
        for (m, D, _) in keys:
            r = recs[(m, D, name)]
            C[(m, D)] = (np.array(r["predictions"]["pred"]) == np.array(r["predictions"]["label"])).mean(0)
        S, I = next(iter(C.values())).shape
        bs = PairedBootstrap(S, I, B=cfg["bootstrap"]["B"], seed=cfg["bootstrap"]["seed"])
        maps = {}
        for m in MAP_ORDER:
            Ds = sorted(D for (mm, D) in C if mm == m)
            if not Ds:
                continue
            pts = [dict(bs.mean_ci(C[(m, D)]), D=D, selected=recs[(m, D, name)]["metrics"]["selected"]) for D in Ds]
            best = max(p["mean"] for p in pts)
            sat = min(p["D"] for p in pts if p["mean"] >= best - 0.01)
            maps[m] = {"points": pts, "best": best, "saturation_D": sat}
        a = [(D, C[("A", D)]) for (mm, D) in C if mm == "A"]
        A = dict(bs.mean_ci(a[0][1]), D=a[0][0]) if a else None
        f_vs_h = {}
        for f in ("F2-fourier", "F2-scatter"):
            for (mm, D) in list(C):
                if mm == f and ("H", D) in C:
                    f_vs_h[f"{f} D={D}"] = bs.diff(C[(f, D)], C[("H", D)])
        out["regimes"][name] = {"maps": maps, "A": A, "F_minus_H": f_vs_h}
    write_json(reports_dir() / "e4_results.json", out)
    make_figure(out, reports_dir() / "e4_fig3.png")
    write_markdown(out, reports_dir() / "e4_table.md")
    print("E4 report ->", reports_dir() / "e4_table.md")
    return out


def write_markdown(out, path):
    dec = out["placement"]
    L = ["# E4 · Optical dimension (protocol v2 encoding)", ""]
    if dec:
        L += [f"Placement (frozen decision D-E3): **{dec['layer']} · {dec['pooling']} pooling · {dec['integration']}**.", ""]
    L += ["Mean dev accuracy over 3 map seeds (and 10 paired subsamples at N = 100), 95% paired-bootstrap CI.",
          "Saturation D = smallest D within 1 pp of the map's best accuracy in that regime.", ""]
    for reg, R in out["regimes"].items():
        allD = sorted({p["D"] for v in R["maps"].values() for p in v["points"]})
        L += [f"## {'N = 100' if reg == 'N100' else 'Full data'}", ""]
        if R["A"]:
            L += [f"Identity A (D = {R['A']['D']}): {100 * R['A']['mean']:.1f}% "
                  f"[{100 * R['A']['ci_low']:.1f}, {100 * R['A']['ci_high']:.1f}]", ""]
        L += ["| map | " + " | ".join(f"D={D}" for D in allD) + " | saturation D |",
              "|---|" + "---|" * len(allD) + "---|"]
        for m, v in R["maps"].items():
            cells = {p["D"]: f"{100 * p['mean']:.1f}" for p in v["points"]}
            L.append(f"| {LABELS.get(m, m)} | " + " | ".join(cells.get(D, "—") for D in allD) + f" | {v['saturation_D']} |")
        L += ["", "Optical minus H at the same D (pp, 95% CI; * = excludes 0):", ""]
        for k, c in R["F_minus_H"].items():
            s = " *" if c["ci_excludes_0"] else ""
            L.append(f"- {k}: {100 * c['diff']:+.1f} [{100 * c['ci_low']:+.1f}, {100 * c['ci_high']:+.1f}]{s}")
        L.append("")
    L += ["![Figure 3](e4_fig3.png)", ""]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def make_figure(out, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regs = [r for r in ("N100", "full") if r in out["regimes"]]
    if not regs:
        return
    fig, axes = plt.subplots(1, len(regs), figsize=(6.5 * len(regs), 5), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, reg in zip(axes, regs):
        R = out["regimes"][reg]
        style_axes(ax)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for m, v in R["maps"].items():
            col = ENTITY_COLORS.get(m, INK2)
            D = [p["D"] for p in v["points"]]
            mu = [100 * p["mean"] for p in v["points"]]
            lo = [100 * p["ci_low"] for p in v["points"]]
            hi = [100 * p["ci_high"] for p in v["points"]]
            ax.fill_between(D, lo, hi, color=col, alpha=0.12, lw=0)
            ax.plot(D, mu, color=col, lw=2, marker="o", ms=8, markeredgecolor=SURFACE, markeredgewidth=2,
                    label=LABELS.get(m, m))
            ax.annotate(LABELS.get(m, m).split(" (")[0], (D[-1], mu[-1]), xytext=(6, 0), textcoords="offset points",
                        color=INK2, fontsize=8, va="center")
        if R["A"]:
            ax.axhline(100 * R["A"]["mean"], color=INK2, lw=1, ls="--", zorder=0)
            ax.annotate("A identity", (ax.get_xlim()[0], 100 * R["A"]["mean"]), xytext=(4, 4),
                        textcoords="offset points", color=INK2, fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("optical / feature dimension D (log₂)", color=INK2, fontsize=9)
        ax.set_ylabel("dev accuracy, %", color=INK2, fontsize=9)
        ax.set_title("N = 100" if reg == "N100" else "Full data", color=INK, fontsize=11, loc="left")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    dec = out["placement"]
    sub = f" at {dec['layer']} · {dec['pooling']} · {dec['integration']}" if dec else ""
    fig.suptitle(f"Figure 3 · E4: accuracy vs dimension D{sub}", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.97, 0.94))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e4.yaml")
    build_report(load_config(ap.parse_args().config))
