"""E3-A report: placement and integration (layer x pooling x integration), and frozen decision D-E3.

    python analysis/e3_report.py --config configs/e3.yaml

For every cell and regime: dev accuracy of A (identity, same layer/pooling), H and the optical map
with 95% paired-bootstrap CIs, and paired differences optical - A and optical - H (Holm across the
cells within a regime). Decision D-E3 (protocol): the cell with the best N = 100 dev accuracy for the
optical map; ties go to the simpler cell (replacement before parallel, mean before last, later layer
first). Written to reports/e3_decision.json and frozen for E4 onward.

Residual integration h' = LN(h + alpha B r) is not run: with a linear head on top it is a constrained
version of the parallel head [h; r], so it adds no information (protocol E3).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.plot_style import ENTITY_COLORS, INK, INK2, LABELS, SURFACE, style_axes  # noqa: E402
from optlm.stats import PairedBootstrap, holm  # noqa: E402
from optlm.utils import load_config, now_iso, read_json, reports_dir, runs_dir, write_json  # noqa: E402

LAYER_ORDER = {"L3": 0, "L6": 1, "L12": 2, "L12raw": 3}


def load(exp):
    recs = {}
    for p in sorted(runs_dir(exp).glob("*.json")):
        r = read_json(p)
        m = r["metrics"]
        recs[(m["layer"], m["pooling"], m["integration"], m["map"], m["regime"])] = r
    return recs


def corr(rec):
    pred = np.array(rec["predictions"]["pred"])
    return (pred == np.array(rec["predictions"]["label"])).mean(0)  # (S, I)


def build_report(cfg: dict) -> dict:
    exp, fmap, ref = cfg.get("experiment", "E3"), cfg["optical_map"], cfg["digital_ref"]
    recs = load(exp)
    c = cfg["cells"]
    cells = [(l, p, i) for l in c["layers"] for p in c["poolings"] for i in c["integrations"]]
    out = {"experiment": exp, "optical_map": fmap, "digital_ref": ref, "regimes": {}}
    for reg in cfg["regimes"]:
        name = reg["name"]
        rows, comps = [], []
        bs = None
        for (l, p, i) in cells:
            rf, rh, ra = recs.get((l, p, i, fmap, name)), recs.get((l, p, i, ref, name)), recs.get((l, p, "replacement", "A", name))
            if not (rf and rh and ra):
                continue
            Cf, Ch, Ca = corr(rf), corr(rh), corr(ra)
            if bs is None:
                bs = PairedBootstrap(*Cf.shape, B=cfg["bootstrap"]["B"], seed=cfg["bootstrap"]["seed"])
            row = {"layer": l, "pooling": p, "integration": i,
                   "F": dict(bs.mean_ci(Cf), selected=rf["metrics"]["selected"], params=rf["metrics"]["trainable_params"]),
                   "H": dict(bs.mean_ci(Ch), selected=rh["metrics"]["selected"]),
                   "A": dict(bs.mean_ci(Ca), selected=ra["metrics"]["selected"]),
                   "F_minus_A": bs.diff(Cf, Ca), "F_minus_H": bs.diff(Cf, Ch)}
            rows.append(row)
            comps += [row["F_minus_A"], row["F_minus_H"]]
        for cmp_, a in zip(comps, holm([x["p"] for x in comps]) if comps else []):
            cmp_["p_holm"] = a
        out["regimes"][name] = rows

    # Decision D-E3: best N = 100 dev accuracy of the optical map; ties -> simpler cell
    n100 = out["regimes"].get("N100", [])
    if n100:
        def key(r):
            return (round(r["F"]["mean"], 10), r["integration"] == "replacement", r["pooling"] == "mean",
                    LAYER_ORDER[r["layer"]])
        best = max(n100, key=key)
        decision = {"decision": "D-E3", "layer": best["layer"], "pooling": best["pooling"],
                    "integration": best["integration"], "optical_map": fmap,
                    "n100_dev_acc": best["F"]["mean"], "created": now_iso(),
                    "rule": "best N=100 dev accuracy of the optical map; ties -> replacement, mean, later layer"}
        write_json(reports_dir() / "e3_decision.json", decision)
        out["decision"] = decision
    write_json(reports_dir() / "e3_results.json", out)
    make_figure(out, reports_dir() / "e3_fig.png")
    write_markdown(out, reports_dir() / "e3_table.md")
    print("E3 report ->", reports_dir() / "e3_table.md")
    if "decision" in out:
        d = out["decision"]
        print(f"Decision D-E3: layer {d['layer']}, {d['pooling']} pooling, {d['integration']} "
              f"(N=100 dev acc {100 * d['n100_dev_acc']:.1f}%)")
    return out


def cell_label(r):
    return f"{r['layer']} · {r['pooling']} · {'[h; r]' if r['integration'] == 'parallel' else 'r'}"


def fd(c):
    s = " *" if c["ci_excludes_0"] else ""
    return f"{100 * c['diff']:+.1f} [{100 * c['ci_low']:+.1f}, {100 * c['ci_high']:+.1f}]{s}"


def write_markdown(out, path):
    f, h = out["optical_map"], out["digital_ref"]
    L = ["# E3-A · Placement and integration (protocol v2 encoding)", "",
         f"SciQ dev, Pythia-160M, D = 4d, optical map **{f}**, digital reference **{h}**, identity **A** at the same",
         "layer and pooling. Mean over 3 map seeds (and 10 paired subsamples at N = 100), 95% paired-bootstrap CIs.",
         "Integration: `r` = replacement (head sees r), `[h; r]` = parallel (head sees both).",
         "Residual integration is not run: under a linear head it is a constrained parallel head (protocol E3).",
         "Δ = paired difference in percentage points; * = 95% CI excludes 0; Holm p across all Δ in the regime.", ""]
    if "decision" in out:
        d = out["decision"]
        L += [f"## Decision D-E3 (frozen for E4 onward): **{d['layer']} · {d['pooling']} pooling · "
              f"{'parallel [h; r]' if d['integration'] == 'parallel' else 'replacement r'}** "
              f"(N = 100 dev acc {100 * d['n100_dev_acc']:.1f}%)", ""]
    for reg, rows in out["regimes"].items():
        L += [f"## {'N = 100' if reg == 'N100' else 'Full data'}", "",
              f"| cell | A % | {h} % | {f} % | Δ {f} − A | Δ {f} − {h} | Holm p (vs A) |", "|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {cell_label(r)} | {100 * r['A']['mean']:.1f} | {100 * r['H']['mean']:.1f} | "
                     f"**{100 * r['F']['mean']:.1f}** [{100 * r['F']['ci_low']:.1f}, {100 * r['F']['ci_high']:.1f}] | "
                     f"{fd(r['F_minus_A'])} | {fd(r['F_minus_H'])} | {('< 1e-04' if r['F_minus_A']['p_holm'] == 0 else format(r['F_minus_A']['p_holm'], '.3g'))} |")
        L.append("")
    L += ["![E3 figure](e3_fig.png)", ""]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def make_figure(out, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regs = [r for r in ("N100", "full") if out["regimes"].get(r)]
    if not regs:
        return
    fig, axes = plt.subplots(1, len(regs), figsize=(6.5 * len(regs), 6.5), facecolor=SURFACE, sharey=True)
    axes = np.atleast_1d(axes)
    series = [("A", "A", -0.22), (out["digital_ref"], "H", 0.0), (out["optical_map"], "F", 0.22)]
    for ax, reg in zip(axes, regs):
        rows = out["regimes"][reg]
        ys = np.arange(len(rows))[::-1]
        style_axes(ax)
        for ent, key, off in series:
            col = ENTITY_COLORS.get(ent, INK2)
            for y, r in zip(ys, rows):
                s = r[key]
                ax.errorbar(100 * s["mean"], y + off, xerr=[[100 * (s["mean"] - s["ci_low"])], [100 * (s["ci_high"] - s["mean"])]],
                            fmt="o", ms=7, color=col, ecolor=col, elinewidth=2, capsize=0,
                            markeredgecolor=SURFACE, markeredgewidth=1.5,
                            label=LABELS.get(ent, ent) if y == ys[0] else None)
        ax.set_yticks(ys)
        ax.set_yticklabels([cell_label(r) for r in rows], color=INK, fontsize=9)
        ax.set_title("N = 100" if reg == "N100" else "Full data", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("dev accuracy, % (mean, 95% CI)", color=INK2, fontsize=9)
    axes[0].legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("E3-A · Placement and integration (layer · pooling · integration), D = 4d",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e3.yaml")
    build_report(load_config(ap.parse_args().config))
