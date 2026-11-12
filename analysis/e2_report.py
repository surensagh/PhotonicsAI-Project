"""E2 report: Table 2, Figure 2 and the pre-written interpretation matrix, regenerated from run records.

    python analysis/e2_report.py --config configs/e2.yaml

Statistics (protocol 3.8): per map, correctness is averaged over map seeds, giving an (S, I) matrix
(S = 10 subsamples at N = 100, S = 1 at full data; I = 1,000 dev items). 95% CIs and paired
differences come from one shared paired bootstrap (10,000 resamples of subsamples and items).
Holm correction is applied across all F-row comparisons within a panel. Gate G2 uses the
uncorrected 95% CI, exactly as the protocol states it; Holm-adjusted p is shown alongside.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optlm.stats import PairedBootstrap, holm  # noqa: E402
from optlm.utils import load_config, read_json, reports_dir, runs_dir, write_json  # noqa: E402

LABELS = {
    "A": "A identity", "B": "B random linear", "C": "C random quadratic", "D": "D random tanh",
    "E": "E complex intensity", "H": "H random Fourier features", "G": "G MLP (trainable)",
    "F-scatter": "F-scatter", "F-fourier": "F-fourier",
    "F-scatter-untuned": "F-scatter (β = π, untuned)", "F-fourier-untuned": "F-fourier (β = π, untuned)",
}
ROW_ORDER = ["A", "B", "C", "D", "E", "H", "G", "F-scatter", "F-fourier", "F-scatter-untuned", "F-fourier-untuned"]
F_TUNED = ["F-scatter", "F-fourier"]
OPTICAL = {"F-scatter", "F-fourier", "F-scatter-untuned", "F-fourier-untuned"}
INTERPRETATION = {
    1: ("F > A, B and F > H, C (CIs separate)", "Optical structure adds something beyond random features",
        "Proceed; E10 must explain why"),
    2: ("F > A, B but F ≈ H or C", "Benefit is random-feature expansion; optics is a physical implementation of it",
        "Proceed with reframed H2 (Section 7)"),
    3: ("F ≈ A", "Expansion adds nothing at this layer", "Run E3 and E4 before concluding"),
    4: ("F < A", "Map destroys information (check β range, normalization)", "Debug; re-run E0 tests"),
    0: ("F > A but ≈ B", "Not covered by the matrix: a random linear projection explains the gain",
        "Check standardization / ridge; investigate before E3"),
}


def load_records(exp="E2"):
    recs = {}
    for p in sorted(runs_dir(exp).glob("*.json")):
        r = read_json(p)
        m = r["metrics"]
        recs[(m["map"], m["D"], m["regime"])] = r
    return recs


def correctness(rec) -> np.ndarray:
    pred = np.array(rec["predictions"]["pred"])        # (seeds, S, I)
    y = np.array(rec["predictions"]["label"])
    return (pred == y).mean(0)                          # (S, I)


def panel_rows(recs, D, regime, d):
    rows = {}
    for m in ROW_ORDER:
        key_D = d if m == "A" else D
        if (m, key_D, regime) in recs:
            rows[m] = recs[(m, key_D, regime)]
    return rows


def interpret(dvs) -> int:
    """dvs: dict comparator -> diff result for the best tuned F."""
    sig_pos = {k: v["ci_excludes_0"] and v["diff"] > 0 for k, v in dvs.items()}
    if dvs["A"]["ci_excludes_0"] and dvs["A"]["diff"] < 0:
        return 4
    if not sig_pos["A"]:
        return 3
    if not sig_pos.get("B", True):
        return 0
    if sig_pos.get("H", False) and sig_pos.get("C", False):
        return 1
    return 2


def make_figure(panels, path, exp="E2"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid_c = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    optical_c, digital_c = "#2a78d6", "#52514e"
    keys = list(panels)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), facecolor=surface, sharey=True)
    for ax, key in zip(axes.ravel(), keys):
        P = panels[key]
        ax.set_facecolor(surface)
        order = [m for m in ROW_ORDER if m in P["rows"]]
        ys = np.arange(len(order))[::-1]
        for y, m in zip(ys, order):
            r = P["rows"][m]
            c = optical_c if m in OPTICAL else digital_c
            ax.errorbar(100 * r["mean"], y, xerr=[[100 * (r["mean"] - r["ci_low"])], [100 * (r["ci_high"] - r["mean"])]],
                        fmt="o", ms=8, color=c, ecolor=c, elinewidth=2, capsize=0,
                        markeredgecolor=surface, markeredgewidth=2)
        if "A" in P["rows"]:
            ax.axvline(100 * P["rows"]["A"]["mean"], color=ink2, lw=1, ls="--", zorder=0)
        ax.set_yticks(ys)
        ax.set_yticklabels([LABELS[m] for m in order], color=ink, fontsize=9)
        ax.set_title(P["title"], color=ink, fontsize=11, loc="left")
        ax.set_xlabel("dev accuracy, % (mean, 95% CI)", color=ink2, fontsize=9)
        ax.grid(axis="x", color=grid_c, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(grid_c)
        ax.tick_params(colors=ink2, length=0)
    fig.suptitle(f"Figure 2 · {exp}: optical vs digital feature maps (SciQ dev, Pythia-160M L12 mean). "
                 "Blue = optical; dashed = identity A", color=ink, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200, facecolor=surface)
    plt.close(fig)


def build_report(cfg: dict) -> dict:
    exp = cfg.get("experiment", "E2")
    tag = exp.lower()
    recs = load_records(exp)
    d = min(cfg["dims"])
    B, bseed = cfg["bootstrap"]["B"], cfg["bootstrap"]["seed"]
    panels, out = {}, {"panels": {}, "B": B, "experiment": exp, "exploratory": bool(cfg.get("exploratory")),
                       "note": cfg.get("note", "")}
    for D in cfg["dims"]:
        for reg in cfg["regimes"]:
            rows = panel_rows(recs, D, reg["name"], d)
            if not rows:
                continue
            C = {m: correctness(r) for m, r in rows.items()}
            S, I = next(iter(C.values())).shape
            bs = PairedBootstrap(S, I, B=B, seed=bseed)
            stats = {}
            for m, r in rows.items():
                s = bs.mean_ci(C[m])
                s.update({"selected": r["metrics"]["selected"], "params": r["metrics"]["trainable_params"],
                          "knob_name": r["metrics"]["knob_name"], "knob_values": r["metrics"]["knob_values"],
                          "lambdas": r["metrics"]["lambdas"], "grid": r["metrics"]["grid_mean_dev_acc"],
                          "seed_sd": float(np.array(r["metrics"]["per_run_dev_acc"]).mean(1).std(ddof=1))
                          if len(r["metrics"]["per_run_dev_acc"]) > 1 else 0.0})
                stats[m] = s
            digital_best = max((m for m in ("C", "H") if m in C), key=lambda m: stats[m]["mean"], default=None)
            comps, flat = {}, []
            for f in [m for m in ROW_ORDER if m in OPTICAL and m in C]:
                comps[f] = {}
                for ref in ["A", "B", "C", "H", "D", "E", "G"]:
                    if ref in C:
                        comps[f][ref] = bs.diff(C[f], C[ref])
                        flat.append((f, ref))
                if digital_best:
                    comps[f]["best_digital"] = dict(bs.diff(C[f], C[digital_best]), ref=digital_best)
            adj = holm([comps[f][ref]["p"] for f, ref in flat]) if flat else []
            for (f, ref), a in zip(flat, adj):
                comps[f][ref]["p_holm"] = a
            tuned = [f for f in F_TUNED if f in stats]
            best_f = max(tuned, key=lambda f: stats[f]["mean"]) if tuned else None
            row = interpret(comps[best_f]) if best_f and "A" in comps[best_f] else None
            key = f"D{D}_{reg['name']}"
            title = f"D = {D} ({'d' if D == d else f'{D // d}d'}) · {'N = ' + str(reg['N']) if 'N' in reg else 'full data'}"
            panels[key] = {"title": title, "rows": stats}
            out["panels"][key] = {"title": title, "rows": stats, "comparisons": comps, "best_digital": digital_best,
                                  "best_F": best_f, "interpretation_row": row, "S": S}

    # Gate G2: F beats A at N = 100 (CI excludes 0) for some D
    g2_hits = []
    for key, P in out["panels"].items():
        if key.endswith("N100"):
            for f in F_TUNED:
                c = P["comparisons"].get(f, {}).get("A")
                if c and c["ci_excludes_0"] and c["diff"] > 0:
                    g2_hits.append(f"{f} at {P['title']}")
    out["g2"] = {"passed": bool(g2_hits), "hits": g2_hits}
    write_json(reports_dir() / f"{tag}_results.json", out)
    if panels:
        make_figure(panels, reports_dir() / f"{tag}_fig2.png", exp)
    write_markdown(out, reports_dir() / f"{tag}_table2.md", tag)
    print(f"{exp} report ->", reports_dir() / f"{tag}_table2.md")
    print("Gate G2:", "PASSED" if out["g2"]["passed"] else "NOT PASSED", out["g2"]["hits"])
    return out


def pct(x):
    return f"{100 * x:.1f}"


def fmt_knob(name, v):
    if name == "beta":
        for frac, s in ((1 / 64, "π/64"), (1 / 32, "π/32"), (1 / 16, "π/16"), (0.125, "π/8"), (0.25, "π/4"),
                        (0.5, "π/2"), (1.0, "π"), (2.0, "2π")):
            if abs(v - frac * np.pi) < 1e-9:
                return f"β={s}"
    label = {"s": "s", "sigma_over_sqrt_d": "σ/√d", "dropout": "dropout"}.get(name, name)
    return f"{label}={v:g}"


def fmt_p(p, B):
    return f"< {1 / B:.0e}" if p == 0 else f"{p:.3g}"


def fmt_diff(c):
    if not c:
        return "—"
    star = " *" if c["ci_excludes_0"] else ""
    return f"{100 * c['diff']:+.1f} [{100 * c['ci_low']:+.1f}, {100 * c['ci_high']:+.1f}]{star}"


def write_markdown(out, path, tag="e2"):
    L = [f"# Table 2 · {out['experiment']}: optical vs digital feature maps", ""]
    if out["exploratory"]:
        L += ["> **EXPLORATORY diagnostic (logged deviation, see deviations.md).** Not part of the pre-registered",
              "> analysis; the gate line below is informational only and does not replace gate G2.", "",
              out["note"], ""]
    L += [
         "SciQ dev (1,000 questions), Pythia-160M-deduped, L12, mean pooling, linear scoring head.",
         "Mean dev accuracy over map seeds (and the 10 paired subsamples at N = 100) with 95% paired-bootstrap CIs.",
         "Hyperparameters selected on dev (protocol 3.6): these are post-selection development numbers.",
         "Δ columns: paired difference F − reference in percentage points [95% CI]; * = CI excludes 0 (uncorrected).", ""]
    g2 = out["g2"]
    L += [f"## Gate G2 (F beats A at N = 100, CI excludes 0, for some D): **{'PASSED' if g2['passed'] else 'NOT PASSED'}**",
          ""] + [f"- {h}" for h in g2["hits"]] + [""]
    for key, P in out["panels"].items():
        L += [f"## {P['title']}", "",
              "| map | trainable params | selected config | dev acc % | 95% CI | seed SD (pp) |", "|---|---|---|---|---|---|"]
        for m in ROW_ORDER:
            if m not in P["rows"]:
                continue
            r = P["rows"][m]
            sel = r["selected"]
            parts = []
            kv = [k for k in r["knob_values"] if k is not None]
            lams = [x for x in r["lambdas"] if x is not None]
            if sel["knob"] is not None:
                edge = " (edge)" if len(kv) > 1 and sel["knob"] in (min(kv), max(kv)) else ""
                parts.append(fmt_knob(r["knob_name"], sel["knob"]) + edge)
            if sel["lambda"] is not None:
                edge = " (edge)" if sel["lambda"] in (min(lams), max(lams)) else ""
                parts.append(f"λ={sel['lambda']:g}{edge}")
            cfg_s = ", ".join(parts)
            L.append(f"| {LABELS[m]} | {r['params']:,} | {cfg_s or '—'} | {pct(r['mean'])} | "
                     f"[{pct(r['ci_low'])}, {pct(r['ci_high'])}] | {100 * r['seed_sd']:.2f} |")
        L += ["", f"Best digital random-feature map (of C, H): **{P['best_digital']}**", "",
              "| optical row | Δ vs A | Δ vs B | Δ vs C | Δ vs H | Δ vs best digital | Δ vs G (MLP) | Holm p vs A |",
              "|---|---|---|---|---|---|---|---|"]
        for f, comp in P["comparisons"].items():
            L.append(f"| {LABELS[f]} | {fmt_diff(comp.get('A'))} | {fmt_diff(comp.get('B'))} | {fmt_diff(comp.get('C'))} | "
                     f"{fmt_diff(comp.get('H'))} | {fmt_diff(comp.get('best_digital'))} | {fmt_diff(comp.get('G'))} | "
                     f"{fmt_p(comp['A']['p_holm'], out['B'])} |" if "A" in comp else f"| {LABELS[f]} | — | — | — | — | — | — | — |")
        row = P["interpretation_row"]
        if row is not None:
            obs, concl, nxt = INTERPRETATION[row]
            L += ["", f"**Interpretation matrix (best tuned F = {P['best_F']}):** observed *{obs}* → {concl}. Next: {nxt}.", ""]
        # Knob profile: best dev accuracy over lambda at each knob value (shows where the optimum sits)
        prof = [m for m in ROW_ORDER if m in P["rows"] and len([k for k in P["rows"][m]["knob_values"] if k is not None]) > 1]
        if prof:
            L += ["", "**Knob profile** (dev acc %, best λ at each knob value; selected value in bold):", ""]
            for m in prof:
                r = P["rows"][m]
                g = np.array(r["grid"], dtype=float)
                cells = []
                for k, v in enumerate(r["knob_values"]):
                    s_ = f"{fmt_knob(r['knob_name'], v)}: {100 * g[k].max():.1f}"
                    cells.append(f"**{s_}**" if v == r["selected"]["knob"] else s_)
                L.append(f"- {LABELS[m]}: " + " · ".join(cells))
        L.append("")
    L += ["", f"![Figure 2]({tag}_fig2.png)", ""]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e2.yaml")
    build_report(load_config(ap.parse_args().config))
