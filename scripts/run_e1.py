"""E1: baselines on SciQ dev with full training data (Core part: zero-shot, linear probe, MLP head).

    python scripts/run_e1.py --config configs/e1.yaml [--skip-zero-shot]

Writes: <workspace>/reports/e1_table1.md, e1_results.json, and run records in runs/E1/.
Note (protocol 3.6): hyperparameters are selected on dev and reported on dev, so these are
post-selection dev numbers; the unbiased number comes from the locked test set in E12.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optlm.data_sciq import load_items  # noqa: E402
from optlm.features import Standardizer, load_sciq_features  # noqa: E402
from optlm.heads import evaluate_scores, predictions_record, select_lambda, train_mlp  # noqa: E402
from optlm.utils import load_config, now_iso, reports_dir, run_name, save_run_record, write_json  # noqa: E402
from optlm.zero_shot import load_causal_lm, zero_shot_scores  # noqa: E402

TASK = "sciq"


def run_zero_shot(cfg, ids, ydev):
    started = now_iso()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok, model = load_causal_lm(cfg["hf_id"], cfg["revision"], device, dtype)
    res = zero_shot_scores(model, tok, load_items("dev"), cfg["zero_shot"]["batch_size"], device)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    out = {}
    for norm in ("per_token", "per_char", "sum"):
        m = evaluate_scores(res[norm], ydev)
        name = run_name("E1", f"zeroshot_{norm}", cfg["model_tag"], TASK, "0", "NA", "NA")
        save_run_record("E1", name, cfg, {"dev": m, "seconds": res["seconds"]},
                        predictions_record(ids, res[norm], ydev), started)
        out[norm] = {"accuracy": m["accuracy"], "seconds": res["seconds"]}
        print(f"zero-shot {norm:9s}: dev acc {100 * m['accuracy']:.1f}%")
    return out


def run_linear(cfg, Xtr, ytr, Xdev, ydev, ids):
    started = now_iso()
    t0 = time.time()
    sel = select_lambda(Xtr, ytr, Xdev, ydev, cfg["linear"]["lambdas"], cfg["linear"]["max_iter"])
    best = sel["best"]
    sweep = [{"lambda": r["lambda"], "dev_acc": r["metrics"]["accuracy"], "n_iter": r["n_iter"]} for r in sel["all"]]
    for r in sweep:
        print(f"linear lambda={r['lambda']:.0e}: dev acc {100 * r['dev_acc']:.1f}% ({r['n_iter']} iters)")
    name = run_name("E1", "linear_probe", cfg["model_tag"], TASK, "full", "NA", "NA")
    save_run_record("E1", name, cfg, {"dev": best["metrics"], "selected_lambda": best["lambda"], "sweep": sweep,
                                      "fit_seconds": best["seconds"], "tuning_seconds": time.time() - t0},
                    predictions_record(ids, best["scores"], ydev), started)
    return {"metrics": best["metrics"], "lambda": best["lambda"], "sweep": sweep, "fit_seconds": best["seconds"],
            "tuning_seconds": time.time() - t0, "n_params": int(Xtr.shape[-1])}


def run_mlp(cfg, Xtr, ytr, Xdev, ydev, ids):
    mc = cfg["mlp"]
    table = {}
    for p in mc["dropouts"]:
        per_seed = []
        for seed in mc["seeds"]:
            started = now_iso()
            r = train_mlp(Xtr, ytr, Xdev, ydev, hidden=mc["hidden"], out_dim=mc["out_dim"], dropout=p, lr=mc["lr"],
                          weight_decay=mc["weight_decay"], batch_size=mc["batch_size"], patience=mc["patience"],
                          max_epochs=mc["max_epochs"], seed=seed)
            m = evaluate_scores(r["scores"], ydev)
            name = run_name("E1", f"mlp_do{p}", cfg["model_tag"], TASK, "full", seed, "NA")
            save_run_record("E1", name, {**cfg, "dropout": p, "seed": seed},
                            {"dev": m, "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
                             "seconds": r["seconds"], "n_params": r["n_params"], "history": r["history"]},
                            predictions_record(ids, r["scores"], ydev), started)
            per_seed.append({"seed": seed, "metrics": m, "seconds": r["seconds"], "best_epoch": r["best_epoch"],
                             "n_params": r["n_params"]})
            print(f"mlp dropout={p} seed={seed}: dev acc {100 * m['accuracy']:.1f}% "
                  f"(best epoch {r['best_epoch']}, {r['seconds']:.0f}s)")
        accs = [s["metrics"]["accuracy"] for s in per_seed]
        table[p] = {"mean_acc": float(np.mean(accs)), "sd_acc": float(np.std(accs, ddof=1)), "runs": per_seed}
    best_p = max(table, key=lambda p: (table[p]["mean_acc"], p))  # tie -> more dropout
    return {"selected_dropout": best_p, "by_dropout": table}


def fmt_pct(m, s=None):
    return f"{100 * m:.1f}" + (f" ± {100 * s:.1f}" if s is not None else "")


def write_table(cfg, results, path):
    lin, mlp = results["linear"], results["mlp"]
    best = mlp["by_dropout"][mlp["selected_dropout"]]
    runs = best["runs"]
    mlp_nll = np.mean([r["metrics"]["nll"] for r in runs])
    mlp_ece = np.mean([r["metrics"]["ece"] for r in runs])
    mlp_time = np.mean([r["seconds"] for r in runs])
    lines = [
        "# Table 1: E1 baselines (SciQ dev, full training data)",
        "",
        f"Model: {cfg['hf_id']} @ {cfg['revision']}, layer {cfg['layer']}, {cfg['pooling']} pooling. "
        f"Train items: {results['n_train']}, dev items: {results['n_dev']}.",
        "Hyperparameters were selected on dev and are reported on dev (post-selection numbers; see protocol 3.6).",
        "",
        "| Method | Trainable params | Dev accuracy % (mean ± SD) | Dev NLL | Dev ECE | Train time | Selected config |",
        "|---|---|---|---|---|---|---|",
        "| Random guess | 0 | 25.0 | — | — | — | — |",
    ]
    if "zero_shot" in results:
        zs = results["zero_shot"]
        lines.append(f"| Zero-shot LM, per-token normalized (primary) | 0 | {fmt_pct(zs['per_token']['accuracy'])} | — | — | "
                     f"{zs['per_token']['seconds']:.0f}s inference | — |")
        lines.append(f"| Zero-shot LM, per-character normalized | 0 | {fmt_pct(zs['per_char']['accuracy'])} | — | — | — | — |")
        lines.append(f"| Zero-shot LM, unnormalized sum | 0 | {fmt_pct(zs['sum']['accuracy'])} | — | — | — | — |")
    lines.append(f"| Linear probe (map A) | {lin['n_params']:,} | {fmt_pct(lin['metrics']['accuracy'])} "
                 f"(deterministic: identical across seeds) | {lin['metrics']['nll']:.3f} | {lin['metrics']['ece']:.3f} | "
                 f"{lin['fit_seconds']:.1f}s (tuning {lin['tuning_seconds']:.0f}s) | λ = {lin['lambda']:g} |")
    lines.append(f"| MLP head (map G + linear) | {runs[0]['n_params']:,} | {fmt_pct(best['mean_acc'], best['sd_acc'])} "
                 f"(n = {len(runs)} seeds) | {mlp_nll:.3f} | {mlp_ece:.3f} | {mlp_time:.0f}s per seed | "
                 f"dropout = {mlp['selected_dropout']} |")
    lines += ["", "## Linear probe: λ sweep (dev accuracy %)", "", "| λ | dev acc | L-BFGS iters |", "|---|---|---|"]
    for r in lin["sweep"]:
        lines.append(f"| {r['lambda']:g} | {100 * r['dev_acc']:.1f} | {r['n_iter']} |")
    lines += ["", "## MLP head: dropout sweep (dev accuracy %, mean ± SD over seeds)", "",
              "| dropout | dev acc | best epochs |", "|---|---|---|"]
    for p, t in mlp["by_dropout"].items():
        lines.append(f"| {p} | {fmt_pct(t['mean_acc'], t['sd_acc'])} | {[r['best_epoch'] for r in t['runs']]} |")
    gate = results["gate"]
    lines += ["", f"## Gate: linear probe ≥ {100 * gate['threshold']:.0f}% → **{'PASSED' if gate['passed'] else 'FAILED'}** "
              f"({fmt_pct(gate['value'])}%)"]
    if not gate["passed"]:
        lines.append("Protocol E1: representation or pooling is broken. Try last-token pooling and the middle layer (L6) before E2.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e1.yaml")
    ap.add_argument("--skip-zero-shot", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    feats = load_sciq_features(cfg["model_tag"], cfg["layer"], cfg["pooling"])
    Xtr_raw, ytr, _ = feats["train"]
    Xdev_raw, ydev, ids = feats["dev"]
    std = Standardizer().fit(Xtr_raw)
    Xtr, Xdev = std.transform(Xtr_raw), std.transform(Xdev_raw)
    print(f"train {Xtr.shape}, dev {Xdev.shape}")

    results = {"config": cfg, "n_train": int(len(ytr)), "n_dev": int(len(ydev))}
    if not args.skip_zero_shot:
        results["zero_shot"] = run_zero_shot(cfg, ids, ydev)
    results["linear"] = run_linear(cfg, Xtr, ytr, Xdev, ydev, ids)
    results["mlp"] = run_mlp(cfg, Xtr, ytr, Xdev, ydev, ids)
    acc = results["linear"]["metrics"]["accuracy"]
    results["gate"] = {"threshold": cfg["gate"]["min_linear_probe_acc"], "value": acc,
                       "passed": acc >= cfg["gate"]["min_linear_probe_acc"]}

    write_json(reports_dir() / "e1_results.json", results)
    write_table(cfg, results, reports_dir() / "e1_table1.md")
    print("\nTable 1 ->", reports_dir() / "e1_table1.md")
    print("Linear-probe gate:", "PASSED" if results["gate"]["passed"] else "FAILED")


if __name__ == "__main__":
    main()
