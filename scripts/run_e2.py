"""E2: optical vs digital feature maps (protocol E2, the project's most important experiment).

    python scripts/run_e2.py --config configs/e2.yaml [--maps A F-scatter] [--force]

For every map x D x regime, every configuration (knob x lambda, or dropout for G) is trained for
every map seed (and every paired subsample in the N = 100 regime) and scored on dev. The selected
configuration is the one with the highest mean dev accuracy over seeds and subsamples (ties: stronger
regularization). One run record per (map, D, regime) holds the full accuracy grid and the per-item
dev predictions of the selected configuration; the full prediction grid goes to a .npz next to it.

Resumable: a (map, D) whose records exist is skipped unless --force. Ends by writing the report.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optlm.featuremaps import KNOB_NAMES, FeatureMap, standardize_fit  # noqa: E402
from optlm.features import Standardizer, load_sciq_features  # noqa: E402
from optlm.heads import fit_linear_batched, n_params, train_mlp, MLPScorer  # noqa: E402
from optlm.subsamples import get_subsamples  # noqa: E402
from optlm.utils import load_config, now_iso, run_name, runs_dir, save_run_record  # noqa: E402

EXP, TASK = "E2", "sciq"


def record_name(cfg, m, D, regime, variant_suffix=""):
    n = regime.get("N", "full")
    seeds = "0" if m == "A" else f"{cfg['map_seeds'][0]}-{cfg['map_seeds'][-1]}"
    ss = f"{cfg['subsample_seeds'][0]}-{cfg['subsample_seeds'][-1]}" if "N" in regime else "NA"
    return run_name(EXP, f"{m}{variant_suffix}-D{D}", cfg["model_tag"], TASK, n, seeds, ss)


def select(pred, ydev, knob_values, lambdas):
    """pred: (K, L, seeds, subs, items) int8. Highest mean accuracy; ties -> larger lambda, then first knob."""
    acc = (pred == ydev).mean(-1)                     # (K, L, seeds, subs)
    grid = acc.mean((2, 3))                           # (K, L)
    best, best_key = None, None
    for k in range(grid.shape[0]):
        for l in range(grid.shape[1]):
            lam = lambdas[l] if lambdas[l] is not None else 0.0
            key = (round(float(grid[k, l]), 10), lam, -k)
            if best_key is None or key > best_key:
                best, best_key = (k, l), key
    return best, grid, acc


def save(cfg, m, D, regime, knob_name, knob_values, lambdas, pred, ydev, ids, params, started,
         seeds, extra_name="", restrict_knob=None, timing=None):
    kv = list(knob_values)
    p = pred
    if restrict_knob is not None:
        k0 = kv.index(restrict_knob)
        p, kv = pred[k0:k0 + 1], [restrict_knob]
    (k, l), grid, acc = select(p, ydev, kv, lambdas)
    name = record_name(cfg, m, D, regime, extra_name)
    sel_pred = p[k, l]                                # (seeds, subs, items)
    metrics = {
        "map": m + extra_name, "base_map": m, "D": D, "regime": regime["name"], "N": regime.get("N", "full"),
        "knob_name": knob_name, "knob_values": kv, "lambdas": lambdas,
        "grid_mean_dev_acc": grid.round(5).tolist(),
        "selected": {"knob": kv[k], "lambda": lambdas[l], "mean_dev_acc": float(grid[k, l])},
        "per_run_dev_acc": acc[k, l].round(5).tolist(),  # [seed][subsample]
        "map_seeds": seeds, "subsample_seeds": cfg["subsample_seeds"] if "N" in regime else None,
        "trainable_params": params, "seconds": timing,
    }
    preds = {"ids": ids, "label": ydev.tolist(), "pred": sel_pred.astype(int).tolist()}
    save_run_record(EXP, name, cfg, metrics, preds, started)
    if not extra_name:
        np.savez_compressed(runs_dir(EXP) / f"{name}_grid.npz", pred=pred, knob_values=np.array(kv, dtype=object),
                            lambdas=np.array(lambdas, dtype=object))
    print(f"  -> {name}: selected {knob_name}={kv[k]} lambda={lambdas[l]} mean dev acc {100 * grid[k, l]:.2f}%")


def run_linear_map(cfg, m, D, data, subs, device):
    Htr, Hdev, ytr_t, ydev, ids = data["Htr"], data["Hdev"], data["ytr_t"], data["ydev"], data["ids"]
    ntr, ndev, d = data["ntr"], data["ndev"], data["d"]
    knobs = cfg["knobs"].get(m, [None])
    lambdas = cfg["lambdas"]
    seeds = [0] if m == "A" else cfg["map_seeds"]
    preds = {r["name"]: np.zeros((len(knobs), len(lambdas), len(seeds), subs[r["name"]].shape[0] if "N" in r else 1, ndev),
                                 dtype=np.int8) for r in cfg["regimes"]}
    started, t0 = now_iso(), time.time()
    for si, ms in enumerate(seeds):
        fm = FeatureMap(m, d, D, ms, device=device)
        for ki, knob in enumerate(knobs):
            Rtr = fm(Htr, knob)
            Rdev = fm(Hdev, knob)
            mean, sd = standardize_fit(Rtr)
            Xtr = ((Rtr - mean) / sd).view(ntr, 4, D)
            Xdev = ((Rdev - mean) / sd).view(ndev, 4, D)
            del Rtr, Rdev
            for r in cfg["regimes"]:
                if "N" in r:
                    idx = torch.as_tensor(subs[r["name"]], device=device)
                    Xs, ys = Xtr[idx], ytr_t[idx]                   # (S, N, 4, D), (S, N)
                else:
                    Xs, ys = Xtr.unsqueeze(0), ytr_t.unsqueeze(0)   # (1, ntr, 4, D)
                for li, lam in enumerate(lambdas):
                    W = fit_linear_batched(Xs, ys, lam, cfg["max_iter"])["W"]
                    with torch.no_grad():
                        sc = torch.einsum("ncd,sd->snc", Xdev, W)
                    preds[r["name"]][ki, li, si] = sc.argmax(-1).cpu().numpy()
            print(f"  [{m} D={D}] seed {ms} knob {knob}: done ({time.time() - t0:.0f}s)", flush=True)
            del Xtr, Xdev
            if device == "cuda":
                torch.cuda.empty_cache()
    knob_name = KNOB_NAMES.get(m)
    params = d if m == "A" else D
    for r in cfg["regimes"]:
        save(cfg, m, D, r, knob_name, knobs, lambdas, preds[r["name"]], ydev, ids, params, started, seeds,
             timing=round(time.time() - t0, 1))
        if m in cfg.get("untuned", {}):
            save(cfg, m, D, r, knob_name, knobs, lambdas, preds[r["name"]], ydev, ids, params, started, seeds,
                 extra_name="-untuned", restrict_knob=cfg["untuned"][m])


def run_mlp_map(cfg, D, data, subs):
    Htr_np, Hdev_np, ytr, ydev, ids = data["Htr_np"], data["Hdev_np"], data["ytr"], data["ydev"], data["ids"]
    mc = cfg["mlp"]
    dropouts = cfg["knobs"]["G"]
    seeds = cfg["map_seeds"]
    started, t0 = now_iso(), time.time()
    params = n_params(MLPScorer(data["d"], mc["hidden"], D, 0.0))
    for r in cfg["regimes"]:
        S = subs[r["name"]].shape[0] if "N" in r else 1
        pred = np.zeros((len(dropouts), 1, len(seeds), S, data["ndev"]), dtype=np.int8)
        for ki, p in enumerate(dropouts):
            for si, ms in enumerate(seeds):
                for s in range(S):
                    rows = subs[r["name"]][s] if "N" in r else slice(None)
                    out = train_mlp(Htr_np[rows], ytr[rows], Hdev_np, ydev, hidden=mc["hidden"], out_dim=D, dropout=p,
                                    lr=mc["lr"], weight_decay=mc["weight_decay"], batch_size=mc["batch_size"],
                                    patience=mc["patience"], max_epochs=mc["max_epochs"], seed=ms)
                    pred[ki, 0, si, s] = out["scores"].argmax(1)
            print(f"  [G D={D} {r['name']}] dropout {p}: done ({time.time() - t0:.0f}s)", flush=True)
        save(cfg, "G", D, r, "dropout", dropouts, [None], pred, ydev, ids, params, started, seeds,
             timing=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e2.yaml")
    ap.add_argument("--maps", nargs="*", help="subset of maps (default: all in config)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    global EXP
    EXP = cfg.get("experiment", "E2")   # "E2diag" for the exploratory diagnostic (separate records)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("experiment:", EXP, "| device:", device)

    feats = load_sciq_features(cfg["model_tag"], cfg["layer"], cfg["pooling"])
    Xtr_raw, ytr, _ = feats["train"]
    Xdev_raw, ydev, ids = feats["dev"]
    std = Standardizer().fit(Xtr_raw)
    Htr_np, Hdev_np = std.transform(Xtr_raw), std.transform(Xdev_raw)
    ntr, ndev, d = Htr_np.shape[0], Hdev_np.shape[0], Htr_np.shape[-1]
    data = {
        "Htr_np": Htr_np, "Hdev_np": Hdev_np, "ytr": ytr, "ydev": ydev, "ids": ids, "ntr": ntr, "ndev": ndev, "d": d,
        "Htr": torch.from_numpy(Htr_np.reshape(-1, d)).to(device),
        "Hdev": torch.from_numpy(Hdev_np.reshape(-1, d)).to(device),
        "ytr_t": torch.from_numpy(ytr).to(device),
    }
    subs = {r["name"]: get_subsamples(r["N"], cfg["subsample_seeds"]) for r in cfg["regimes"] if "N" in r}
    print(f"train {Htr_np.shape}, dev {Hdev_np.shape}, subsamples: { {k: v.shape for k, v in subs.items()} }")

    for m in (args.maps or cfg["maps"]):
        for D in cfg["dims"]:
            if m == "A" and D != d:
                continue
            first = record_name(cfg, m, D, cfg["regimes"][0])
            if (runs_dir(EXP) / f"{first}.json").exists() and not args.force:
                print(f"[{m} D={D}] records exist, skipping")
                continue
            print(f"[{m} D={D}] start", flush=True)
            if m == "G":
                run_mlp_map(cfg, D, data, subs)
            else:
                run_linear_map(cfg, m, D, data, subs, device)

    if not args.no_report:
        from analysis.e2_report import build_report
        build_report(cfg)


if __name__ == "__main__":
    main()
