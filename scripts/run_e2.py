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
# Current placement cell (E3/E4): {"layer", "pooling", "integration", "tag"}; None = the E2 default cell.
CELL = None


def cell_tag(c):
    return f"{c['layer']}{c['pooling']}-{'par' if c['integration'] == 'parallel' else 'rep'}"


def record_name(cfg, m, D, regime, variant_suffix=""):
    n = regime.get("N", "full")
    seeds = "0" if m == "A" else f"{cfg['map_seeds'][0]}-{cfg['map_seeds'][-1]}"
    ss = f"{cfg['subsample_seeds'][0]}-{cfg['subsample_seeds'][-1]}" if "N" in regime else "NA"
    cell = f"-{CELL['tag']}" if CELL else ""
    return run_name(EXP, f"{m}{variant_suffix}{cell}-D{D}", cfg["model_tag"], TASK, n, seeds, ss)


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
        "layer": (CELL or cfg)["layer"], "pooling": (CELL or cfg)["pooling"],
        "integration": (CELL or {}).get("integration", "replacement"), "cell": CELL["tag"] if CELL else None,
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
    parallel = bool(CELL) and CELL["integration"] == "parallel" and m != "A"
    preds = {r["name"]: np.zeros((len(knobs), len(lambdas), len(seeds), subs[r["name"]].shape[0] if "N" in r else 1, ndev),
                                 dtype=np.int8) for r in cfg["regimes"]}
    started, t0 = now_iso(), time.time()
    for si, ms in enumerate(seeds):
        fm = FeatureMap(m, d, D, ms, device=device)
        for ki, knob in enumerate(knobs):
            Rtr = fm(Htr, knob)
            Rdev = fm(Hdev, knob)
            mean, sd = standardize_fit(Rtr)
            Rtr, Rdev = (Rtr - mean) / sd, (Rdev - mean) / sd
            if parallel:  # parallel integration: the head sees [h; r] (both standardized)
                Rtr, Rdev = torch.cat([Htr, Rtr], 1), torch.cat([Hdev, Rdev], 1)
            nf = Rtr.shape[1]
            Xtr, Xdev = Rtr.view(ntr, 4, nf), Rdev.view(ndev, 4, nf)
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
    params = d if m == "A" else (D + d if parallel else D)
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


def build_data(cfg, layer, pooling, device):
    feats = load_sciq_features(cfg["model_tag"], layer, pooling)
    Xtr_raw, ytr, _ = feats["train"]
    Xdev_raw, ydev, ids = feats["dev"]
    std = Standardizer().fit(Xtr_raw)
    Htr_np, Hdev_np = std.transform(Xtr_raw), std.transform(Xdev_raw)
    ntr, ndev, d = Htr_np.shape[0], Hdev_np.shape[0], Htr_np.shape[-1]
    return {
        "Htr_np": Htr_np, "Hdev_np": Hdev_np, "ytr": ytr, "ydev": ydev, "ids": ids, "ntr": ntr, "ndev": ndev, "d": d,
        "Htr": torch.from_numpy(Htr_np.reshape(-1, d)).to(device),
        "Hdev": torch.from_numpy(Hdev_np.reshape(-1, d)).to(device),
        "ytr_t": torch.from_numpy(ytr).to(device),
    }


def resolve_cells(cfg):
    """None for E2 (single default cell); else a list of placement cells."""
    if "cells_from" in cfg:  # the frozen decision D-E3 written by analysis/e3_report.py
        from optlm.utils import read_json, reports_dir
        dec = read_json(reports_dir() / cfg["cells_from"])
        cells = [{k: dec[k] for k in ("layer", "pooling", "integration")}]
    elif "cells" in cfg:
        c = cfg["cells"]
        cells = [{"layer": l, "pooling": p, "integration": i}
                 for l in c["layers"] for p in c["poolings"] for i in c["integrations"]]
    else:
        return None
    for c in cells:
        c["tag"] = cell_tag(c)
    return cells


def run_maps(cfg, maps, dims, data, subs, device, force):
    d = data["d"]
    where = f" {CELL['tag']}" if CELL else ""
    for m in maps:
        # identity A exists only at D = d (and only as replacement): run it once, whatever dims are listed
        for D in ([d] if m == "A" else dims):
            if m == "A" and CELL and CELL["integration"] == "parallel":
                continue
            first = record_name(cfg, m, D, cfg["regimes"][0])
            if (runs_dir(EXP) / f"{first}.json").exists() and not force:
                print(f"[{m} D={D}{where}] records exist, skipping")
                continue
            print(f"[{m} D={D}{where}] start", flush=True)
            if m == "G":
                run_mlp_map(cfg, D, data, subs)
            else:
                run_linear_map(cfg, m, D, data, subs, device)


def mean_acc(cfg, m, D, regime):
    from optlm.utils import read_json
    p = runs_dir(EXP) / f"{record_name(cfg, m, D, regime)}.json"
    return read_json(p)["metrics"]["selected"]["mean_dev_acc"] if p.exists() else None


def main():
    global EXP, CELL
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e2.yaml")
    ap.add_argument("--maps", nargs="*", help="subset of maps (default: all in config)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    EXP = cfg.get("experiment", "E2")   # E2, E2diag, E2v2, E3, E4: separate record folders
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("experiment:", EXP, "| device:", device)
    maps = args.maps or cfg["maps"]
    subs = {r["name"]: get_subsamples(r["N"], cfg["subsample_seeds"]) for r in cfg["regimes"] if "N" in r}

    cells = resolve_cells(cfg)
    for cell in (cells or [None]):
        CELL = cell
        layer = cell["layer"] if cell else cfg["layer"]
        pooling = cell["pooling"] if cell else cfg["pooling"]
        data = build_data(cfg, layer, pooling, device)
        d = data["d"]
        dims = [int(round(f * d)) for f in cfg["dims_over_d"]] if "dims_over_d" in cfg else cfg["dims"]
        print(f"cell {cell['tag'] if cell else 'default'}: train {data['Htr_np'].shape}, dev {data['Hdev_np'].shape}, "
              f"D = {dims}", flush=True)
        run_maps(cfg, maps, dims, data, subs, device, args.force)

        # E4 rule: add D = factor * d only if a listed map still gains > min_gain from 2d to 4d (either regime)
        ext = cfg.get("extend_dims")
        if ext:
            gains = {}
            for m in ext["maps"]:
                for r in cfg["regimes"]:
                    a2, a4 = mean_acc(cfg, m, 2 * d, r), mean_acc(cfg, m, 4 * d, r)
                    if a2 is not None and a4 is not None:
                        gains[f"{m} {r['name']}"] = a4 - a2
            trigger = any(g > ext["min_gain"] for g in gains.values())
            print("extension check (acc 4d - 2d, pp):", {k: round(100 * v, 2) for k, v in gains.items()},
                  "->", f"run D = {ext['factor']}d" if trigger else "no extension")
            if trigger:
                run_maps(cfg, maps, [ext["factor"] * d], data, subs, device, args.force)

    if not args.no_report:
        import importlib
        importlib.import_module(cfg.get("report", "analysis.e2_report")).build_report(cfg)


if __name__ == "__main__":
    main()
