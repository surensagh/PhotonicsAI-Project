"""E2diag follow-up (EXPLORATORY): where does F lose accuracy in its small-beta (linear) limit?

    python scripts/diag_linear_limit.py --config configs/e2_diag.yaml

Theory. To first order in beta, F-scatter gives r_j = |a_j|^2 + sum_k c_jk phi_k with
a_j = sum_k U_jk and c_jk = -2 Im(conj(a_j) U_jk). Every row of c sums to zero (global-phase
invariance), so the linear limit of F is "a linear map of tanh(h) with the common-mode direction
(the all-ones vector) removed". This script trains the same linear scoring head (lambda grid,
selection on dev, full data and N = 100) on representations that isolate each step:

  h                         raw standardized representation (= map A)
  h minus common mode       only the phase-invariance loss
  tanh(h)                   only the squashing in the phase encoding
  tanh(h) minus common mode both losses = predicted linear limit of F
  F first-order term        phi @ c^T computed from F-scatter's actual U (seed 0, D = d)
  F-scatter, beta = pi/256  the real map at a tiny beta (seed 0, D = d)
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optlm.featuremaps import FeatureMap, standardize_fit  # noqa: E402
from optlm.features import Standardizer, load_sciq_features  # noqa: E402
from optlm.heads import fit_linear_batched  # noqa: E402
from optlm.subsamples import get_subsamples  # noqa: E402
from optlm.utils import load_config, reports_dir, write_json  # noqa: E402


def evaluate(Xtr, ytr_t, Xdev, ydev, subs, lambdas):
    """Best mean dev accuracy over lambda, for full data and for the N = 100 subsamples."""
    out = {}
    for regime, idx in (("full", None), ("N100", subs)):
        if idx is None:
            Xs, ys = Xtr.unsqueeze(0), ytr_t.unsqueeze(0)
        else:
            t = torch.as_tensor(idx, device=Xtr.device)
            Xs, ys = Xtr[t], ytr_t[t]
        accs = []
        for lam in lambdas:
            W = fit_linear_batched(Xs, ys, lam)["W"]
            with torch.no_grad():
                pred = torch.einsum("ncd,sd->snc", Xdev, W).argmax(-1).cpu().numpy()
            accs.append(float((pred == ydev).mean()))
        k = int(np.argmax(accs))
        out[regime] = {"acc": accs[k], "lambda": lambdas[k], "by_lambda": accs}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e2_diag.yaml")
    cfg = load_config(ap.parse_args().config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats = load_sciq_features(cfg["model_tag"], cfg["layer"], cfg["pooling"])
    Xtr_raw, ytr, _ = feats["train"]
    Xdev_raw, ydev, _ = feats["dev"]
    std = Standardizer().fit(Xtr_raw)
    Htr = torch.from_numpy(std.transform(Xtr_raw).reshape(-1, Xtr_raw.shape[-1])).to(device)
    Hdev = torch.from_numpy(std.transform(Xdev_raw).reshape(-1, Xdev_raw.shape[-1])).to(device)
    ntr, ndev, d = Xtr_raw.shape[0], Xdev_raw.shape[0], Xtr_raw.shape[-1]
    ytr_t = torch.from_numpy(ytr).to(device)
    subs = get_subsamples(100, cfg["subsample_seeds"])
    lambdas = cfg["lambdas"]

    ones = torch.ones(d, device=device) / math.sqrt(d)
    remove_cm = lambda X: X - (X @ ones)[:, None] * ones[None, :]  # noqa: E731

    fm = FeatureMap("F-scatter", d, d, 0, device=device)
    Ur, Ui = fm.Ur, fm.Ui
    ar, ai = Ur.sum(1), Ui.sum(1)                       # a_j = sum_k U_jk
    C = -2 * (ar[:, None] * Ui - ai[:, None] * Ur)      # c_jk = -2 Im(conj(a_j) U_jk)
    row_sum_max = float(C.sum(1).abs().max())

    reps = {
        "h (= map A)": lambda H: H,
        "h minus common mode": remove_cm,
        "tanh(h)": torch.tanh,
        "tanh(h) minus common mode (predicted F limit)": lambda H: remove_cm(torch.tanh(H)),
        "F first-order term (from U, seed 0)": lambda H: ((torch.tanh(H) + 1) / 2) @ C.T,
        "F-scatter, beta = pi/256 (seed 0, D = d)": lambda H: fm(H, math.pi / 256),
    }
    results = {"tanh_saturation_frac_abs_h_gt_2": float((Htr.abs() > 2).float().mean()),
               "c_row_sum_max_abs": row_sum_max, "rows": {}}
    for name, f in reps.items():
        Rtr, Rdev = f(Htr), f(Hdev)
        mean, sd = standardize_fit(Rtr)
        Xtr = ((Rtr - mean) / sd).view(ntr, 4, -1)
        Xdev = ((Rdev - mean) / sd).view(ndev, 4, -1)
        results["rows"][name] = evaluate(Xtr, ytr_t, Xdev, ydev, subs, lambdas)
        r = results["rows"][name]
        print(f"{name:48s} full {100 * r['full']['acc']:.1f}%   N=100 {100 * r['N100']['acc']:.1f}%", flush=True)

    write_json(reports_dir() / "e2diag_linear_limit.json", results)
    L = ["# E2diag follow-up: the linear (small-β) limit of F — EXPLORATORY", "",
         "Same linear scoring head, λ grid and dev selection as E2diag. Full data = 11,642 train questions;",
         "N = 100 = mean over the 10 paired subsamples (no map-seed averaging; seed 0 where a seed applies).", "",
         "| representation | full data dev acc % | N = 100 dev acc % |", "|---|---|---|"]
    for name, r in results["rows"].items():
        L.append(f"| {name} | {100 * r['full']['acc']:.1f} (λ={r['full']['lambda']:g}) | "
                 f"{100 * r['N100']['acc']:.1f} (λ={r['N100']['lambda']:g}) |")
    L += ["", f"Fraction of standardized h entries with |h| > 2 (where tanh saturates): "
              f"{100 * results['tanh_saturation_frac_abs_h_gt_2']:.2f}%  ",
          f"Max |row sum| of the first-order matrix c (theory: 0): {row_sum_max:.2e}", ""]
    (reports_dir() / "e2diag_linear_limit.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("->", reports_dir() / "e2diag_linear_limit.md")


if __name__ == "__main__":
    main()
