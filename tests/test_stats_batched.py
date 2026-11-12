"""Batched linear fit equals individual fits; bootstrap and Holm behave as specified."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optlm.heads import evaluate_scores, fit_linear_batched, fit_linear_scorer  # noqa: E402
from optlm.stats import PairedBootstrap, holm  # noqa: E402


def _problem(seed, n=100, d=200):  # n < D: the low-data regime
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(d)
    X = rng.standard_normal((n, 4, d)).astype(np.float32)
    y = rng.integers(0, 4, n)
    X[np.arange(n), y] += (0.3 * w / np.linalg.norm(w) * 3).astype(np.float32)
    return X, y


def test_batched_matches_individual():
    probs = [_problem(s) for s in range(5)]
    Xdev, ydev = _problem(99, n=400)
    for lam in (1e-3, 1e-1, 10.0):
        Xb = torch.from_numpy(np.stack([p[0] for p in probs]))
        yb = torch.from_numpy(np.stack([p[1] for p in probs]))
        W = fit_linear_batched(Xb, yb, lam)["W"].numpy()
        for s, (X, y) in enumerate(probs):
            w = fit_linear_scorer(X, y, lam)["w"]
            acc_i = evaluate_scores(Xdev @ w, ydev)["accuracy"]
            acc_b = evaluate_scores(Xdev @ W[s], ydev)["accuracy"]
            rel = np.linalg.norm(W[s] - w) / max(np.linalg.norm(w), 1e-12)
            assert abs(acc_i - acc_b) <= 0.0051, (lam, s, acc_i, acc_b)
            assert rel < 0.02 or lam < 1e-2, (lam, s, rel)


def test_bootstrap_ci_and_paired_diff():
    rng = np.random.default_rng(0)
    a = (rng.random((10, 1000)) < 0.6).astype(np.float32)
    b = a.copy()
    flip = rng.random((10, 1000)) < 0.05
    b[flip] = 0  # b is a strictly worse copy of a
    bs = PairedBootstrap(10, 1000, B=2000)
    ci = bs.mean_ci(a)
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"]
    d = bs.diff(a, b)
    assert d["diff"] > 0 and d["ci_excludes_0"] and d["p"] < 0.01
    same = bs.diff(a, a)
    assert same["diff"] == 0 and not same["ci_excludes_0"]


def test_holm():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
