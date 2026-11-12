"""Unit tests for the scoring heads and metrics on synthetic data."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optlm.features import Standardizer  # noqa: E402
from optlm.heads import evaluate_scores, fit_linear_scorer, linear_scores, select_lambda, train_mlp  # noqa: E402


def _synthetic(n, d=20, seed=0, signal=1.0):
    rng = np.random.default_rng(seed)
    w_true = rng.standard_normal(d)
    X = rng.standard_normal((n, 4, d)).astype(np.float32)
    y = rng.integers(0, 4, n)
    X[np.arange(n), y] += (signal * w_true / np.linalg.norm(w_true)).astype(np.float32)
    return X, y


def test_linear_head_learns_and_loss_monotone():
    X, y = _synthetic(2000, signal=3.0)
    fit = fit_linear_scorer(X[:1500], y[:1500], lam=1e-3)
    L = np.array(fit["losses"])
    assert np.all(np.diff(L) <= 1e-7 * np.maximum(1, np.abs(L[1:])))
    acc = evaluate_scores(linear_scores(X[1500:], fit["w"]), y[1500:])["accuracy"]
    assert acc > 0.8


def test_linear_head_deterministic():
    X, y = _synthetic(500)
    a = fit_linear_scorer(X, y, 1e-2)["w"]
    b = fit_linear_scorer(X, y, 1e-2)["w"]
    assert np.array_equal(a, b)


def test_select_lambda_tie_breaks_to_larger():
    X, y = _synthetic(300, signal=0.0)
    Xz = np.zeros_like(X)  # every lambda gives all-zero scores -> identical accuracy
    sel = select_lambda(Xz, y, Xz, y, [1e-3, 1e-1, 10.0])
    assert sel["best"]["lambda"] == 10.0


def test_metrics():
    scores = np.array([[5.0, 0, 0, 0], [0, 5.0, 0, 0], [0, 0, 0, 5.0]])
    m = evaluate_scores(scores, np.array([0, 1, 2]))
    assert abs(m["accuracy"] - 2 / 3) < 1e-12 and m["nll"] > 0 and 0 <= m["ece"] <= 1


def test_standardizer_constant_features_to_zero():
    X = np.ones((10, 4, 3), dtype=np.float32)
    X[..., 0] += np.random.default_rng(0).standard_normal((10, 4))
    X[..., 1] += 1e-9  # constant up to rounding
    Z = Standardizer().fit(X).transform(X)
    assert np.all(Z[..., 1:] == 0) and abs(Z[..., 0].std() - 1) < 1e-5


def test_mlp_runs_and_learns():
    X, y = _synthetic(1200, signal=3.0)
    r = train_mlp(X[:1000], y[:1000], X[1000:], y[1000:], hidden=32, out_dim=20, dropout=0.1, lr=1e-3,
                  weight_decay=1e-2, batch_size=64, patience=5, max_epochs=50, seed=0)
    assert evaluate_scores(r["scores"], y[1000:])["accuracy"] > 0.6
    assert r["n_params"] == 20 * 32 + 32 + 32 * 20 + 20 + 20
