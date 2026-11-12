"""Scoring heads and metrics (protocol 3.5, 3.7).

Multiple-choice scoring: every option j of a question gets a score s_j = w^T r_j, and the head is
trained with softmax cross-entropy over the options. No bias (it cancels in the softmax).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _t(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=device())


# --------------------------------------------------------------------------- metrics
def evaluate_scores(scores: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    """Accuracy, NLL and expected calibration error (15 equal-width bins) of option scores (n, C)."""
    scores = np.asarray(scores, dtype=np.float64)
    z = scores - scores.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    p = np.exp(logp)
    pred = scores.argmax(1)  # ties -> lowest index
    correct = pred == y
    conf = p.max(1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return {
        "accuracy": float(correct.mean()),
        "nll": float(-logp[np.arange(len(y)), y].mean()),
        "ece": float(ece),
        "n": int(len(y)),
    }


def predictions_record(ids: list[str], scores: np.ndarray, y: np.ndarray) -> dict:
    """Per-item predictions (needed later for paired bootstrap / McNemar)."""
    scores = np.asarray(scores)
    return {
        "ids": list(ids),
        "label": [int(v) for v in y],
        "pred": [int(v) for v in scores.argmax(1)],
        "scores": np.round(scores, 6).tolist(),
    }


# --------------------------------------------------------------------------- linear scorer (L-BFGS)
def fit_linear_scorer(X: np.ndarray, y: np.ndarray, lam: float, max_iter: int = 500, tol: float = 1e-10) -> dict:
    """Minimize mean CE(X w, y) + (lam / 2) ||w||^2 with L-BFGS (strong-Wolfe line search).

    X: (n, C, D) features, y: (n,) index of the correct option.
    Returns {"w": np.ndarray (D,), "losses": per-iteration objective, "n_iter", "seconds"}.
    The objective is convex and the optimizer deterministic, so there is no optimizer seed.
    """
    t0 = time.time()
    lam = float(lam)
    Xt, yt = _t(X), _t(y, torch.long)
    w = torch.zeros(X.shape[-1], device=Xt.device, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1.0, max_iter=1, max_eval=25, history_size=20,
                            tolerance_grad=1e-9, tolerance_change=1e-14, line_search_fn="strong_wolfe")

    def objective():
        return F.cross_entropy(Xt @ w, yt) + 0.5 * lam * (w * w).sum()

    def closure():
        opt.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    with torch.no_grad():
        losses = [float(objective())]
    for _ in range(max_iter):
        opt.step(closure)
        with torch.no_grad():
            losses.append(float(objective()))
        if abs(losses[-2] - losses[-1]) <= tol * max(1.0, abs(losses[-1])):
            break
    return {"w": w.detach().cpu().numpy(), "losses": losses, "n_iter": len(losses) - 1,
            "seconds": time.time() - t0}


def fit_linear_batched(X: torch.Tensor, y: torch.Tensor, lam: float, max_iter: int = 500, tol: float = 1e-10) -> dict:
    """S independent problems with the same lambda, solved in one L-BFGS run.

    X: (S, n, C, D) tensor, y: (S, n) long tensor. Minimizes
        sum_s [ mean_n CE(X_s w_s, y_s) + (lam / 2) ||w_s||^2 ]
    The problems share no parameters, so the joint minimum is each problem's own minimum.
    Problems are grouped by lambda (never mixing lambdas), so they are similarly conditioned.
    Returns {"W": (S, D) tensor, "n_iter", "losses"}.
    """
    lam = float(lam)
    S, n, C, Dd = X.shape
    W = torch.zeros((S, Dd), device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([W], lr=1.0, max_iter=1, max_eval=25, history_size=20,
                            tolerance_grad=1e-9, tolerance_change=1e-14, line_search_fn="strong_wolfe")
    yflat = y.reshape(-1)

    def objective():
        scores = torch.einsum("sncd,sd->snc", X, W)
        ce = F.cross_entropy(scores.reshape(-1, C), yflat, reduction="none").reshape(S, n).mean(1)
        return (ce + 0.5 * lam * (W * W).sum(1)).sum()

    def closure():
        opt.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    with torch.no_grad():
        losses = [float(objective())]
    for _ in range(max_iter):
        opt.step(closure)
        with torch.no_grad():
            losses.append(float(objective()))
        if abs(losses[-2] - losses[-1]) <= tol * max(1.0, abs(losses[-1])):
            break
    return {"W": W.detach(), "n_iter": len(losses) - 1, "losses": losses}


def linear_scores(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float32) @ np.asarray(w, dtype=np.float32)


def select_lambda(Xtr, ytr, Xdev, ydev, lambdas, max_iter=500) -> dict:
    """Fit every lambda on train, score on the tuning split; pick highest accuracy (tie -> larger lambda)."""
    results = []
    for lam in lambdas:
        fit = fit_linear_scorer(Xtr, ytr, lam, max_iter)
        sc = linear_scores(Xdev, fit["w"])
        m = evaluate_scores(sc, ydev)
        results.append({"lambda": lam, "metrics": m, "n_iter": fit["n_iter"], "seconds": fit["seconds"],
                        "w": fit["w"], "scores": sc})
    best = max(results, key=lambda r: (r["metrics"]["accuracy"], r["lambda"]))
    return {"best": best, "all": results}


# --------------------------------------------------------------------------- MLP head (map G + linear score)
class MLPScorer(nn.Module):
    """r = W2 GELU(W1 h) (hidden 256, dropout after GELU), then s = w^T r."""

    def __init__(self, d: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.map = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim))
        self.score = nn.Linear(out_dim, 1, bias=False)

    def forward(self, x):  # x: (n, C, d) -> (n, C)
        return self.score(self.map(x)).squeeze(-1)


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def train_mlp(Xtr, ytr, Xdev, ydev, *, hidden, out_dim, dropout, lr, weight_decay, batch_size,
              patience, max_epochs, seed) -> dict:
    """AdamW, early stopping on tuning-split accuracy with patience (epochs); best epoch restored."""
    t0 = time.time()
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    Xt, yt = _t(Xtr), _t(ytr, torch.long)
    Xd = _t(Xdev)
    model = MLPScorer(Xtr.shape[-1], hidden, out_dim, dropout).to(Xt.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_acc, best_state, best_epoch, since = -1.0, None, 0, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(yt), generator=gen).to(Xt.device)
        total = 0.0
        for s in range(0, len(yt), batch_size):
            idx = perm[s : s + batch_size]
            loss = F.cross_entropy(model(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            acc = float((model(Xd).argmax(1).cpu().numpy() == ydev).mean())
        history.append({"epoch": epoch, "train_loss": total / len(yt), "dev_acc": acc})
        if acc > best_acc:
            best_acc, best_state, best_epoch, since = acc, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = model(Xd).cpu().numpy()
    return {"scores": scores, "best_epoch": best_epoch, "epochs_run": len(history), "history": history,
            "n_params": n_params(model), "seconds": time.time() - t0}
