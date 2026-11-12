"""Paired bootstrap, Holm correction and effect sizes (protocol 3.8).

Correctness matrices have shape (S, I): S subsamples (already averaged over map seeds), I test items.
Every comparison uses the SAME resampling indices (paired design): resample subsamples and items
with replacement, B = 10,000 times.
"""
from __future__ import annotations

import numpy as np


class PairedBootstrap:
    def __init__(self, n_subsamples: int, n_items: int, B: int = 10_000, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.S, self.I, self.B = n_subsamples, n_items, B
        # resampling expressed as counts (multinomial), so a statistic is a weighted mean
        self.item_counts = rng.multinomial(n_items, np.full(n_items, 1 / n_items), size=B).astype(np.float32)
        if n_subsamples > 1:
            self.sub_counts = rng.multinomial(n_subsamples, np.full(n_subsamples, 1 / n_subsamples), size=B).astype(np.float32)
        else:
            self.sub_counts = np.ones((B, 1), dtype=np.float32)

    def replicates(self, M: np.ndarray) -> np.ndarray:
        """Bootstrap replicates of the grand mean of M (S, I)."""
        M = np.asarray(M, dtype=np.float32)
        assert M.shape == (self.S, self.I), M.shape
        per_sub = self.item_counts @ M.T  # (B, S): item-weighted sums per subsample
        return (per_sub * self.sub_counts).sum(1) / (self.S * self.I)

    def mean_ci(self, M: np.ndarray, alpha: float = 0.05) -> dict:
        reps = self.replicates(M)
        lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
        return {"mean": float(np.mean(M)), "ci_low": float(lo), "ci_high": float(hi)}

    def diff(self, Ma: np.ndarray, Mb: np.ndarray, alpha: float = 0.05) -> dict:
        """Paired difference a - b: mean, 95% CI, two-sided bootstrap p, Cohen's d over items."""
        D = np.asarray(Ma, np.float32) - np.asarray(Mb, np.float32)
        reps = self.replicates(D)
        lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
        p = 2 * min((reps <= 0).mean(), (reps >= 0).mean())
        item_diff = D.mean(0)
        sd = item_diff.std(ddof=1)
        return {"diff": float(D.mean()), "ci_low": float(lo), "ci_high": float(hi), "p": float(min(1.0, p)),
                "cohens_d": float(item_diff.mean() / sd) if sd > 0 else 0.0,
                "ci_excludes_0": bool(lo > 0 or hi < 0)}


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test for one head-to-head run (binomial on discordant pairs)."""
    from scipy.stats import binomtest

    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return {"a_only": n10, "b_only": n01, "p": float(p)}
