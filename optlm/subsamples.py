"""Paired low-data subsamples (protocol 3.8, E5): every method sees identical training subsets.

Subsample (N, ss) = rng([N, ss]).choice(train pool, N, replace=False), sorted.
Saved once to <workspace>/splits/subsamples_N<N>.json with item IDs and a SHA-256; later calls
load the file and verify it reproduces exactly.
"""
from __future__ import annotations

import numpy as np

from .data_sciq import load_splits
from .utils import read_json, sha256_json, splits_dir, write_json


def draw(n_pool: int, N: int, ss: int) -> np.ndarray:
    return np.sort(np.random.default_rng([N, ss]).choice(n_pool, N, replace=False))


def get_subsamples(N: int, seeds: list[int]) -> np.ndarray:
    """(len(seeds), N) indices into the locked train split."""
    spl = load_splits()
    train_ids = spl["ids"]["train"]
    idx = np.stack([draw(len(train_ids), N, ss) for ss in seeds])
    ids = {str(ss): [train_ids[i] for i in row] for ss, row in zip(seeds, idx)}
    path = splits_dir() / f"subsamples_N{N}.json"
    record = {"N": N, "train_sha256": spl["sha256"]["train"], "ids": ids, "sha256": sha256_json(ids)}
    if path.exists():
        old = read_json(path)
        for ss in map(str, seeds):
            if ss in old["ids"] and old["ids"][ss] != ids[ss]:
                raise RuntimeError(f"subsample N={N} ss={ss} does not reproduce the stored one")
        merged = {**old["ids"], **ids}
        record["ids"], record["sha256"] = merged, sha256_json(merged)
    write_json(path, record)
    return idx
