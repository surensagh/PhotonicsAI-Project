"""SciQ: locked splits, option shuffling and E0 data checks (protocol 3.2, 3.3, E0).

Outputs (in <workspace>/splits/):
  sciq_items.jsonl     one line per question: id, split, question, options (shuffled), label, correct_answer
  sciq_splits.json     ID lists per split + SHA-256 hashes (the locked split definition)
  sciq_data_checks.json  leakage, near-duplicate and chi-square results used by the E0 report
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
from scipy import sparse
from scipy.stats import chisquare

from .utils import now_iso, read_json, sha256_file, sha256_json, splits_dir, write_json

# HF split name -> protocol split name, and a fixed index used in the shuffle seed
SPLIT_MAP = {"train": "train", "validation": "dev", "test": "test"}
SPLIT_INDEX = {"train": 0, "dev": 1, "test": 2}

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Lowercase, replace punctuation with spaces, collapse whitespace."""
    s = _PUNCT.sub(" ", s.lower())
    return _WS.sub(" ", s).strip()


def token_set(s: str) -> frozenset:
    return frozenset(normalize_text(s).split())


# --------------------------------------------------------------------------- loading
def load_raw(hf_dataset: str = "allenai/sciq") -> tuple[dict[str, list[dict]], dict]:
    """Return {protocol_split: [row, ...]} in the official order, plus provenance info."""
    from datasets import load_dataset

    ds = load_dataset(hf_dataset)
    raw, prov = {}, {"hf_dataset": hf_dataset, "content_sha256": {}, "fingerprint": {}}
    for hf_name, name in SPLIT_MAP.items():
        rows = [dict(r) for r in ds[hf_name]]
        raw[name] = rows
        fields = [[r["question"], r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]] for r in rows]
        prov["content_sha256"][name] = sha256_json(fields)
        prov["fingerprint"][name] = getattr(ds[hf_name], "_fingerprint", None)
    return raw, prov


# --------------------------------------------------------------------------- option shuffle
def assign_options(rows: list[dict], split: str, seed: int = 0) -> list[dict]:
    """Shuffle the 4 options of every question with a fixed seed.

    The correct-answer position is drawn as a *balanced* random permutation of
    [0,1,2,3,0,1,2,3,...] over the split, so every position holds the correct answer
    for n/4 questions (stratified shuffle). The three distractors fill the remaining
    slots in random order. This keeps the answer position uniform by construction
    instead of only in expectation.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, SPLIT_INDEX[split]]))
    n = len(rows)
    labels = rng.permutation(np.arange(n) % 4)
    items = []
    for i, (r, lab) in enumerate(zip(rows, labels)):
        distractors = [r["distractor1"], r["distractor2"], r["distractor3"]]
        order = rng.permutation(3)
        options, k = [None] * 4, 0
        for pos in range(4):
            if pos == lab:
                options[pos] = r["correct_answer"]
            else:
                options[pos] = distractors[order[k]]
                k += 1
        items.append(
            {
                "id": f"sciq-{split}-{i:05d}",
                "split": split,
                "question": r["question"],
                "options": [str(o) for o in options],
                "label": int(lab),
                "correct_answer": r["correct_answer"],
            }
        )
    return items


# --------------------------------------------------------------------------- leakage checks
def exact_duplicates(a_items: list[dict], b_items: list[dict]) -> list[tuple[int, int]]:
    """Index pairs (i in a, j in b) whose normalized question text is identical."""
    index: dict[str, list[int]] = {}
    for j, it in enumerate(b_items):
        index.setdefault(normalize_text(it["question"]), []).append(j)
    pairs = []
    for i, it in enumerate(a_items):
        for j in index.get(normalize_text(it["question"]), []):
            pairs.append((i, j))
    return pairs


def _binary_matrix(sets: list[frozenset], vocab: dict[str, int]) -> sparse.csr_matrix:
    rows, cols = [], []
    for i, s in enumerate(sets):
        for t in s:
            rows.append(i)
            cols.append(vocab[t])
    data = np.ones(len(rows), dtype=np.int32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(sets), len(vocab)))


def jaccard_pairs(a_texts: list[str], b_texts: list[str], threshold: float, chunk: int = 2000) -> list[tuple[int, int, float]]:
    """All (i, j, jaccard) with token-Jaccard(a_i, b_j) > threshold, computed with sparse products."""
    a_sets = [token_set(t) for t in a_texts]
    b_sets = [token_set(t) for t in b_texts]
    vocab: dict[str, int] = {}
    for s in a_sets + b_sets:
        for t in s:
            vocab.setdefault(t, len(vocab))
    A = _binary_matrix(a_sets, vocab)
    B = _binary_matrix(b_sets, vocab)
    la = np.asarray(A.sum(1)).ravel()
    lb = np.asarray(B.sum(1)).ravel()
    out = []
    for start in range(0, A.shape[0], chunk):
        inter = (A[start : start + chunk] @ B.T).toarray().astype(np.float64)
        union = la[start : start + chunk, None] + lb[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            jac = np.where(union > 0, inter / union, 0.0)
        ii, jj = np.nonzero(jac > threshold)
        for i, j in zip(ii, jj):
            out.append((int(start + i), int(j), float(jac[i, j])))
    return out


def chi2_positions(items: list[dict]) -> dict:
    counts = Counter(it["label"] for it in items)
    obs = [counts.get(k, 0) for k in range(4)]
    stat, p = chisquare(obs)
    return {"counts": obs, "chi2": float(stat), "p": float(p)}


# --------------------------------------------------------------------------- main build
def build_splits(cfg: dict) -> dict:
    raw, prov = load_raw(cfg["hf_dataset"])
    seed = int(cfg["seed"])
    thr = float(cfg["near_duplicate"]["jaccard_threshold"])

    items = {s: assign_options(raw[s], s, seed) for s in ("train", "dev", "test")}
    official_counts = {s: len(v) for s, v in items.items()}

    # 1) exact duplicates of normalized question text across splits
    exact = {
        "train-dev": exact_duplicates(items["train"], items["dev"]),
        "train-test": exact_duplicates(items["train"], items["test"]),
        "dev-test": exact_duplicates(items["dev"], items["test"]),
    }

    # 2) near duplicates: token-Jaccard > thr between train and dev/test questions
    near = []
    for other in ("dev", "test"):
        pairs = jaccard_pairs([it["question"] for it in items["train"]], [it["question"] for it in items[other]], thr)
        for i, j, jac in pairs:
            a, b = items["train"][i], items[other][j]
            near.append(
                {
                    "train_id": a["id"],
                    "other_id": b["id"],
                    "other_split": other,
                    "jaccard": round(jac, 4),
                    "same_correct_answer": normalize_text(a["correct_answer"]) == normalize_text(b["correct_answer"]),
                    "train_question": a["question"],
                    "other_question": b["question"],
                }
            )
    dev_test_near = jaccard_pairs([it["question"] for it in items["dev"]], [it["question"] for it in items["test"]], thr)

    # Remove from train every item that is an exact or near duplicate of a dev/test item.
    remove_idx = {i for key in ("train-dev", "train-test") for i, _ in exact[key]}
    train_index = {it["id"]: k for k, it in enumerate(items["train"])}
    remove_idx |= {train_index[p["train_id"]] for p in near}
    removed_ids = sorted(items["train"][i]["id"] for i in remove_idx)
    items["train"] = [it for k, it in enumerate(items["train"]) if k not in remove_idx]

    # Verify nothing remains after removal.
    residual_exact = len(exact_duplicates(items["train"], items["dev"])) + len(exact_duplicates(items["train"], items["test"]))

    # 3) answer-position uniformity
    chi2 = {s: chi2_positions(items[s]) for s in ("train", "dev", "test")}

    # ---- write files
    d = splits_dir()
    items_path = d / "sciq_items.jsonl"
    import json

    with open(items_path, "w", encoding="utf-8", newline="\n") as f:
        for s in ("train", "dev", "test"):
            for it in items[s]:
                f.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")

    ids = {s: [it["id"] for it in items[s]] for s in ("train", "dev", "test")}
    splits = {
        "task": "sciq",
        "seed": seed,
        "created": now_iso(),
        "provenance": prov,
        "official_counts": official_counts,
        "counts": {s: len(v) for s, v in ids.items()},
        "removed_from_train": removed_ids,
        "ids": ids,
        "sha256": {s: sha256_json(v) for s, v in ids.items()},
        "items_file_sha256": sha256_file(items_path),
    }
    write_json(d / "sciq_splits.json", splits)

    checks = {
        "exact_duplicates": {k: len(v) for k, v in exact.items()},
        "exact_duplicates_after_removal_train_vs_devtest": residual_exact,
        "near_duplicate_threshold": thr,
        "near_duplicates_train_vs_devtest": near,
        "near_duplicates_dev_vs_test": len(dev_test_near),
        "n_removed_from_train": len(removed_ids),
        "chi2_positions": chi2,
        "chi2_alpha": cfg["chi2_alpha"],
    }
    write_json(d / "sciq_data_checks.json", checks)
    return {"splits": splits, "checks": checks}


# --------------------------------------------------------------------------- loaders used downstream
def load_splits() -> dict:
    return read_json(splits_dir() / "sciq_splits.json")


def load_items(split: str) -> list[dict]:
    """Items of one split in the locked order of sciq_splits.json. Verifies the items-file hash."""
    import json

    d = splits_dir()
    spl = load_splits()
    if sha256_file(d / "sciq_items.jsonl") != spl["items_file_sha256"]:
        raise RuntimeError("sciq_items.jsonl does not match the hash recorded in sciq_splits.json")
    by_id = {}
    with open(d / "sciq_items.jsonl", encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            if it["split"] == split:
                by_id[it["id"]] = it
    return [by_id[i] for i in spl["ids"][split]]


def format_option(question: str, option: str) -> str:
    """Pre-registered SciQ template (protocol 3.3). Support paragraph excluded."""
    return f"Question: {question}\nAnswer: {option}"
