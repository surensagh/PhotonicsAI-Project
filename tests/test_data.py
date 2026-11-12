"""Unit tests for the SciQ split utilities (no download needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optlm.data_sciq import assign_options, chi2_positions, exact_duplicates, jaccard_pairs, normalize_text  # noqa: E402


def _rows(n):
    return [{"question": f"What is thing number {i}?", "correct_answer": f"right{i}",
             "distractor1": f"wrong{i}a", "distractor2": f"wrong{i}b", "distractor3": f"wrong{i}c"} for i in range(n)]


def test_normalize_text():
    assert normalize_text("  What's the  ANSWER?\n") == "what s the answer"


def test_assign_options_balanced_and_correct():
    rows = _rows(400)
    items = assign_options(rows, "train", seed=0)
    labels = np.array([it["label"] for it in items])
    assert np.bincount(labels, minlength=4).tolist() == [100, 100, 100, 100]
    for r, it in zip(rows, items):
        assert it["options"][it["label"]] == r["correct_answer"]
        assert sorted(it["options"]) == sorted([r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]])
    assert chi2_positions(items)["p"] > 0.99


def test_assign_options_deterministic_and_split_specific():
    rows = _rows(50)
    a = assign_options(rows, "dev", 0)
    b = assign_options(rows, "dev", 0)
    c = assign_options(rows, "test", 0)
    assert a == [dict(x) for x in b]
    assert [x["label"] for x in a] != [x["label"] for x in c]


def test_exact_duplicates():
    a = [{"question": "What is H2O?"}, {"question": "Name a gas."}]
    b = [{"question": "what is h2o"}, {"question": "Something else"}]
    assert exact_duplicates(a, b) == [(0, 0)]


def test_jaccard_pairs():
    a = ["the cell membrane controls what enters the cell", "photosynthesis happens in chloroplasts"]
    b = ["The cell membrane controls what enters the cell!", "chloroplasts are green", "unrelated words here"]
    pairs = jaccard_pairs(a, b, 0.9)
    assert [(i, j) for i, j, _ in pairs] == [(0, 0)]
    assert abs(pairs[0][2] - 1.0) < 1e-12
    # one word changed in a 7-distinct-word question -> 6/8 = 0.75, not flagged at 0.9
    assert jaccard_pairs(["a b c d e f g"], ["a b c d e f h"], 0.9) == []
