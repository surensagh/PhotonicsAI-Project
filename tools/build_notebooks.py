"""Generate the Colab notebooks in colab/ from one place, so their setup cells stay identical.

    python tools/build_notebooks.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab"

SETUP = r'''# ---- Setup (edit the two lines below once) ------------------------------------------
GITHUB_REPO = "surensagh/PhotonicsAI-Project"                    # repo that holds this code
WORKSPACE   = "/content/drive/MyDrive/optical_lm_workspace"      # splits, cache, runs, reports
# -------------------------------------------------------------------------------------
import os, subprocess, sys
from google.colab import drive

drive.mount("/content/drive")
os.makedirs(WORKSPACE, exist_ok=True)
os.environ["OPTLM_WORKSPACE"] = WORKSPACE
os.environ["HF_HOME"] = "/content/hf_cache"          # downloads stay on the fast local disk

# Private repo: add a GitHub token as a Colab secret named GH_TOKEN (key icon in the left bar).
try:
    from google.colab import userdata
    token = userdata.get("GH_TOKEN")
except Exception:
    token = None
url = f"https://{token}@github.com/{GITHUB_REPO}.git" if token else f"https://github.com/{GITHUB_REPO}.git"
REPO_DIR = "/content/" + GITHUB_REPO.split("/")[-1]
cmd = ["git", "clone", "-q", url, REPO_DIR] if not os.path.isdir(REPO_DIR) else ["git", "-C", REPO_DIR, "pull", "-q", "--ff-only"]
res = subprocess.run(cmd, capture_output=True, text=True)
if res.returncode != 0:
    msg = res.stderr.replace(token, "***") if token else res.stderr
    raise RuntimeError("git failed (check GITHUB_REPO / GH_TOKEN):\n" + msg)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
print("repo   :", REPO_DIR)
print("commit :", subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip())
print("dirty  :", bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"]).strip()))
print("workspace:", WORKSPACE)'''

GPU_CHECK = r'''import torch
assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > T4 GPU"
print(torch.cuda.get_device_name(0))'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells, gpu):
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []}}
    if gpu:
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = "T4"
    return {"nbformat": 4, "nbformat_minor": 0, "metadata": meta, "cells": cells}


NOTEBOOKS = {
    "01_splits_and_data_checks.ipynb": (False, [
        md("""
# 01 · Locked SciQ splits and E0 data checks

**Runtime:** CPU is enough. **Time:** ~2 minutes.

What this does (protocol 3.2 and E0 data checks):
1. Downloads SciQ (`allenai/sciq`): official train 11,679 / validation 1,000 / test 1,000.
2. Shuffles the 4 options of each question with seed 0. The correct-answer position is a balanced
   random permutation, so each position holds the correct answer for 1/4 of the questions.
3. Leakage: exact duplicates of normalized question text across splits, and train questions with
   token-Jaccard > 0.9 to any dev/test question. Flagged train items are removed.
4. χ² test that the correct-answer position is uniform in each split.
5. Saves ID lists + SHA-256 hashes to `WORKSPACE/splits/`.

Running it again never overwrites the splits: it rebuilds them in a temp folder and checks that the
hashes match (reproducibility check).
"""),
        code(SETUP),
        code("!python scripts/build_splits.py --config configs/splits.yaml"),
        md("### Review the flagged near-duplicate pairs by eye"),
        code(r'''import json, os
chk = json.load(open(os.path.join(WORKSPACE, "splits", "sciq_data_checks.json")))
pairs = chk["near_duplicates_train_vs_devtest"]
print(f"{len(pairs)} flagged pairs; {sum(p['same_correct_answer'] for p in pairs)} share the correct answer\n")
for p in pairs[:30]:
    print(f"J={p['jaccard']:.2f} same_answer={p['same_correct_answer']}  {p['train_id']} vs {p['other_id']}")
    print("   train:", p["train_question"])
    print("   other:", p["other_question"])'''),
        md("### Look at a few formatted inputs (the exact text the LM sees)"),
        code(r'''from optlm.data_sciq import load_items, format_option
for it in load_items("dev")[:2]:
    for j, o in enumerate(it["options"]):
        mark = "  <-- correct" if j == it["label"] else ""
        print(repr(format_option(it["question"], o)) + mark)
    print()'''),
        md("""
### Before moving on
Copy the three split hashes printed above into today's notebook entry (`notebook/TEMPLATE.md`).
Optionally download `WORKSPACE/splits/sciq_splits.json` into `data/splits/` in the repo and commit it.
"""),
    ]),
    "02_extract_features.ipynb": (True, [
        md("""
# 02 · Extract and cache representations

**Runtime:** T4 GPU. **Time:** roughly 10–20 minutes for both models (most of it is writing to Drive).

For Pythia-160M-deduped (revision `step143000`) and GPT-2 Small, every option text
`Question: {q}\\nAnswer: {option}` is run through the frozen model in fp16, then pooled:

| layer name | what it is |
|---|---|
| `L3`, `L6` | raw residual stream after block 3 / 6 (`hidden_states[3]`, `[6]`) |
| `L12` | last block output **after** the final LayerNorm (default "final layer") |
| `L12raw` | last block output **before** the final LayerNorm |

Poolings: `mean` (over non-padding tokens) and `last` (last non-padding token).
Saved as float32 `WORKSPACE/cache/<model>/<layer>_<pooling>/<split>.npy`, shape `(n_items, 4, d)`.

E0 checks done here: dev is re-extracted with identical settings and compared with the cache
(max |diff| must be < 1e-3); a batch-size-32 re-extraction is reported for information.

**Resumable:** if Colab disconnects, just run again. Finished models are skipped.
"""),
        code(SETUP),
        code(GPU_CHECK),
        code("!python scripts/extract_features.py --config configs/extract.yaml"),
        md("### Cache summary"),
        code(r'''import json, glob, os
for path in sorted(glob.glob(os.path.join(WORKSPACE, "cache", "*", "manifest.json"))):
    m = json.load(open(path))
    print(f"{m['model_tag']}: complete={m['complete']}, files={len(m['files'])}, d={m['hidden_size']}, "
          f"total {m.get('seconds_total')}s")
    for k, v in m.get("determinism", {}).get("results", {}).items():
        print(f"   {k:14s} same-settings diff {v['max_abs_diff_same_settings']:.1e}   "
              f"alt-batch diff {v['max_abs_diff_alt_batch_size']:.1e}   mean|h| {v['mean_abs_value']:.3f}")'''),
    ]),
    "03_e0_validation.ipynb": (False, [
        md("""
# 03 · E0 report and gate G1

**Runtime:** CPU works; GPU is faster for the head checks. **Time:** a few minutes.
Needs notebooks 01 and 02 to have finished.

Collects every E0 check into `WORKSPACE/reports/e0_report.md`:
- data checks from notebook 01, the floor baselines (random, longest answer, answer position)
- cache integrity and determinism from notebook 02
- all optical-map unit tests (`tests/test_optics.py`)
- head sanity: monotone training loss, permuted labels → 25 ± 3 %, Gaussian-noise inputs → 25 ± 3 %,
  optical map with β → 0 → constant features and chance accuracy

**G1 passes only if every HARD check passes.** WARN lines are reported but do not block
(informational, or properties of the official dataset we are not allowed to change).
"""),
        code(SETUP),
        md("### All unit tests (data, heads, optics)"),
        code("!python -m pytest -q tests"),
        md("### E0 report"),
        code("!python scripts/run_e0.py --config configs/e0.yaml"),
        code(r'''from IPython.display import Markdown, display
display(Markdown(open(os.path.join(WORKSPACE, "reports", "e0_report.md"), encoding="utf-8").read()))'''),
    ]),
    "04_e1_baselines.ipynb": (True, [
        md("""
# 04 · E1 baselines (Table 1)

**Runtime:** T4 GPU. **Time:** ~10 minutes.
Needs notebooks 01–03 (G1 passed).

SciQ, Pythia-160M, layer `L12`, mean pooling, full training data, evaluated on **dev** only
(the locked test set is not touched until E12).

| method | details |
|---|---|
| Zero-shot LM | log-prob of the option tokens given `Question: …\\nAnswer:`; per-token normalized (primary), per-character and unnormalized also reported |
| Linear probe (map A) | s = wᵀh, softmax over the 4 options, L-BFGS, λ ∈ {1e-4 … 1e2}; deterministic, so one run |
| MLP head (map G) | 768 → 256 → GELU → dropout → 768 → score; AdamW lr 1e-3, wd 1e-2, batch 64, early stopping patience 10; dropout ∈ {0, 0.1, 0.3, 0.5} × 3 seeds |

Selection rule (fixed in advance): highest dev accuracy (mean over seeds for the MLP); ties go to
the stronger regularization. These are post-selection dev numbers, as the protocol intends.

**Gate:** if the linear probe is below 35 %, fix the representation before E2 (try `last` pooling and `L6`).
"""),
        code(SETUP),
        code(GPU_CHECK),
        code("!python scripts/run_e1.py --config configs/e1.yaml"),
        code(r'''from IPython.display import Markdown, display
display(Markdown(open(os.path.join(WORKSPACE, "reports", "e1_table1.md"), encoding="utf-8").read()))'''),
        md("""
Every run wrote a JSON record with config, commit hash, environment and per-item predictions to
`WORKSPACE/runs/E1/`. Write today's notebook entry from `notebook/TEMPLATE.md`.
"""),
    ]),
    "05_e2_operators.ipynb": (True, [
        md("""
# 05 · E2: optical vs digital feature maps (Table 2, Figure 2, gate G2)

**Runtime:** T4 GPU. **Time:** roughly 45–75 minutes. **Resumable:** finished maps are skipped on re-run.
Needs notebooks 01–02 (cache) and G1 passed.

**Question.** Does the phase-encoded optical map do something that generic random-feature expansion does not?

**Design** (protocol E2). SciQ, Pythia-160M, `L12`, mean pooling, linear scoring head, tuned on dev.
9 maps (A, B, C, D, E, F-scatter, F-fourier, G, H) × D ∈ {768, 3072} × {N = 100 with 10 paired subsamples, full data}
× 3 map seeds, plus F with β fixed at π (untuned) to show how much of F's score comes from tuning.

**Tuning budget** (protocol 3.6): knob maps (D tanh s, F β, H σ) get 4 knob values × 7 λ = 28 configs;
knob-free maps (A, B, C, E) get 7; G gets 4 dropout values. Selection = highest mean dev accuracy over
seeds and subsamples; ties go to the stronger regularization.

### Interpretation matrix (written before seeing any results)

| Observed (best tuned F) | Conclusion | Next step |
|---|---|---|
| F > A, B and F > H, C (CIs separate) | Optical structure adds something beyond random features | Proceed; E10 must explain why |
| F > A, B but F ≈ H or C | Benefit is random-feature expansion; optics is a physical implementation of it | Proceed with reframed H2 (Section 7) |
| F ≈ A | Expansion adds nothing at this layer | Run E3 and E4 before concluding |
| F < A | Map destroys information (check β range, normalization) | Debug; re-run E0 tests |

**Gate G2 (Nov 23):** F beats A at N = 100 (95% paired-bootstrap CI excludes 0) for some D.
"""),
        code(SETUP),
        code(GPU_CHECK),
        code("!python -m pytest -q tests/test_featuremaps.py tests/test_stats_batched.py"),
        code("!python scripts/run_e2.py --config configs/e2.yaml"),
        md("### Table 2"),
        code(r'''from IPython.display import Markdown, Image, display
display(Markdown(open(os.path.join(WORKSPACE, "reports", "e2_table2.md"), encoding="utf-8").read()))'''),
        md("### Figure 2"),
        code(r'''display(Image(os.path.join(WORKSPACE, "reports", "e2_fig2.png")))'''),
        md("""
The report is regenerated from the run records alone (`python analysis/e2_report.py`); no hand-edited numbers.
Raw records: `WORKSPACE/runs/E2/` (selected-config predictions in .json, the full prediction grid in .npz).
"""),
    ]),
}


def main():
    OUT.mkdir(exist_ok=True)
    for name, (gpu, cells) in NOTEBOOKS.items():
        (OUT / name).write_text(json.dumps(notebook(cells, gpu), indent=1, ensure_ascii=False) + "\n",
                                encoding="utf-8", newline="\n")
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
