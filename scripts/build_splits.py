"""Build the locked SciQ splits and run the E0 data checks.

    python scripts/build_splits.py --config configs/splits.yaml

If the splits already exist, they are NOT overwritten: the script rebuilds them in a temporary
folder and checks that the hashes are identical (reproducibility check).
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optlm.data_sciq import build_splits, load_splits  # noqa: E402
from optlm.utils import load_config, splits_dir  # noqa: E402


def _print_summary(spl: dict, chk: dict) -> None:
    print("Official counts:", spl["official_counts"])
    print("Final counts:   ", spl["counts"])
    print("Exact duplicates:", chk["exact_duplicates"])
    near = chk["near_duplicates_train_vs_devtest"]
    print("Near duplicates (train vs dev/test, Jaccard > %.2f): %d pairs, %d share the correct answer"
          % (chk["near_duplicate_threshold"], len(near), sum(p["same_correct_answer"] for p in near)))
    print("Near duplicates dev vs test (informational):", chk["near_duplicates_dev_vs_test"])
    print("Removed from train:", chk["n_removed_from_train"])
    for s, c in chk["chi2_positions"].items():
        print(f"  chi2 {s:5s} counts={c['counts']} p={c['p']:.3f}")
    for s in ("train", "dev", "test"):
        print(f"  sha256 {s:5s} {spl['sha256'][s]}")
    print("items file sha256:", spl["items_file_sha256"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/splits.yaml")
    ap.add_argument("--force", action="store_true", help="overwrite existing splits (never do this after E1)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if (splits_dir() / "sciq_splits.json").exists() and not args.force:
        print("Splits already exist -> reproducibility check (rebuild in a temp folder, compare hashes).")
        old = load_splits()
        real_ws = os.environ.get("OPTLM_WORKSPACE")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["OPTLM_WORKSPACE"] = tmp
            try:
                new = build_splits(cfg)["splits"]
            finally:
                if real_ws is None:
                    os.environ.pop("OPTLM_WORKSPACE", None)
                else:
                    os.environ["OPTLM_WORKSPACE"] = real_ws
        same = old["sha256"] == new["sha256"] and old["items_file_sha256"] == new["items_file_sha256"]
        for s in ("train", "dev", "test"):
            print(f"  {s:5s} n={old['counts'][s]:6d} sha256={old['sha256'][s]}")
        print("Reproducible:", same)
        if not same:
            sys.exit("ERROR: rebuilt splits differ from the stored ones. Nothing was overwritten; investigate.")
        return

    res = build_splits(cfg)
    _print_summary(res["splits"], res["checks"])
    print("Written to", splits_dir())


if __name__ == "__main__":
    main()
