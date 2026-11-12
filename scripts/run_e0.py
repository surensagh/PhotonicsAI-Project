"""E0: pipeline validation report (gate G1).

    python scripts/run_e0.py --config configs/e0.yaml

Needs: splits (build_splits.py) and the feature cache (extract_features.py).
Writes: <workspace>/reports/e0_report.md, e0_results.json, and a run record in runs/E0/.
Check levels: HARD checks must pass for G1; WARN checks are reported but do not block.
"""
import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optics.maps import FScatter, phase_encode  # noqa: E402
from optlm.data_sciq import load_items, load_splits  # noqa: E402
from optlm.features import Standardizer, load_sciq_features  # noqa: E402
from optlm.heads import evaluate_scores, fit_linear_scorer, linear_scores  # noqa: E402
from optlm.utils import (cache_dir, load_config, now_iso, read_json, reports_dir, run_name,  # noqa: E402
                         save_run_record, splits_dir, write_json)

CHECKS = []


def check(section, name, expected, observed, passed, level="HARD"):
    status = "PASS" if passed else ("FAIL" if level == "HARD" else "WARN")
    CHECKS.append({"section": section, "check": name, "expected": expected, "observed": observed,
                   "status": status, "level": level})
    print(f"[{status}] {section} | {name} | observed: {observed}")


# --------------------------------------------------------------------------- 1. data checks
def data_checks():
    spl = load_splits()
    chk = read_json(splits_dir() / "sciq_data_checks.json")
    ex = chk["exact_duplicates"]
    check("Data", "Exact duplicate questions train vs dev/test (before removal)", "0 (any found are removed from train)",
          f"train-dev={ex['train-dev']}, train-test={ex['train-test']}", ex["train-dev"] + ex["train-test"] == 0, "WARN")
    check("Data", "Exact duplicates train vs dev/test after removal", "0",
          chk["exact_duplicates_after_removal_train_vs_devtest"], chk["exact_duplicates_after_removal_train_vs_devtest"] == 0)
    check("Data", "Exact duplicates dev vs test (official splits, cannot be changed)", "0",
          ex["dev-test"], ex["dev-test"] == 0, "WARN")
    near = chk["near_duplicates_train_vs_devtest"]
    check("Data", f"Near duplicates train vs dev/test (Jaccard > {chk['near_duplicate_threshold']})",
          "flagged and removed from train",
          f"{len(near)} pairs ({sum(p['same_correct_answer'] for p in near)} share the correct answer); "
          f"{chk['n_removed_from_train']} train items removed", True)
    check("Data", "Near duplicates dev vs test (informational)", "0", chk["near_duplicates_dev_vs_test"],
          chk["near_duplicates_dev_vs_test"] == 0, "WARN")
    for s, c in chk["chi2_positions"].items():
        check("Data", f"Correct-answer position uniform ({s})", f"chi2 p > {chk['chi2_alpha']}",
              f"counts={c['counts']}, p={c['p']:.3f}", c["p"] > chk["chi2_alpha"])
    check("Data", "Locked split sizes", "train <= 11,679, dev = 1,000, test = 1,000", spl["counts"],
          spl["counts"]["dev"] == 1000 and spl["counts"]["test"] == 1000 and spl["counts"]["train"] <= 11679)
    return spl, chk


# --------------------------------------------------------------------------- 2. trivial baselines (dev)
def trivial_baselines():
    train, dev = load_items("train"), load_items("dev")
    y = np.array([it["label"] for it in dev])
    longest = np.array([int(np.argmax([len(o) for o in it["options"]])) for it in dev])
    shortest = np.array([int(np.argmin([len(o) for o in it["options"]])) for it in dev])
    majority = int(np.bincount([it["label"] for it in train], minlength=4).argmax())
    res = {
        "random_guess": 0.25,
        "longest_answer": float((longest == y).mean()),
        "shortest_answer": float((shortest == y).mean()),
        "answer_position_majority": float((y == majority).mean()),
    }
    check("Floor", "Random guess", "25%", "25.0% (analytic)", True)
    for k in ("longest_answer", "shortest_answer", "answer_position_majority"):
        check("Floor", k.replace("_", " ").capitalize() + " (dev)", "25-30%; higher = exploitable artifact",
              f"{100 * res[k]:.1f}%", res[k] <= 0.30, "WARN")
    return res


# --------------------------------------------------------------------------- 3. cache integrity + determinism
def cache_checks(spl):
    out = {}
    for man_path in sorted(cache_dir().glob("*/manifest.json")):
        man = read_json(man_path)
        tag = man["model_tag"]
        out[tag] = man
        check("Cache", f"{tag}: extraction complete", "complete", man.get("complete"), bool(man.get("complete")))
        if man.get("splits_sha256") != spl["sha256"]:
            check("Cache", f"{tag}: built from the current locked splits", "hashes match", "MISMATCH", False)
        bad_shape, nonfinite = [], 0
        for key, f in man["files"].items():
            split = key.split("/")[-1]
            if f["shape"] != [spl["counts"][split], 4, man["hidden_size"]]:
                bad_shape.append(key)
            nonfinite += f["n_nan"] + f["n_inf"]
        check("Cache", f"{tag}: shapes (n_items, 4, d={man['hidden_size']})", "all files", f"{len(man['files'])} files, "
              f"{len(bad_shape)} wrong", not bad_shape)
        check("Cache", f"{tag}: no NaN / Inf", "0", nonfinite, nonfinite == 0)
        trunc = sum(i["n_truncated"] for i in man["extraction"].values())
        check("Cache", f"{tag}: truncated sequences", "0", trunc, trunc == 0, "WARN")
        det = man.get("determinism")
        if det:
            worst = max(v["max_abs_diff_same_settings"] for v in det["results"].values())
            check("Cache", f"{tag}: determinism, re-extraction max |diff| ({det['split']})", f"< {det['threshold']:.0e}",
                  f"{worst:.2e}", worst < det["threshold"])
            alt = max(v["max_abs_diff_alt_batch_size"] for v in det["results"].values())
            check("Cache", f"{tag}: batch-size invariance (bs {det['alt_batch_size']}) max |diff| (informational)",
                  "small", f"{alt:.2e}", True, "WARN")
        conv = next(iter(man["extraction"].values()))["convention"]
        check("Cache", f"{tag}: hidden_states[last] convention (informational)", "post-final-LN",
              "post-final-LN" if conv["hidden_states_last_is_post_final_ln"] else "raw", True, "WARN")
    if not out:
        check("Cache", "feature cache present", "manifest.json per model", "none found", False)
    return out


# --------------------------------------------------------------------------- 4. optics unit tests
def optics_tests():
    xml_path = reports_dir() / "e0_pytest.xml"
    proc = subprocess.run([sys.executable, "-m", "pytest", str(ROOT / "tests" / "test_optics.py"), "-q",
                           f"--junitxml={xml_path}"], cwd=ROOT, capture_output=True, text=True)
    print(proc.stdout[-2000:])
    results = []
    for tc in ET.parse(xml_path).getroot().iter("testcase"):
        failed = any(child.tag in ("failure", "error") for child in tc)
        skipped = any(child.tag == "skipped" for child in tc)
        results.append({"test": tc.get("name"), "passed": not failed and not skipped})
        check("Optics", tc.get("name"), "pass", "pass" if not failed else "FAIL", not failed and not skipped)
    return results


# --------------------------------------------------------------------------- 5. head sanity
def head_sanity(cfg):
    sc = cfg["sanity"]
    lam, chance, tol = sc["lambda"], sc["chance"], sc["tolerance_pp"] / 100
    feats = load_sciq_features(cfg["model_tag"], cfg["layer"], cfg["pooling"])
    Xtr_raw, ytr, _ = feats["train"]
    Xdev_raw, ydev, _ = feats["dev"]
    std = Standardizer().fit(Xtr_raw)
    Xtr, Xdev = std.transform(Xtr_raw), std.transform(Xdev_raw)
    rng = np.random.default_rng(sc["seed"])
    res = {}

    # (a) training loss falls monotonically on 1,000 items
    n = sc["n_loss_items"]
    fit = fit_linear_scorer(Xtr[:n], ytr[:n], lam)
    L = np.array(fit["losses"])
    increases = np.diff(L) > 1e-7 * np.maximum(1.0, np.abs(L[1:]))
    res["loss_curve"] = {"initial": float(L[0]), "final": float(L[-1]), "n_iter": fit["n_iter"],
                         "n_increases": int(increases.sum())}
    check("Head", f"Training loss monotone (linear head, {n} items, lambda={lam})", "never increases",
          f"{L[0]:.4f} -> {L[-1]:.4f} in {fit['n_iter']} iters, {int(increases.sum())} increases",
          not increases.any() and L[-1] < L[0])

    def acc_of(Xa, ya, Xb, yb):
        f = fit_linear_scorer(Xa, ya, lam)
        return evaluate_scores(linear_scores(Xb, f["w"]), yb)["accuracy"]

    # informational: real labels with the same fixed lambda
    res["real_labels_acc"] = acc_of(Xtr, ytr, Xdev, ydev)
    check("Head", f"Real labels, same fixed lambda (reference)", "> chance", f"{100 * res['real_labels_acc']:.1f}%",
          res["real_labels_acc"] > chance + tol, "WARN")

    # (b) permuted labels
    res["permuted_labels_acc"] = acc_of(Xtr, rng.permutation(ytr), Xdev, ydev)
    check("Head", "Permuted training labels -> dev accuracy", f"25% +- {sc['tolerance_pp']} pp",
          f"{100 * res['permuted_labels_acc']:.1f}%", abs(res["permuted_labels_acc"] - chance) <= tol)

    # (c) Gaussian-noise inputs of the same shape
    res["noise_inputs_acc"] = acc_of(rng.standard_normal(Xtr.shape).astype(np.float32), ytr,
                                     rng.standard_normal(Xdev.shape).astype(np.float32), ydev)
    check("Head", "Gaussian-noise inputs -> dev accuracy", f"25% +- {sc['tolerance_pp']} pp",
          f"{100 * res['noise_inputs_acc']:.1f}%", abs(res["noise_inputs_acc"] - chance) <= tol)

    # (d) beta -> 0 through the optical map: features constant, accuracy at chance
    d = Xtr.shape[-1]
    fmap = FScatter(d=d, D=sc["beta_zero_D"], seed=sc["seed"])
    rtr = fmap(Xtr[:n], beta=sc["beta_zero"])
    rdev = fmap(Xdev, beta=sc["beta_zero"])
    rel_spread = float(((rtr.reshape(-1, rtr.shape[-1]).std(0)) / rtr.reshape(-1, rtr.shape[-1]).mean(0)).max())
    rstd = Standardizer().fit(rtr)
    res["beta_zero_acc"] = acc_of(rstd.transform(rtr), ytr[:n], rstd.transform(rdev), ydev)
    res["beta_zero_rel_spread"] = rel_spread
    check("Head", f"Optical map beta={sc['beta_zero']:.0e}: features constant", "relative SD ~ 0",
          f"{rel_spread:.1e}", rel_spread < 1e-4)
    check("Head", f"Optical map beta={sc['beta_zero']:.0e}: dev accuracy", f"25% +- {sc['tolerance_pp']} pp",
          f"{100 * res['beta_zero_acc']:.1f}%", abs(res["beta_zero_acc"] - chance) <= tol)
    return res


# --------------------------------------------------------------------------- report
def write_report(spl, results, path):
    hard_fail = [c for c in CHECKS if c["status"] == "FAIL"]
    warns = [c for c in CHECKS if c["status"] == "WARN"]
    lines = [
        "# E0 report: pipeline validation",
        "",
        f"Generated: {now_iso()}  ",
        f"Git commit: `{results['git']['commit']}`{' (DIRTY working tree)' if results['git']['dirty'] else ''}  ",
        f"Model for head checks: {results['config']['model_tag']} / {results['config']['layer']} / {results['config']['pooling']}",
        "",
        f"## Gate G1: **{'PASSED' if not hard_fail else 'FAILED'}**",
        "",
        f"{len(CHECKS)} checks: {sum(c['status'] == 'PASS' for c in CHECKS)} pass, {len(hard_fail)} fail, {len(warns)} warnings.",
        "WARN = reported, does not block G1 (informational or a property of the official dataset).",
        "",
        "## Locked split hashes (SHA-256 of the ID lists)",
        "",
        "| split | n | sha256 |",
        "|---|---|---|",
    ]
    for s in ("train", "dev", "test"):
        lines.append(f"| {s} | {spl['counts'][s]} | `{spl['sha256'][s]}` |")
    lines += ["", f"Items file sha256: `{spl['items_file_sha256']}`", ""]
    section = None
    for c in CHECKS:
        if c["section"] != section:
            section = c["section"]
            lines += ["", f"## {section}", "", "| status | check | expected | observed |", "|---|---|---|---|"]
        lines.append(f"| {c['status']} | {c['check']} | {c['expected']} | {c['observed']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e0.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    started = now_iso()

    spl, chk = data_checks()
    floors = trivial_baselines()
    cache = cache_checks(spl)
    optics = optics_tests()
    sanity = head_sanity(cfg) if cfg["model_tag"] in cache else {}

    from optlm.utils import git_info
    results = {"config": cfg, "git": git_info(), "checks": CHECKS, "floor_baselines": floors,
               "head_sanity": sanity, "optics_tests": optics,
               "g1_passed": not any(c["status"] == "FAIL" for c in CHECKS)}
    write_json(reports_dir() / "e0_results.json", results)
    write_report(spl, results, reports_dir() / "e0_report.md")
    save_run_record("E0", run_name("E0", "validation", cfg["model_tag"], "sciq", "all", "NA", "NA"), cfg,
                    {"g1_passed": results["g1_passed"], "floor_baselines": floors, "head_sanity": sanity},
                    None, started, extra={"checks": CHECKS})
    print("\nG1:", "PASSED" if results["g1_passed"] else "FAILED", "->", reports_dir() / "e0_report.md")
    if not results["g1_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
