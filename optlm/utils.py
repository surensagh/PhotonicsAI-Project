"""Paths, config loading, hashing, environment capture and run records (protocol section 9)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- paths
def workspace() -> Path:
    """Root for large artifacts. On Colab, set OPTLM_WORKSPACE to a Google Drive folder."""
    root = Path(os.environ.get("OPTLM_WORKSPACE", REPO_ROOT / "workspace"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def ws_path(*parts: str) -> Path:
    p = workspace().joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def splits_dir() -> Path:
    d = workspace() / "splits"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    d = workspace() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_dir(experiment: str) -> Path:
    d = workspace() / "runs" / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d


def reports_dir() -> Path:
    d = workspace() / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, encoding="utf-8") as f:
        return _fix_sci_floats(yaml.safe_load(f))


_SCI = __import__("re").compile(r"^[-+]?\d+(\.\d*)?[eE][-+]?\d+$")


def _fix_sci_floats(obj):
    """PyYAML (YAML 1.1) reads '1.0e1' as a string; convert such strings to float."""
    if isinstance(obj, dict):
        return {k: _fix_sci_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fix_sci_floats(v) for v in obj]
    if isinstance(obj, str) and _SCI.match(obj):
        return float(obj)
    return obj


# --------------------------------------------------------------------------- hashing
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_json(obj: Any) -> str:
    """Hash of a canonical JSON encoding (sorted keys, no whitespace)."""
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- environment
def git_info() -> dict:
    def _run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return ""

    commit = _run("rev-parse", "HEAD") or "unknown"
    dirty = bool(_run("status", "--porcelain", "--untracked-files=no"))
    return {"commit": commit, "dirty": dirty}


def env_info() -> dict:
    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "scipy", "torch", "transformers", "datasets"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch

        info["cuda"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        info["cuda"] = False
        info["gpu"] = None
    return info


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- json helpers
class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_NpEncoder, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- run records
def run_name(exp: str, variant: str, model: str, task: str, n: int | str, ms: int | str, ss: int | str) -> str:
    """Protocol 9: E{n}-{variant}-{model}-{task}-N{N}-ms{map seed}-ss{subsample seed}."""
    return f"{exp}-{variant}-{model}-{task}-N{n}-ms{ms}-ss{ss}"


def save_run_record(
    exp: str,
    name: str,
    config: dict,
    metrics: dict,
    predictions: dict | None,
    started: str,
    extra: dict | None = None,
) -> Path:
    record = {
        "name": name,
        "experiment": exp,
        "config": config,
        "git": git_info(),
        "env": env_info(),
        "started": started,
        "finished": now_iso(),
        "metrics": metrics,
        "predictions": predictions,
    }
    if extra:
        record.update(extra)
    path = runs_dir(exp) / f"{name}.json"
    write_json(path, record)
    return path
