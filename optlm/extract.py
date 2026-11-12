"""Representation extraction and on-disk cache (protocol 3.1, 3.3; E0 checks 3 and 4).

Cache layout:  <workspace>/cache/<model_tag>/<layer>_<pooling>/<split>.npy   shape (n_items, 4, d), float32
               <workspace>/cache/<model_tag>/manifest.json

Layer names (for a model with n blocks):
  L<k>, k < n  -> hidden_states[k]: raw residual stream after block k
  L<n>         -> output of the last block AFTER the final LayerNorm (default "final layer")
  L<n>raw      -> output of the last block BEFORE the final LayerNorm
Both versions of the last layer are computed explicitly (forward hook + the model's own final
norm module), so the result does not depend on how a transformers version fills hidden_states.
"""
from __future__ import annotations

import re
import time

import numpy as np
import torch

from .data_sciq import format_option, load_items, load_splits
from .utils import cache_dir, env_info, now_iso, read_json, sha256_file, write_json

_LAYER_RE = re.compile(r"^L(\d+)(raw)?$")


# --------------------------------------------------------------------------- model loading
def load_base_model(hf_id: str, revision: str, device: str, dtype: torch.dtype):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    # Load in default precision, then cast: robust across transformers versions (dtype/torch_dtype kwarg).
    model = AutoModel.from_pretrained(hf_id, revision=revision)
    model.to(device=device, dtype=dtype).eval()
    return tok, model


def blocks_and_final_norm(model):
    """(list of transformer blocks, final norm module) for GPT-NeoX (Pythia) and GPT-2 base models."""
    if hasattr(model, "layers") and hasattr(model, "final_layer_norm"):  # GPTNeoXModel
        return model.layers, model.final_layer_norm
    if hasattr(model, "h") and hasattr(model, "ln_f"):  # GPT2Model
        return model.h, model.ln_f
    raise ValueError(f"Unsupported architecture: {type(model).__name__}")


def parse_layer(name: str, n_blocks: int) -> tuple[int, bool]:
    m = _LAYER_RE.match(name)
    if not m:
        raise ValueError(f"Bad layer name {name!r}")
    k, raw = int(m.group(1)), bool(m.group(2))
    if not 1 <= k <= n_blocks:
        raise ValueError(f"Layer {name} outside 1..{n_blocks}")
    if raw and k != n_blocks:
        raise ValueError(f"'raw' suffix only applies to the last layer (L{n_blocks}raw)")
    return k, raw


# --------------------------------------------------------------------------- extraction
@torch.no_grad()
def extract_texts(
    model,
    tok,
    texts: list[str],
    layers: list[str],
    poolings: list[str],
    batch_size: int,
    max_length: int,
    device: str,
) -> tuple[dict[tuple[str, str], np.ndarray], dict]:
    """Pooled representations for every (layer, pooling). Returns float32 arrays of shape (len(texts), d)."""
    blocks, final_norm = blocks_and_final_norm(model)
    n_blocks = len(blocks)
    parsed = {name: parse_layer(name, n_blocks) for name in layers}
    d = model.config.hidden_size

    enc = tok(texts, add_special_tokens=True, truncation=False)["input_ids"]
    n_truncated = sum(len(x) > max_length for x in enc)
    enc = [x[:max_length] for x in enc]
    lengths = np.array([len(x) for x in enc])
    order = np.argsort(-lengths, kind="stable")  # long first: stable memory, less padding

    out = {(l, p): np.empty((len(texts), d), dtype=np.float32) for l in layers for p in poolings}
    captured = {}

    def hook(_mod, _inp, output):
        captured["raw_last"] = output[0] if isinstance(output, (tuple, list)) else output

    handle = blocks[-1].register_forward_hook(hook)
    convention = None
    try:
        for start in range(0, len(texts), batch_size):
            idx = order[start : start + batch_size]
            batch = [enc[i] for i in idx]
            L = max(len(x) for x in batch)
            ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(batch), L), dtype=torch.long)
            for r, x in enumerate(batch):
                ids[r, : len(x)] = torch.tensor(x)
                mask[r, : len(x)] = 1
            ids, mask = ids.to(device), mask.to(device)
            res = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            hs = res.hidden_states
            raw_last = captured["raw_last"]
            ln_last = final_norm(raw_last)

            if convention is None:  # record how this transformers version fills hidden_states[n]
                diff_ln = (hs[n_blocks].float() - ln_last.float()).abs().max().item()
                diff_raw = (hs[n_blocks].float() - raw_last.float()).abs().max().item()
                convention = {
                    "hidden_states_last_vs_final_ln_maxdiff": diff_ln,
                    "hidden_states_last_vs_raw_maxdiff": diff_raw,
                    "hidden_states_last_is_post_final_ln": diff_ln < diff_raw,
                }

            m = mask.unsqueeze(-1).float()
            counts = m.sum(1)
            last_pos = mask.sum(1) - 1
            ar = torch.arange(len(batch), device=device)
            for name, (k, raw) in parsed.items():
                if k == n_blocks:
                    h = raw_last if raw else ln_last
                else:
                    h = hs[k]
                h = h.float()
                for p in poolings:
                    if p == "mean":
                        v = (h * m).sum(1) / counts
                    elif p == "last":
                        v = h[ar, last_pos]
                    else:
                        raise ValueError(p)
                    out[(name, p)][idx] = v.cpu().numpy()
    finally:
        handle.remove()

    info = {"n_texts": len(texts), "n_truncated": int(n_truncated), "max_tokens": int(lengths.max()),
            "mean_tokens": float(lengths.mean()), "convention": convention}
    return out, info


def split_texts(items: list[dict]) -> list[str]:
    return [format_option(it["question"], o) for it in items for o in it["options"]]


def cache_path(model_tag: str, layer: str, pooling: str, split: str):
    p = cache_dir() / model_tag / f"{layer}_{pooling}" / f"{split}.npy"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_cached(model_tag: str, layer: str, pooling: str, split: str) -> np.ndarray:
    return np.load(cache_path(model_tag, layer, pooling, split)).astype(np.float32)


# --------------------------------------------------------------------------- main
def run_extraction(cfg: dict, force: bool = False, only: list[str] | None = None) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, cfg["dtype"]) if device == "cuda" else torch.float32
    if device == "cpu":
        print("WARNING: no GPU found; extracting in float32 on CPU (slow, and not the protocol's fp16).")
    spl = load_splits()
    summaries = {}

    for mcfg in cfg["models"]:
        tag = mcfg["tag"]
        if only and tag not in only:
            continue
        man_path = cache_dir() / tag / "manifest.json"
        if man_path.exists() and not force and read_json(man_path).get("complete"):
            print(f"[{tag}] cache complete, skipping (use --force to redo)")
            summaries[tag] = read_json(man_path)
            continue

        t0 = time.time()
        tok, model = load_base_model(mcfg["hf_id"], mcfg["revision"], device, dtype)
        n_blocks = len(blocks_and_final_norm(model)[0])
        manifest = {
            "model_tag": tag, "hf_id": mcfg["hf_id"], "revision": mcfg["revision"],
            "hidden_size": model.config.hidden_size, "n_blocks": n_blocks,
            "dtype": str(dtype), "device": device, "started": now_iso(), "env": env_info(),
            "splits_sha256": spl["sha256"], "files": {}, "extraction": {}, "complete": False,
        }
        for split in cfg["splits"]:
            items = load_items(split)
            texts = split_texts(items)
            print(f"[{tag}] {split}: {len(items)} items, {len(texts)} sequences ...", flush=True)
            ts = time.time()
            arrays, info = extract_texts(model, tok, texts, cfg["layers"], cfg["poolings"],
                                         cfg["batch_size"], cfg["max_length"], device)
            info["seconds"] = round(time.time() - ts, 1)
            manifest["extraction"][split] = info
            for (layer, pool), arr in arrays.items():
                arr = arr.reshape(len(items), 4, -1)
                p = cache_path(tag, layer, pool, split)
                np.save(p, arr)
                manifest["files"][f"{layer}_{pool}/{split}"] = {
                    "shape": list(arr.shape),
                    "n_nan": int(np.isnan(arr).sum()),
                    "n_inf": int(np.isinf(arr).sum()),
                    "mean": float(arr.mean()),
                    "sd": float(arr.std()),
                    "sha256": sha256_file(p),
                }
            write_json(man_path, manifest)  # checkpoint after each split

        # ---- E0 check 3: determinism (identical settings, re-extract and compare with cache)
        dcfg = cfg["determinism"]
        dsplit = dcfg["split"]
        texts = split_texts(load_items(dsplit))
        again, _ = extract_texts(model, tok, texts, cfg["layers"], cfg["poolings"],
                                 cfg["batch_size"], cfg["max_length"], device)
        alt, _ = extract_texts(model, tok, texts, cfg["layers"], cfg["poolings"],
                               dcfg["alt_batch_size"], cfg["max_length"], device)
        det = {}
        for (layer, pool), arr in again.items():
            cached = np.load(cache_path(tag, layer, pool, dsplit)).reshape(arr.shape)
            det[f"{layer}_{pool}"] = {
                "max_abs_diff_same_settings": float(np.abs(arr - cached).max()),
                "max_abs_diff_alt_batch_size": float(np.abs(alt[(layer, pool)] - cached).max()),
                "mean_abs_value": float(np.abs(cached).mean()),
            }
        manifest["determinism"] = {"split": dsplit, "threshold": dcfg["max_abs_diff"],
                                   "alt_batch_size": dcfg["alt_batch_size"], "results": det}
        manifest["finished"] = now_iso()
        manifest["seconds_total"] = round(time.time() - t0, 1)
        manifest["complete"] = True
        write_json(man_path, manifest)
        summaries[tag] = manifest
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return summaries
