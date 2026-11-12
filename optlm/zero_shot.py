"""Zero-shot LM baseline (protocol E1).

For each option: context = "Question: {q}\nAnswer:", continuation = " {option}".
Context and continuation are tokenized separately and concatenated, and the log-probabilities of the
continuation tokens are summed. Three scores per option:
  sum        unnormalized total log-probability
  per_token  sum / number of continuation tokens        (primary)
  per_char   sum / number of characters of the option   (tokenizer-independent, secondary)
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F


def load_causal_lm(hf_id: str, revision: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_id, revision=revision)
    model.to(device=device, dtype=dtype).eval()
    return tok, model


@torch.no_grad()
def zero_shot_scores(model, tok, items: list[dict], batch_size: int, device: str) -> dict:
    reqs = []  # (item index, option index, ctx ids, cont ids, n chars)
    for i, it in enumerate(items):
        ctx = tok(f"Question: {it['question']}\nAnswer:")["input_ids"]
        for j, opt in enumerate(it["options"]):
            cont = tok(" " + opt)["input_ids"]
            reqs.append((i, j, ctx, cont, max(len(opt), 1)))
    n = len(items)
    total = np.zeros((n, 4))
    ntok = np.zeros((n, 4))
    nchar = np.zeros((n, 4))
    t0 = time.time()
    order = sorted(range(len(reqs)), key=lambda k: -(len(reqs[k][2]) + len(reqs[k][3])))
    for s in range(0, len(order), batch_size):
        batch = [reqs[k] for k in order[s : s + batch_size]]
        seqs = [c + t for _, _, c, t, _ in batch]
        L = max(len(x) for x in seqs)
        ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for r, x in enumerate(seqs):
            ids[r, : len(x)] = torch.tensor(x)
            mask[r, : len(x)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        logp = F.log_softmax(logits.float(), dim=-1)
        for r, (i, j, c, t, nc) in enumerate(batch):
            pos = torch.arange(len(c) - 1, len(c) + len(t) - 1, device=device)  # positions predicting cont
            tgt = torch.tensor(t, device=device)
            total[i, j] = float(logp[r, pos, tgt].sum())
            ntok[i, j] = len(t)
            nchar[i, j] = nc
    return {
        "sum": total,
        "per_token": total / ntok,
        "per_char": total / nchar,
        "seconds": time.time() - t0,
        "mean_option_tokens": float(ntok.mean()),
    }
