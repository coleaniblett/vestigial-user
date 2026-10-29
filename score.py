"""Per-token NLL of the content tokens in every condition, for one model.

    python score.py --model sft                     # a stage name from config.MODELS, or any HF id
    python score.py --model HuggingFaceTB/SmolLM2-135M-Instruct --tokenizer HuggingFaceTB/SmolLM2-135M-Instruct

Writes <out>/<stage>.csv with columns text_id, condition, n_tokens, mean_nll.
"""

import argparse
import csv
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

import config
from render import load_jsonl, load_tokenizer, render_all


def resolve(model_arg, stage=None):
    model_id = config.MODELS.get(model_arg, model_arg)
    if stage is None:
        stage = model_arg if model_arg in config.MODELS else model_id.split("/")[-1]
    return model_id, stage


def load_model(model_id, dtype=None):
    """dtype None: bf16 on GPU, fp32 on CPU. Scoring passes fp32: in bf16 a text's NLL shifts by
    up to ~0.05 nats with batch composition (measured on OLMo 3 7B), comparable to slot effects."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return model, device


@torch.no_grad()
def content_nll(model, device, items, pad_id, batch_size):
    """items: list of (key, Rendered). Returns {key: per-token NLL tensor over content}."""
    out = {}
    order = sorted(items, key=lambda kv: len(kv[1].input_ids))
    for i in range(0, len(order), batch_size):
        batch = order[i : i + batch_size]
        width = max(len(r.input_ids) for _, r in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for j, (_, r) in enumerate(batch):
            ids[j, : len(r.input_ids)] = torch.tensor(r.input_ids)
            mask[j, : len(r.input_ids)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        for j, (key, r) in enumerate(batch):
            # logits at position t predict token t+1
            lp = F.log_softmax(logits[j, r.start - 1 : r.end - 1].float(), dim=-1)
            target = ids[j, r.start : r.end].to(device)
            out[key] = -lp.gather(-1, target[:, None])[:, 0].cpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", help="label for the output file (default: stage name or model id)")
    ap.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
    ap.add_argument("--dataset", default=config.DATA_PATH)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=8)  # fp32 7B on 48 GB: 16 runs short on long texts
    ap.add_argument("--preceding-turn", action="store_true", default=config.PRECEDING_TURN)
    ap.add_argument("--out", default="results/scores")
    args = ap.parse_args()

    model_id, stage = resolve(args.model, args.stage)
    tok = load_tokenizer(args.tokenizer)
    rows = load_jsonl(args.dataset)[: args.limit]
    items = []
    for row in rows:
        for cond, r in render_all(tok, row, args.preceding_turn).items():
            items.append(((row["id"], cond), r))

    model, device = load_model(model_id, torch.float32)
    n_vocab = model.get_input_embeddings().num_embeddings
    assert max(max(r.input_ids) for _, r in items) < n_vocab, "tokenizer ids exceed model vocab"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    nll = content_nll(model, device, items, pad_id, args.batch_size)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{stage}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text_id", "condition", "n_tokens", "mean_nll"])
        for (text_id, cond), _ in items:
            v = nll[(text_id, cond)]
            w.writerow([text_id, cond, len(v), f"{v.mean().item():.6f}"])
    print(f"{model_id}: wrote {len(items)} rows to {path}")


if __name__ == "__main__":
    main()
