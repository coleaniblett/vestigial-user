"""Add a model-written assistant source: a greedy reply to each user text of one source.

    python add_replies.py                     # OLMo 3 7B Instruct replies to the WildChat user turns

Each user text is rendered as the only user turn with the Instruct chat template (default system
turn included) and the reply is decoded greedily for up to --max-new-tokens tokens. Replies that
do not end their turn within that budget are kept and marked truncated=true: under causal
attention a truncated reply's NLL is that of the first tokens of the full reply. Replies go
through the same filters as sampled texts (MIN_TOKENS-MAX_TOKENS tokens, clean rendering). Rows are appended to
data/texts.jsonl as source "olmo-instruct", kind "assistant", with the prompt's pair_id; existing
rows of that source are replaced.
"""

import argparse
import json
from collections import Counter

import torch

import config
from prepare_data import clean, usable
from render import encode, load_jsonl, load_tokenizer
from score import load_model, resolve


def stop_ids(tok):
    vocab = tok.get_vocab()
    return [vocab[t] for t in ("<|im_end|>", "<|endoftext|>") if t in vocab] or [tok.eos_token_id]


@torch.no_grad()
def greedy_replies(model, device, tok, prompts, max_new_tokens, batch_size):
    """[(token ids of the reply up to its stop token or the budget, truncated?)]."""
    stops = stop_ids(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    encoded = [encode(tok, tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                   add_generation_prompt=True)) for p in prompts]
    out = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(encoded[i]))
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        width = max(len(encoded[i]) for i in idx)
        ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, i in enumerate(idx):  # left padding for generation
            ids[j, width - len(encoded[i]) :] = torch.tensor(encoded[i])
            mask[j, width - len(encoded[i]) :] = 1
        gen = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), do_sample=False,
                             temperature=None, top_p=None, top_k=None, max_new_tokens=max_new_tokens,
                             eos_token_id=stops, pad_token_id=pad_id)
        for j, i in enumerate(idx):
            seq = gen[j, width:].tolist()
            cut = next((k for k, t in enumerate(seq) if t in stops), None)
            out[i] = (seq[:cut], False) if cut is not None else (seq, True)
        print(f"  {min(b + batch_size, len(order))}/{len(order)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.REPLY_MODEL)
    ap.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
    ap.add_argument("--dataset", default=config.DATA_PATH)
    ap.add_argument("--prompt-source", default=config.PAIR_SOURCE)
    ap.add_argument("--source", default=config.REPLY_SOURCE)
    ap.add_argument("--max-new-tokens", type=int, default=config.REPLY_MAX_NEW_TOKENS)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    model_id, _ = resolve(args.model)
    tok = load_tokenizer(args.tokenizer)
    rows = [r for r in load_jsonl(args.dataset) if r["source"] != args.source]
    prompts = [r for r in rows if r["source"] == args.prompt_source and r["kind"] == "user"]
    model, device = load_model(model_id)  # bf16 on GPU, as for generate.py
    replies = greedy_replies(model, device, tok, [r["text"] for r in prompts], args.max_new_tokens, args.batch_size)

    stats, added = Counter(), []
    for p, (seq, truncated) in zip(prompts, replies):
        text = clean(tok.decode(seq, skip_special_tokens=False))
        n = usable(tok, text) if text else None
        if n is None:
            stats["length/render"] += 1
            continue
        added.append(dict(id=f"{args.source}-assistant-{p['pair_id'][-4:]}", source=args.source, kind="assistant",
                          text=text, n_tokens=n, truncated=truncated, pair_id=p["pair_id"], source_id=p["id"],
                          dataset=f"{model_id} greedy, max_new_tokens={args.max_new_tokens}"))
    stats["kept"] = len(added)
    stats["kept truncated"] = sum(r["truncated"] for r in added)
    with open(args.dataset, "w", encoding="utf-8") as f:
        for r in rows + added:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{model_id}: {len(prompts)} prompts, {dict(stats)}; wrote {len(rows) + len(added)} rows to {args.dataset}")


if __name__ == "__main__":
    main()
