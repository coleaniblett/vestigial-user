"""Sample first user turn + first assistant reply pairs and a user-turn prefill set.

    python prepare_data.py                    # WildChat-1M (gated; needs HF_TOKEN)
    python prepare_data.py --dataset oasst1   # open stand-in

Writes data/texts.jsonl (id, user, assistant, source, n_user_tokens, n_assistant_tokens)
and data/prefill.jsonl (id, text_id, prefill).
"""

import argparse
import json
import os
import re

from datasets import load_dataset

import config
from render import AlignmentError, load_tokenizer, render


def wildchat_pairs(seed):
    ds = load_dataset(config.DATASET_IDS["wildchat"], split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=20_000)
    for ex in ds:
        if ex["language"] != "English" or ex["toxic"] or ex.get("redacted"):
            continue
        conv = ex["conversation"]
        if len(conv) < 2 or conv[0]["role"] != "user" or conv[1]["role"] != "assistant":
            continue
        if any(t.get("toxic") for t in conv[:2]):
            continue
        yield ex["conversation_hash"], conv[0]["content"], conv[1]["content"]


def oasst_toxicity(msg):
    labels = msg.get("labels") or {}
    vals = dict(zip(labels.get("name") or [], labels.get("value") or []))
    if "toxicity" in vals:
        return vals["toxicity"]
    return (msg.get("detoxify") or {}).get("toxicity") or 0.0


def oasst_pairs(seed):
    import random

    ds = load_dataset(config.DATASET_IDS["oasst1"], split="train")
    ok = [m for m in ds if not m["deleted"] and m["review_result"] and m["lang"] == "en"]
    replies = {}
    for m in ok:
        if m["role"] == "assistant" and m["parent_id"] is not None:
            replies.setdefault(m["parent_id"], []).append(m)
    roots = [m for m in ok if m["role"] == "prompter" and m["parent_id"] is None]
    random.Random(seed).shuffle(roots)
    for root in roots:
        kids = replies.get(root["message_id"])
        if not kids:
            continue
        # "first" reply: best-ranked, else earliest
        reply = min(kids, key=lambda m: (m["rank"] is None, m["rank"] or 0, m["created_date"]))
        if oasst_toxicity(root) >= 0.5 or oasst_toxicity(reply) >= 0.5:
            continue
        yield root["message_id"], root["text"], reply["text"]


def clean(text):
    return text.strip()


def usable(tok, text):
    n = len(tok(text, add_special_tokens=False)["input_ids"])
    if not config.MIN_TOKENS <= n <= config.MAX_TOKENS:
        return None
    try:
        for slot in ("user", "assistant", None):
            render(tok, text, slot)
            render(tok, text, slot, preceding=True)
    except AlignmentError:
        return None
    return n


FIRST_SENTENCE = re.compile(r"^(.+?[.!?])\s+\S", re.S)


def first_sentence(text):
    m = FIRST_SENTENCE.match(text)
    if not m:
        return None
    s = m.group(1)
    return s if "\n" not in s and 5 <= len(s.split()) <= 40 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(config.DATASET_IDS), default=config.DATASET)
    ap.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
    ap.add_argument("--n", type=int, default=config.N_TEXTS)
    ap.add_argument("--n-prefill", type=int, default=config.N_PREFILL)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", default=config.DATA_PATH)
    ap.add_argument("--prefill-out", default=config.PREFILL_PATH)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    pairs = {"wildchat": wildchat_pairs, "oasst1": oasst_pairs}[args.dataset](args.seed)
    rows, seen = [], set()
    for src_id, user, asst in pairs:
        user, asst = clean(user), clean(asst)
        if src_id in seen or user in seen:
            continue
        nu, na = usable(tok, user), usable(tok, asst)
        if nu is None or na is None:
            continue
        seen.update([src_id, user])
        rows.append(
            dict(id=f"{args.dataset}-{len(rows):04d}", source_id=src_id, source=config.DATASET_IDS[args.dataset],
                 user=user, assistant=asst, n_user_tokens=nu, n_assistant_tokens=na)
        )
        if len(rows) == args.n:
            break
    if len(rows) < args.n:
        raise SystemExit(f"only found {len(rows)} usable pairs")

    prefill, seen_sentences = [], set()
    for row in rows:
        s = first_sentence(row["user"])
        if s and s not in seen_sentences:  # distinct conversations can share an opening sentence
            seen_sentences.add(s)
            prefill.append(dict(id=f"prefill-{len(prefill):02d}", text_id=row["id"], prefill=s))
        if len(prefill) == args.n_prefill:
            break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for path, items in ((args.out, rows), (args.prefill_out, prefill)):
        with open(path, "w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(items)} rows to {path}")


if __name__ == "__main__":
    main()
