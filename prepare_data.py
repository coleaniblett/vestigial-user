"""Sample held-out user + assistant texts and a user-turn prefill set.

    python prepare_data.py                                  # WildChat pairs + oasst1 user prompts
    python prepare_data.py --pairs oasst1 --extra-user none --no-exclude-dolci   # smoke test

Pair source (default WildChat-1M): first user turn + first assistant reply of each conversation.
Extra user source (default oasst1): root prompts only. Conversations whose user turn appears in
the Dolci-Instruct SFT, DPO or RL data are skipped before the other filters (see
check_contamination.py), and sampling continues until --n clean texts per source survive.

Writes data/texts.jsonl, one text per row (id, source, kind, text, n_tokens, pair_id,
source_id, dataset), and data/prefill.jsonl (id, text_id, prefill) from the pair source's user
turns. OLMo replies are added afterwards by add_replies.py.
"""

import argparse
import json
import os
import random
import re
from collections import Counter

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


def oasst_messages(seed):
    ds = load_dataset(config.DATASET_IDS["oasst1"], split="train")
    ok = [m for m in ds if not m["deleted"] and m["review_result"] and m["lang"] == "en"]
    roots = [m for m in ok if m["role"] == "prompter" and m["parent_id"] is None]
    random.Random(seed).shuffle(roots)
    return ok, roots


def oasst_pairs(seed):
    ok, roots = oasst_messages(seed)
    replies = {}
    for m in ok:
        if m["role"] == "assistant" and m["parent_id"] is not None:
            replies.setdefault(m["parent_id"], []).append(m)
    for root in roots:
        kids = replies.get(root["message_id"])
        if not kids:
            continue
        # "first" reply: best-ranked, else earliest
        reply = min(kids, key=lambda m: (m["rank"] is None, m["rank"] or 0, m["created_date"]))
        if oasst_toxicity(root) >= 0.5 or oasst_toxicity(reply) >= 0.5:
            continue
        yield root["message_id"], root["text"], reply["text"]


def oasst_prompts(seed):
    """Root prompts only (no reply required), same language/review/toxicity filters."""
    _, roots = oasst_messages(seed)
    for root in roots:
        if oasst_toxicity(root) < 0.5:
            yield root["message_id"], root["text"], None


SOURCES = {"wildchat": wildchat_pairs, "oasst1": oasst_pairs}
USER_ONLY = {"oasst1": oasst_prompts}


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


def sample(source, stream, tok, n, turns, kinds):
    """First n candidates from stream whose texts pass every filter. kinds: ("user",) or ("user", "assistant")."""
    from check_contamination import is_contaminated

    rows, seen, stats = [], set(), Counter()
    for src_id, user, asst in stream:
        stats["candidates"] += 1
        user = clean(user)
        if src_id in seen or user in seen:
            stats["duplicate"] += 1
            continue
        if turns is not None and is_contaminated(turns, user, "user", src_id):
            stats["in Dolci"] += 1
            continue
        texts = {"user": user, "assistant": clean(asst) if asst is not None else None}
        lengths = {k: usable(tok, texts[k]) for k in kinds}
        if any(v is None for v in lengths.values()):
            stats["length/render"] += 1
            continue
        seen.update([src_id, user])
        pair_id = f"{source}-{len(rows) // len(kinds):04d}"
        for k in kinds:
            rows.append(dict(id=f"{source}-{k}-{pair_id[-4:]}", source=source, kind=k, text=texts[k],
                             n_tokens=lengths[k], pair_id=pair_id, source_id=src_id,
                             dataset=config.DATASET_IDS[source]))
        if len(rows) == n * len(kinds):
            break
    stats["kept"] = len(rows) // len(kinds)
    print(f"{source} ({'+'.join(kinds)}): {dict(stats)}", flush=True)
    if len(rows) < n * len(kinds):
        raise SystemExit(f"only found {len(rows) // len(kinds)} usable {source} texts")
    return rows


FIRST_SENTENCE = re.compile(r"^(.+?[.!?])\s+\S", re.S)


def first_sentence(text):
    m = FIRST_SENTENCE.match(text)
    if not m:
        return None
    s = m.group(1)
    return s if "\n" not in s and 5 <= len(s.split()) <= 40 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", choices=list(SOURCES), default=config.PAIR_SOURCE)
    ap.add_argument("--extra-user", choices=[*USER_ONLY, "none"], default=config.EXTRA_USER_SOURCE)
    ap.add_argument("--exclude-dolci", action=argparse.BooleanOptionalAction, default=config.EXCLUDE_DOLCI)
    ap.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
    ap.add_argument("--n", type=int, default=config.N_TEXTS)
    ap.add_argument("--n-prefill", type=int, default=config.N_PREFILL)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", default=config.DATA_PATH)
    ap.add_argument("--prefill-out", default=config.PREFILL_PATH)
    args = ap.parse_args()
    if args.extra_user == args.pairs:
        ap.error("--extra-user must differ from --pairs")

    tok = load_tokenizer(args.tokenizer)
    turns = None
    if args.exclude_dolci:
        from check_contamination import dolci_turns

        turns = dolci_turns()
    rows = sample(args.pairs, SOURCES[args.pairs](args.seed), tok, args.n, turns, ("user", "assistant"))
    if args.extra_user != "none":
        rows += sample(args.extra_user, USER_ONLY[args.extra_user](args.seed), tok, args.n, turns, ("user",))

    prefill, seen_sentences = [], set()
    for row in rows:
        if row["source"] != args.pairs or row["kind"] != "user":
            continue
        s = first_sentence(row["text"])
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
