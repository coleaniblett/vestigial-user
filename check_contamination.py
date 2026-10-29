"""Match texts against OLMo 3 Instruct post-training data (Dolci SFT, DPO and RL sets).

    python check_contamination.py      # overlap of data/texts.jsonl, per source and kind

Matching is exact after whitespace normalization. SFT and DPO store chat messages; the RL
set stores each prompt as one "user: ...\\nassistant: ..." string, so its turns are parsed
out, and its WildChat-derived prompts also carry WildChat's conversation_hash, which is
matched as well. prepare_data.py uses dolci_turns() to exclude contaminated conversations.
"""

import argparse
import collections
import re

from datasets import load_dataset

import config
from render import load_jsonl

DOLCI = {
    "sft": "allenai/Dolci-Instruct-SFT",
    "dpo": "allenai/Dolci-Instruct-DPO",
    "rl": "allenai/Dolci-Instruct-RL",
}
RL_TURN = re.compile(r"(?:^|\n)(user|assistant): ")


def norm(s):
    return " ".join((s or "").split())


def rl_turns(prompt):
    """[(role, text)] from an RL prompt string such as 'user: hi\\nassistant: hello\\nuser: ...'."""
    parts = RL_TURN.split(prompt)
    return list(zip(parts[1::2], parts[2::2]))


def dolci_turns(names=tuple(DOLCI)):
    """{name: {"user": set, "assistant": set, "hash": set}} of normalized turn texts per Dolci set."""
    out = {}
    for name in names:
        ds = load_dataset(DOLCI[name], split="train")
        seen = {"user": set(), "assistant": set(), "hash": set()}
        if name == "rl":
            for batch in ds.select_columns(["prompt", "conversation_hash"]).iter(batch_size=10_000):
                for prompt, h in zip(batch["prompt"], batch["conversation_hash"]):
                    for role, text in rl_turns(prompt or ""):
                        seen[role].add(norm(text))
                    if h:
                        seen["hash"].add(h)
        else:
            cols = ["messages"] if name == "sft" else ["chosen", "rejected"]
            for batch in ds.select_columns(cols).iter(batch_size=10_000):
                for col in cols:
                    for msgs in batch[col]:
                        for m in msgs or []:
                            if m["role"] in ("user", "assistant"):
                                seen[m["role"]].add(norm(m["content"]))
        print(f"{DOLCI[name]}: {len(seen['user'])} user turns, {len(seen['assistant'])} assistant turns, "
              f"{len(seen['hash'])} conversation hashes", flush=True)
        out[name] = seen
    return out


def is_contaminated(turns, text, role="user", conv_hash=None):
    t = norm(text)
    return any(t in s[role] or (conv_hash and conv_hash in s["hash"]) for s in turns.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=config.DATA_PATH)
    args = ap.parse_args()

    turns = dolci_turns()
    rows = load_jsonl(args.dataset)
    hits = collections.defaultdict(collections.Counter)
    totals = collections.Counter()
    for r in rows:
        key = (r["source"], r["kind"])
        totals[key] += 1
        for name, s in turns.items():
            if norm(r["text"]) in s[r["kind"]] or (r["kind"] == "user" and r.get("source_id") in s["hash"]):
                hits[key][name] += 1
    print(f"\nverbatim matches in {args.dataset} (text matched against Dolci turns of the same role):")
    for key, n in totals.items():
        print(f"  {key[0]} {key[1]} (n={n}): " + ", ".join(f"{name} {hits[key][name]}" for name in turns))


if __name__ == "__main__":
    main()
