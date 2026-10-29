"""Role-swapped conversations: the same three turns under the normal and the flipped role labels.

    python role_swap.py prepare                 # -> data/swap.jsonl (300 held-out WildChat conversations)
    python role_swap.py score --model sft       # -> results/role_swap/<stage>.csv
    python role_swap.py analyze                 # -> results/role_swap.md, results/role_swap_summary.csv

Each conversation is a user turn P1, the assistant's reply R1 and the user's next turn P2, from
English WildChat-1M, skipping any conversation whose P1 or P2 (or conversation_hash) is in the
Dolci-Instruct SFT/DPO/RL data. Every turn must be MIN_TOKENS-MAX_TOKENS tokens and render cleanly.
Two labelings keep every content token identical (checked, as for the slot conditions):
    normal:   user P1, assistant R1, user P2
    flipped:  assistant P1, user R1, assistant P2
Both get the template's default system turn. Only the R1 and P2 content tokens are scored.
Label effect = NLL(flipped) - NLL(normal), per turn. Base shows what the flipped labeling's
oddness (the assistant speaking first) costs on its own, so later stages are also reported
relative to base, paired over conversations.
"""

import argparse
import csv
import glob
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

import config
from render import AlignmentError, encode, load_jsonl, load_tokenizer, render_conversation

LABELINGS = {"normal": ["user", "assistant", "user"], "flipped": ["assistant", "user", "assistant"]}
SCORED = {"R1": 1, "P2": 2}  # turn name -> index in the conversation
DATA_PATH = "data/swap.jsonl"


def wildchat_triples(seed):
    from datasets import load_dataset

    ds = load_dataset(config.DATASET_IDS["wildchat"], split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=20_000)
    for ex in ds:
        if ex["language"] != "English" or ex["toxic"] or ex.get("redacted"):
            continue
        conv = ex["conversation"]
        if len(conv) < 3 or [t["role"] for t in conv[:3]] != ["user", "assistant", "user"]:
            continue
        if any(t.get("toxic") for t in conv[:3]):
            continue
        yield ex["conversation_hash"], [t["content"].strip() for t in conv[:3]]


def render_both(tok, turns):
    """{labeling: RenderedConversation}; raises AlignmentError unless content tokens match across labelings."""
    out = {name: render_conversation(tok, turns, roles) for name, roles in LABELINGS.items()}
    spans = [[tuple(r.input_ids[a:b]) for a, b in r.spans] for r in out.values()]
    if any(s != spans[0] for s in spans):
        raise AlignmentError("content tokens differ across labelings")
    return out


def prepare(args):
    from check_contamination import dolci_turns, is_contaminated

    tok = load_tokenizer(args.tokenizer)
    turns = dolci_turns()
    rows, seen, stats = [], set(), Counter()
    for conv_hash, (p1, r1, p2) in wildchat_triples(args.seed):
        stats["candidates"] += 1
        if conv_hash in seen or p1 in seen:
            stats["duplicate"] += 1
            continue
        if is_contaminated(turns, p1, "user", conv_hash) or is_contaminated(turns, p2, "user"):
            stats["in Dolci"] += 1
            continue
        n = [len(encode(tok, t)) for t in (p1, r1, p2)]
        if not all(config.MIN_TOKENS <= k <= config.MAX_TOKENS for k in n):
            stats["length"] += 1
            continue
        try:
            render_both(tok, [p1, r1, p2])
        except AlignmentError:
            stats["render"] += 1
            continue
        seen.update([conv_hash, p1])
        rows.append(dict(id=f"swap-{len(rows):04d}", source_id=conv_hash, source="wildchat", p1=p1, r1=r1, p2=p2,
                         n_p1=n[0], n_r1=n[1], n_p2=n[2]))
        if len(rows) == args.n:
            break
    stats["kept"] = len(rows)
    stats["P2 == P1"] = sum(r["p2"] == r["p1"] for r in rows)
    print(f"wildchat triples: {dict(stats)}", flush=True)
    if len(rows) < args.n:
        raise SystemExit(f"only found {len(rows)} usable conversations")
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")


def score(args):
    import torch

    from score import load_model, resolve, spans_nll

    model_id, stage = resolve(args.model, args.stage)
    tok = load_tokenizer(args.tokenizer)
    rows = load_jsonl(args.dataset)[: args.limit]
    seqs = []
    for row in rows:
        for name, r in render_both(tok, [row["p1"], row["r1"], row["p2"]]).items():
            seqs.append((r.input_ids, {(row["id"], name, turn): r.spans[i] for turn, i in SCORED.items()}))
    model, device = load_model(model_id, torch.float32)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    nll = spans_nll(model, device, seqs, pad_id, args.batch_size)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{stage}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["conv_id", "labeling", "turn", "n_tokens", "mean_nll"])
        for (conv_id, name, turn), v in nll.items():
            w.writerow([conv_id, name, turn, len(v), f"{v.mean().item():.6f}"])
    print(f"{model_id}: wrote {len(nll)} rows to {path}")


def bootstrap_ci(x, n_boot, rng):
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return x.mean(), lo, hi


def analyze(args):
    frames = {os.path.basename(p)[:-4]: pd.read_csv(p) for p in glob.glob(os.path.join(args.scores, "*.csv"))}
    if not frames:
        raise SystemExit(f"no score CSVs in {args.scores}")
    stages = [s for s in config.STAGES if s in frames] + sorted(s for s in frames if s not in config.STAGES)
    wide = {s: frames[s].pivot_table(index="conv_id", columns=["turn", "labeling"], values="mean_nll") for s in stages}
    effect = {s: {t: w[(t, "flipped")] - w[(t, "normal")] for t in SCORED} for s, w in wide.items()}
    rng = np.random.default_rng(args.seed)
    rows = []
    for s in stages:
        for t in SCORED:
            for name in LABELINGS:
                rows.append((s, f"NLL {t} {name}", *bootstrap_ci(wide[s][(t, name)], args.n_boot, rng)))
        for t in SCORED:
            rows.append((s, f"label effect {t} (flipped - normal)", *bootstrap_ci(effect[s][t], args.n_boot, rng)))
        if "base" in effect and s != "base":
            for t in SCORED:
                d = effect[s][t] - effect["base"][t].reindex(effect[s][t].index)
                rows.append((s, f"label effect {t} relative to base", *bootstrap_ci(d, args.n_boot, rng)))
    summary = pd.DataFrame(rows, columns=["stage", "quantity", "mean", "lo", "hi"])
    summary.to_csv(os.path.join(args.out_dir, "role_swap_summary.csv"), index=False)

    summary["cell"] = [f"{m:.3f} [{lo:.3f}, {hi:.3f}]" for m, lo, hi in zip(summary["mean"], summary["lo"], summary["hi"])]
    table = summary.pivot(index="quantity", columns="stage", values="cell")
    order = list(dict.fromkeys(summary["quantity"]))
    table = table.reindex(order)[stages].fillna("")
    n = {s: len(w) for s, w in wide.items()}
    counts = load_jsonl(args.dataset) if os.path.exists(args.dataset) else []
    md = ["# Role-swapped conversations", "",
          f"{n[stages[0]]} held-out WildChat conversations (user P1, assistant R1, user P2), each scored under two "
          "labelings that keep every content token identical:", "",
          "- normal: user P1, assistant R1, user P2",
          "- flipped: assistant P1, user R1, assistant P2", "",
          "Both labelings get the default system turn. Only R1 and P2 are scored. Mean per-token NLL (nats) over "
          "conversations, 95% bootstrap CI over conversations. Label effect = NLL(flipped) - NLL(normal): positive "
          "means the turn is easier under its real role. Relative to base = the stage's label effect minus base's, "
          "paired over conversations; base measures what the flipped labeling's oddness costs on its own.", ""]
    lengths = frames[stages[0]].query("labeling == 'normal'").groupby("turn").n_tokens.median()
    line = f"Median tokens: R1 {int(lengths['R1'])}, P2 {int(lengths['P2'])}"
    if counts and "n_p1" in counts[0]:
        line += f", P1 {int(np.median([r['n_p1'] for r in counts]))}"
    if counts:
        line += f". P2 identical to P1 in {sum(r['p2'] == r['p1'] for r in counts)} conversations"
    md += [line + ".", ""]
    md += [table.reset_index().to_markdown(index=False), ""]
    if counts:
        # users often resend or edit their prompt; a P2 that copies P1 is easy under either labeling
        copies = {r["id"] for r in counts if r["p2"].startswith(r["p1"][:50])}
        rows = []
        for s in stages:
            for t in SCORED:
                e = effect[s][t]
                cells = [bootstrap_ci(x, args.n_boot, rng) for x in (e[~e.index.isin(copies)], e[e.index.isin(copies)])]
                rows.append({"stage": s, "turn": t, **{name: f"{m:.3f} [{lo:.3f}, {hi:.3f}]" for name, (m, lo, hi) in
                             zip([f"excluding P2 copies (n={len(e) - len(copies)})", f"P2 copies only (n={len(copies)})"], cells)}})
        md += ["## Label effect without conversations whose P2 copies P1", "",
               "P2 starting with the first 50 characters of P1 (the user resent or edited the prompt).", "",
               pd.DataFrame(rows).to_markdown(index=False), ""]
    path = os.path.join(args.out_dir, "role_swap.md")
    with open(path, "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
    p.add_argument("--n", type=int, default=config.N_TEXTS)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--out", default=DATA_PATH)
    p = sub.add_parser("score")
    p.add_argument("--model", required=True)
    p.add_argument("--stage")
    p.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
    p.add_argument("--dataset", default=DATA_PATH)
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=4)  # three turns: sequences up to ~1k tokens, fp32
    p.add_argument("--out", default="results/role_swap")
    p = sub.add_parser("analyze")
    p.add_argument("--scores", default="results/role_swap")
    p.add_argument("--dataset", default=DATA_PATH)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    {"prepare": prepare, "score": score, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
