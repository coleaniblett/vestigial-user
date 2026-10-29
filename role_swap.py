"""Role-swapped conversations: the same turns under the normal and the flipped role labels.

    python role_swap.py prepare                 # -> data/swap.jsonl (300 held-out WildChat conversations)
    python role_swap.py replies                 # adds OLMo 3 7B Instruct's greedy reply to each P1 (GPU)
    python role_swap.py score --model sft       # -> results/role_swap/<stage>.csv
    python role_swap.py analyze                 # -> results/role_swap.md, results/role_swap_summary.csv

Each conversation is a user turn P1, the assistant's reply R1 and the user's next turn P2, from
English WildChat-1M, skipping any conversation whose P1 or P2 (or conversation_hash) is in the
Dolci-Instruct SFT/DPO/RL data. Every turn must be MIN_TOKENS-MAX_TOKENS tokens and render cleanly.
Two labelings keep every content token identical (checked, as for the slot conditions):
    normal:   user P1, assistant R1, user P2
    flipped:  assistant P1, user R1, assistant P2
Both get the template's default system turn. Formats:
    3turn  P1, R1, P2 with ChatGPT's R1; R1 and P2 are scored
    2turn  P1, R1 only; R1 is ChatGPT's reply or OLMo 3 7B Instruct's greedy reply to P1
           (replies subcommand: max 300 new tokens, cut replies kept and flagged), R1 scored
Label effect = NLL(flipped) - NLL(normal), per turn. Base shows what the flipped labeling's
oddness (the assistant speaking first) costs on its own, so later stages are also reported
relative to base, paired over conversations. Mean next-token entropy over each scored turn's
content tokens is saved and reported alongside.
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
    """{labeling: RenderedConversation} for the first len(turns) turns; raises AlignmentError unless content
    tokens match across labelings."""
    out = {name: render_conversation(tok, turns, roles[: len(turns)]) for name, roles in LABELINGS.items()}
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


def replies(args):
    """Add OLMo 3 7B Instruct's greedy reply to each P1 as r1_olmo (None if it fails the filters)."""
    from add_replies import greedy_replies
    from score import load_model, resolve

    model_id, _ = resolve(args.model)
    tok = load_tokenizer(args.tokenizer)
    rows = load_jsonl(args.dataset)
    model, device = load_model(model_id)  # bf16 on GPU, as for add_replies.py
    out = greedy_replies(model, device, tok, [r["p1"] for r in rows], args.max_new_tokens, args.batch_size)
    stats = Counter()
    for row, (seq, truncated) in zip(rows, out):
        text = tok.decode(seq, skip_special_tokens=False).strip()
        n = len(encode(tok, text))
        ok = config.MIN_TOKENS <= n <= config.MAX_TOKENS
        if ok:
            try:
                render_both(tok, [row["p1"], text])
            except AlignmentError:
                ok = False
        stats["kept" if ok else "length/render"] += 1
        stats["kept truncated"] += int(ok and truncated)
        row.update(r1_olmo=text if ok else None, r1_olmo_truncated=truncated if ok else None, n_r1_olmo=n if ok else None,
                   r1_olmo_model=f"{model_id} greedy, max_new_tokens={args.max_new_tokens}")
    with open(args.dataset, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{model_id}: {len(rows)} prompts, {dict(stats)}; updated {args.dataset}")


def score_seqs(tok, rows):
    """[(input_ids, {(conv_id, format, r1_source, labeling, turn): span})] for every scored condition."""
    seqs = []
    for row in rows:
        for name, r in render_both(tok, [row["p1"], row["r1"], row["p2"]]).items():
            seqs.append((r.input_ids, {(row["id"], "3turn", "chatgpt", name, t): r.spans[i] for t, i in SCORED.items()}))
        for src, r1 in (("chatgpt", row["r1"]), ("olmo-instruct", row.get("r1_olmo"))):
            if r1 is None:
                continue
            for name, r in render_both(tok, [row["p1"], r1]).items():
                seqs.append((r.input_ids, {(row["id"], "2turn", src, name, "R1"): r.spans[1]}))
    return seqs


def score(args):
    import torch

    from score import load_model, resolve, spans_nll

    model_id, stage = resolve(args.model, args.stage)
    tok = load_tokenizer(args.tokenizer)
    rows = load_jsonl(args.dataset)[: args.limit]
    seqs = score_seqs(tok, rows)
    model, device = load_model(model_id, torch.float32)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    nll, ent = spans_nll(model, device, seqs, pad_id, args.batch_size, entropy=True)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{stage}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["conv_id", "format", "r1_source", "labeling", "turn", "n_tokens", "mean_nll", "mean_entropy"])
        for key, v in nll.items():
            w.writerow([*key, len(v), f"{v.mean().item():.6f}", f"{ent[key].mean().item():.6f}"])
    print(f"{model_id}: wrote {len(nll)} rows to {path}")


def bootstrap_ci(x, n_boot, rng):
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return x.mean(), lo, hi


def ci_cell(x, n_boot, rng):
    m, lo, hi = bootstrap_ci(x, n_boot, rng)
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]", (m, lo, hi)


def wide_table(rows, stages):
    """rows: [(stage, quantity, cell)] -> markdown, quantities in first-seen order, one column per stage."""
    d = pd.DataFrame(rows, columns=["stage", "quantity", "cell"])
    t = d.pivot(index="quantity", columns="stage", values="cell").reindex(list(dict.fromkeys(d.quantity)))
    return t[[s for s in stages if s in t.columns]].fillna("").reset_index().to_markdown(index=False)


def analyze(args):
    frames = {os.path.basename(p)[:-4]: pd.read_csv(p) for p in glob.glob(os.path.join(args.scores, "*.csv"))}
    if not frames:
        raise SystemExit(f"no score CSVs in {args.scores}")
    stages = [s for s in config.STAGES if s in frames] + sorted(s for s in frames if s not in config.STAGES)
    rng = np.random.default_rng(args.seed)
    summary = []

    def add(stage, quantity, x, rows):
        cell, (m, lo, hi) = ci_cell(x, args.n_boot, rng)
        rows.append((stage, quantity, cell))
        summary.append(dict(stage=stage, quantity=quantity, n=len(x), mean=m, lo=lo, hi=hi))

    def pivot(stage, fmt, values="mean_nll"):
        d = frames[stage][frames[stage]["format"] == fmt]
        return d.pivot_table(index="conv_id", columns=["r1_source", "turn", "labeling"], values=values)

    # three-turn conversations, ChatGPT R1
    w3 = {s: pivot(s, "3turn") for s in stages}
    eff3 = {s: {t: w[("chatgpt", t, "flipped")] - w[("chatgpt", t, "normal")] for t in SCORED} for s, w in w3.items()}
    rows3 = []
    for s in stages:
        for t in SCORED:
            for name in LABELINGS:
                add(s, f"NLL {t} {name}", w3[s][("chatgpt", t, name)], rows3)
        for t in SCORED:
            add(s, f"label effect {t} (flipped - normal)", eff3[s][t], rows3)
        if "base" in eff3 and s != "base":
            for t in SCORED:
                add(s, f"label effect {t} relative to base", eff3[s][t] - eff3["base"][t].reindex(eff3[s][t].index), rows3)

    counts = load_jsonl(args.dataset) if os.path.exists(args.dataset) else []
    n3 = len(w3[stages[0]])
    md = ["# Role-swapped conversations", "",
          f"{n3} held-out WildChat conversations (user P1, assistant R1, user P2), each scored under two "
          "labelings that keep every content token identical:", "",
          "- normal: user P1, assistant R1, user P2",
          "- flipped: assistant P1, user R1, assistant P2", "",
          "Both labelings get the default system turn. Mean per-token NLL (nats) over conversations, 95% bootstrap "
          "CI over conversations. Label effect = NLL(flipped) - NLL(normal): positive means the turn is easier under "
          "its real role. Relative to base = the stage's label effect minus base's, paired over conversations; base "
          "measures what the flipped labeling's oddness costs on its own.", "",
          "## Three-turn conversations (P1, R1, P2), ChatGPT's R1; R1 and P2 scored", ""]
    lengths = frames[stages[0]].query("format == '3turn' and labeling == 'normal'").groupby("turn").n_tokens.median()
    line = f"Median tokens: R1 {int(lengths['R1'])}, P2 {int(lengths['P2'])}"
    if counts and "n_p1" in counts[0]:
        line += f", P1 {int(np.median([r['n_p1'] for r in counts]))}"
    if counts:
        line += f". P2 identical to P1 in {sum(r['p2'] == r['p1'] for r in counts)} conversations"
    md += [line + ".", "", wide_table(rows3, stages), ""]
    if counts:
        # users often resend or edit their prompt; a P2 that copies P1 is easy under either labeling
        copies = {r["id"] for r in counts if r["p2"].startswith(r["p1"][:50])}
        rows = []
        for s in stages:
            for t in SCORED:
                e = eff3[s][t]
                cells = [ci_cell(x, args.n_boot, rng)[0] for x in (e[~e.index.isin(copies)], e[e.index.isin(copies)])]
                rows.append({"stage": s, "turn": t, f"excluding P2 copies (n={len(e) - len(copies)})": cells[0],
                             f"P2 copies only (n={len(copies)})": cells[1]})
        md += ["### Label effect without conversations whose P2 copies P1", "",
               "P2 starting with the first 50 characters of P1 (the user resent or edited the prompt).", "",
               pd.DataFrame(rows).to_markdown(index=False), ""]

    # two-turn conversations: ChatGPT's R1 vs OLMo Instruct's own greedy reply, same conversations
    w2 = {s: pivot(s, "2turn") for s in stages}
    both = None  # conversations with both R1 sources, once known
    if all(("olmo-instruct", "R1", "flipped") in w.columns for w in w2.values()):
        both = w2[stages[0]].dropna().index
        w2 = {s: w.reindex(both) for s, w in w2.items()}
        eff2 = {s: {src: w[(src, "R1", "flipped")] - w[(src, "R1", "normal")] for src in ("chatgpt", "olmo-instruct")}
                for s, w in w2.items()}
        rows2 = []
        for s in stages:
            for src in ("chatgpt", "olmo-instruct"):
                for name in LABELINGS:
                    add(s, f"2turn NLL R1 {src} {name}", w2[s][(src, "R1", name)], rows2)
            for src in ("chatgpt", "olmo-instruct"):
                add(s, f"2turn label effect R1 {src}", eff2[s][src], rows2)
            inter = eff2[s]["olmo-instruct"] - eff2[s]["chatgpt"]
            add(s, "2turn interaction: OLMo label effect - ChatGPT label effect", inter, rows2)
            if "base" in eff2 and s != "base":
                for src in ("chatgpt", "olmo-instruct"):
                    add(s, f"2turn label effect R1 {src} relative to base", eff2[s][src] - eff2["base"][src], rows2)
                base_inter = eff2["base"]["olmo-instruct"] - eff2["base"]["chatgpt"]
                add(s, "2turn interaction relative to base", inter - base_inter, rows2)
        trunc = sum(bool(r.get("r1_olmo_truncated")) for r in counts if r.get("r1_olmo") is not None)
        dropped = sum(r.get("r1_olmo") is None for r in counts) if counts and "r1_olmo" in counts[0] else 0
        md += ["## Two-turn conversations (P1, R1): ChatGPT's reply vs OLMo Instruct's own reply", "",
               "Normal: user P1, assistant R1. Flipped: assistant P1, user R1. R1 is either ChatGPT's original reply or "
               "OLMo 3 7B Instruct's greedy reply to P1 (max 300 new tokens; cut replies kept). Only conversations with "
               f"both replies are used (n={len(both)}; {dropped} OLMo replies failed the length/render filters; "
               f"{trunc} of the kept OLMo replies are cut at 300 tokens). Interaction = OLMo label effect - ChatGPT "
               "label effect, paired over conversations.", "",
               "**Circularity at the Instruct stage:** the OLMo replies were generated greedily by the Instruct checkpoint "
               "itself in the normal two-turn context, so at that stage their normal-label NLL is low by construction "
               "and the Instruct-stage OLMo label effect and interaction are partly mechanical. SFT and DPO score the "
               "same text without having generated it.", "",
               wide_table(rows2, stages), ""]

    # entropy of the next-token distribution over the scored content tokens
    rows_e = []
    for s in stages:
        d = frames[s]
        groups = [("3turn", "chatgpt", t) for t in SCORED] + [("2turn", src, "R1") for src in ("chatgpt", "olmo-instruct")]
        for fmt, src, turn in groups:
            for name in LABELINGS:
                g = d[(d["format"] == fmt) & (d.r1_source == src) & (d.turn == turn) & (d.labeling == name)]
                if fmt == "2turn" and both is not None:
                    g = g[g.conv_id.isin(both)]
                if g.empty:
                    continue
                label = f"3turn {turn}" if fmt == "3turn" else f"2turn R1 = {'ChatGPT' if src == 'chatgpt' else 'OLMo'} reply"
                add(s, f"entropy {label}, {name}", g.mean_entropy, rows_e)
    md += ["## Mean next-token entropy over the scored content tokens (nats)", "",
           "Entropy of the model's next-token distribution at each content token of the scored turn, averaged per "
           "conversation; 95% bootstrap CI over conversations.", "", wide_table(rows_e, stages), ""]

    pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, "role_swap_summary.csv"), index=False)
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
    p = sub.add_parser("replies")
    p.add_argument("--model", default=config.REPLY_MODEL)
    p.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
    p.add_argument("--dataset", default=DATA_PATH)
    p.add_argument("--max-new-tokens", type=int, default=config.REPLY_MAX_NEW_TOKENS)
    p.add_argument("--batch-size", type=int, default=32)
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
    {"prepare": prepare, "replies": replies, "score": score, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
