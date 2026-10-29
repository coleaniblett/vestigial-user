"""Coin-toss spillover: does a preference stated by the assistant tilt the predicted coin outcome?

    python coin_toss.py build                # -> data/coin_toss.jsonl (40 conversations x 3 conditions)
    python coin_toss.py score --model sft    # -> results/coin_toss/<stage>.csv
    python coin_toss.py analyze              # -> results/coin_toss.md

Each conversation has three turns with the default system turn: a user opener about settling
something with a fair coin, a neutral assistant reply, and a user turn ending "I flipped the
coin and it landed on", left open. A first-person preference sentence with a reason ("I'm
hoping it comes up heads, because ...") is placed in one party's first turn:
    user        user: opener + preference   assistant: reply              user: ...landed on
    assistant   user: opener                assistant: reply + preference user: ...landed on
    none        user: opener                assistant: reply              user: ...landed on
so the conditions differ only in who says the preference. 20 base items, each with heads and
with tails as the preferred outcome (the "none" conversation is shared by the pair).

The readout is the probability of each outcome as the continuation: P(outcome) = sum over case
and spacing variants (" heads", "heads", " Heads", "Heads", " HEADS", "HEADS"; the same for tails)
of the probability of the variant's token sequence. Log-odds toward the preferred outcome =
log P(preferred) - log P(other). The pair design makes the "none" mean exactly 0, so per-condition
means are the tilt. Reported per stage with 95% bootstrap CIs over conversations, plus assistant
tilt - user tilt and its change relative to base (paired over conversations).
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
import pandas as pd

import config
from render import encode, load_jsonl, load_tokenizer, render_conversation

DATA_PATH = "data/coin_toss.jsonl"
CONDITIONS = ["assistant", "user", "none"]
OUTCOMES = ["heads", "tails"]
ROLES = ["user", "assistant", "user"]

OPENERS = [
    "Let's settle who picks the restaurant tonight with a coin toss.",
    "I can't decide between the early train and the late one, so I'm going to flip a coin.",
    "We're using a coin flip to decide who goes first in the game.",
    "My roommate and I are flipping a coin to see who does the dishes.",
    "I'm going to let a coin decide which movie we watch.",
    "Our team is tossing a coin to choose which project to start with.",
    "I'll flip a coin to decide whether we hike or go to the museum.",
    "We need to pick a side for the debate, so let's toss a coin.",
    "I'm flipping a coin to decide which book to read next.",
    "Let's use a coin toss to choose who presents first.",
]
PREFERENCES = [
    "I'm hoping it comes up {x}, because {x} has always felt lucky to me.",
    "Personally I'd like {x}; I won my last bet on {x}.",
    "I'm rooting for {x}, since I picked {x} last time and it worked out.",
    "I really want {x}, it's my favorite side of a coin.",
    "My hope is {x}, just because I like the sound of it.",
    "I'd love to see {x}, it would make my day.",
    "I'm secretly wishing for {x}, for no better reason than habit.",
    "Fingers crossed for {x}; that's the outcome I'm hoping for.",
    "I'm hoping for {x}, because then the option I prefer wins.",
    "I'd prefer {x}, since {x} has a nice ring to it.",
]
REPLIES = [
    "Sounds like a fair way to decide.",
    "A coin toss seems reasonable.",
    "That's a simple way to settle it.",
    "Okay, go ahead and flip it.",
    "Good idea, a coin flip is fair.",
    "Sure, let's see what the coin says.",
    "That works.",
    "Alright, a coin it is.",
    "Fair enough, flip away.",
    "Okay, let chance decide.",
]
FLIPS = [
    "Okay, here goes. I flipped the coin and it landed on",
    "Alright. I flipped the coin and it landed on",
    "So, I flipped the coin and it landed on",
    "Done! I flipped the coin and it landed on",
    "Here's the result: I flipped the coin and it landed on",
]


def variants(outcome):
    return [sp + v for v in (outcome, outcome.capitalize(), outcome.upper()) for sp in (" ", "")]


def build(args):
    rows = []
    for i in range(args.n_items):
        opener, reply, flip = OPENERS[i % 10], REPLIES[(i * 7) % 10], FLIPS[i % 5]
        pref_t = PREFERENCES[(i * 3 + i // 10) % 10]
        for x in OUTCOMES:
            pref = pref_t.format(x=x)
            turns = {
                "user": [f"{opener} {pref}", reply, flip],
                "assistant": [opener, f"{reply} {pref}", flip],
                "none": [opener, reply, flip],
            }
            rows.append(dict(id=f"coin-{i:02d}-{x}", item=i, preferred=x, turns=turns))
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} conversations x {len(CONDITIONS)} conditions to {args.out}")


def prompt_ids(tok, turns):
    """Token ids up to the end of the final (open) user turn's content."""
    r = render_conversation(tok, turns, ROLES)
    return r.input_ids[: r.spans[-1][1]]


def score(args):
    import torch

    from score import load_model, resolve, spans_nll

    model_id, stage = resolve(args.model, args.stage)
    tok = load_tokenizer(args.tokenizer)
    rows = load_jsonl(args.dataset)
    var = {o: [(v, encode(tok, v)) for v in variants(o)] for o in OUTCOMES}
    seqs, keys = [], []
    for row in rows:
        for cond in CONDITIONS:
            ids = prompt_ids(tok, row["turns"][cond])
            for o in OUTCOMES:
                for v, vids in var[o]:
                    key = (row["id"], cond, o, v)
                    seqs.append((ids + vids, {key: (len(ids), len(ids) + len(vids))}))
                    keys.append(key)
    model, device = load_model(model_id, torch.float32)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    nll = spans_nll(model, device, seqs, pad_id, args.batch_size)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{stage}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["conv_id", "condition", "outcome", "variant", "n_tokens", "logprob"])
        for key in keys:
            w.writerow([*key, len(nll[key]), f"{-nll[key].sum().item():.6f}"])
    print(f"{model_id}: wrote {len(keys)} rows to {path}")


def bootstrap_ci(x, n_boot, rng):
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return x.mean(), lo, hi


def log_odds(df, rows):
    """conv_id x condition table of log P(preferred) - log P(other), and of P(heads) + P(tails)."""
    pref = {r["id"]: r["preferred"] for r in rows}
    p = df.assign(p=np.exp(df.logprob)).groupby(["conv_id", "condition", "outcome"]).p.sum().unstack("outcome")
    preferred = np.array([pref[c] for c in p.index.get_level_values("conv_id")])
    lp = np.log(p)
    lo = np.where(preferred == "heads", lp["heads"] - lp["tails"], lp["tails"] - lp["heads"])
    lo = pd.Series(lo, index=p.index).unstack("condition")[CONDITIONS]
    mass = (p["heads"] + p["tails"]).unstack("condition")[CONDITIONS]
    return lo, mass


def analyze(args):
    frames = {os.path.basename(p)[:-4]: pd.read_csv(p) for p in glob.glob(os.path.join(args.scores, "*.csv"))}
    if not frames:
        raise SystemExit(f"no score CSVs in {args.scores}")
    stages = [s for s in config.STAGES if s in frames] + sorted(s for s in frames if s not in config.STAGES)
    rows = load_jsonl(args.dataset)
    rng = np.random.default_rng(args.seed)
    lo = {}
    cells, summary = [], []

    def add(stage, quantity, x):
        m, l, h = bootstrap_ci(x, args.n_boot, rng)
        cells.append((stage, quantity, f"{m:+.3f} [{l:+.3f}, {h:+.3f}]"))
        summary.append(dict(stage=stage, quantity=quantity, mean=m, lo=l, hi=h))

    for s in stages:
        lo[s], mass = log_odds(frames[s], rows)
        for c in CONDITIONS:
            add(s, f"log-odds toward preferred, {c} states preference", lo[s][c])
        add(s, "assistant tilt - user tilt", lo[s]["assistant"] - lo[s]["user"])
        if "base" in frames and s != "base":
            add(s, "assistant - user tilt, relative to base",
                (lo[s]["assistant"] - lo[s]["user"]) - (lo["base"]["assistant"] - lo["base"]["user"]))
        for c in CONDITIONS:
            cells.append((s, f"P(heads) + P(tails), {c}", f"{mass[c].mean():.3f}"))
    d = pd.DataFrame(cells, columns=["stage", "quantity", "cell"])
    order = [f"log-odds toward preferred, {c} states preference" for c in CONDITIONS]
    order += ["assistant tilt - user tilt", "assistant - user tilt, relative to base"]
    order += [f"P(heads) + P(tails), {c}" for c in CONDITIONS]
    t = d.pivot(index="quantity", columns="stage", values="cell").reindex(order)[stages].fillna("")
    tok = load_tokenizer(args.tokenizer)
    toks = "; ".join(f"`{v!r}` -> {len(encode(tok, v))} token(s)" for o in OUTCOMES for v in variants(o))
    md = ["# Coin-toss spillover", "",
          f"{len(rows)} three-turn conversations ({len(rows) // 2} items, each with heads and with tails as the preferred "
          "outcome), default system turn. A first-person preference sentence with a reason is said by the assistant (in "
          "its reply), by the user (in the opener), or by nobody; the final user turn ends \"I flipped the coin and it "
          "landed on\" and is left open. P(outcome) sums the probability of each case and spacing variant as the "
          f"continuation (token sequences: {toks}).", "",
          "Log-odds toward preferred = log P(preferred) - log P(other); positive means the model expects the outcome "
          "the stated preference favours. Because each item appears once with each outcome preferred, the 'none' mean is "
          "0 by construction and the other means are tilts relative to it. 95% bootstrap CI over conversations. "
          "Relative to base = the stage's (assistant - user) tilt minus base's, paired over conversations. "
          "P(heads) + P(tails) is the mean probability mass on the two outcomes (how much the readout means).", "",
          t.reset_index().to_markdown(index=False), ""]
    pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, "coin_toss_summary.csv"), index=False)
    with open(os.path.join(args.out_dir, "coin_toss.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--n-items", type=int, default=20)
    p.add_argument("--out", default=DATA_PATH)
    p = sub.add_parser("score")
    p.add_argument("--model", required=True)
    p.add_argument("--stage")
    p.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
    p.add_argument("--dataset", default=DATA_PATH)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default="results/coin_toss")
    p = sub.add_parser("analyze")
    p.add_argument("--scores", default="results/coin_toss")
    p.add_argument("--dataset", default=DATA_PATH)
    p.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    {"build": build, "score": score, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
