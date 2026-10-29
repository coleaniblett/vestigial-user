"""Does excluding Dolci-contaminated conversations select different texts?

    python selection_check.py

Compares the held-out WildChat sample (data/texts.jsonl, results/scores/base.csv) with the
contaminated texts of the earlier unfiltered sample (results_v0/: same seed and stream, so its
clean texts reappear in the held-out sample and the rest are the ones the filter removed).
Reports means, Cohen's d, Welch's t with a permutation p-value, and a Mann-Whitney p-value
(normal approximation) for token lengths and base-model plain-prose NLL.
"""

import argparse
import json
import math

import numpy as np
import pandas as pd

from check_contamination import norm
from render import load_jsonl


def mann_whitney_p(a, b):
    x = np.concatenate([a, b])
    ranks = pd.Series(x).rank().to_numpy()
    n1, n2 = len(a), len(b)
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    _, counts = np.unique(x, return_counts=True)
    n = n1 + n2
    var = n1 * n2 / 12 * ((n + 1) - (counts**3 - counts).sum() / (n * (n - 1)))
    z = (u - n1 * n2 / 2) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2))


def welch_t(a, b):
    return (a.mean() - b.mean()) / math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


def permutation_p(a, b, n_perm, rng):
    t0 = abs(welch_t(a, b))
    x = np.concatenate([a, b])
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(x)
        hits += abs(welch_t(x[: len(a)], x[len(a) :])) >= t0
    return (hits + 1) / (n_perm + 1)


def compare(name, a, b, n_perm, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sd = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return dict(quantity=name, clean_mean=a.mean(), clean_median=np.median(a), contaminated_mean=b.mean(),
                contaminated_median=np.median(b), diff=a.mean() - b.mean(), cohens_d=(a.mean() - b.mean()) / sd,
                welch_t=welch_t(a, b), perm_p=permutation_p(a, b, n_perm, rng), mann_whitney_p=mann_whitney_p(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default="data/texts.jsonl")
    ap.add_argument("--scores", default="results/scores/base.csv")
    ap.add_argument("--v0-texts", default="results_v0/data/texts.jsonl")
    ap.add_argument("--v0-scores", default="results_v0/scores/base.csv")
    ap.add_argument("--n-perm", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/selection_check.md")
    args = ap.parse_args()

    rows = [r for r in load_jsonl(args.texts) if r["source"] == "wildchat"]
    clean = {k: [r for r in rows if r["kind"] == k] for k in ("user", "assistant")}
    clean_users = {norm(r["text"]) for r in clean["user"]}
    v0 = load_jsonl(args.v0_texts)
    contam = [r for r in v0 if norm(r["user"]) not in clean_users]
    reappear = len(v0) - len(contam)

    s = pd.read_csv(args.scores).pivot(index="text_id", columns="condition", values="mean_nll")
    s0 = pd.read_csv(args.v0_scores).pivot(index="text_id", columns="condition", values="mean_nll")
    cid = [r["id"] for r in contam]
    rng = np.random.default_rng(args.seed)
    res = pd.DataFrame([
        compare("user tokens", [r["n_tokens"] for r in clean["user"]], [r["n_user_tokens"] for r in contam], args.n_perm, rng),
        compare("assistant tokens", [r["n_tokens"] for r in clean["assistant"]], [r["n_assistant_tokens"] for r in contam],
                args.n_perm, rng),
        compare("base user@plain NLL", s.loc[[r["id"] for r in clean["user"]], "user@plain"], s0.loc[cid, "user@plain"],
                args.n_perm, rng),
        compare("base asst@plain NLL", s.loc[[r["id"] for r in clean["assistant"]], "asst@plain"], s0.loc[cid, "asst@plain"],
                args.n_perm, rng),
    ])
    md = ["# Selection check: held-out vs contaminated WildChat texts", "",
          f"Held-out sample: {len(clean['user'])} pairs. Contaminated: the {len(contam)} of {len(v0)} texts of the earlier "
          f"unfiltered sample (same seed) whose user turn is in Dolci; the other {reappear} reappear in the held-out sample.",
          f"Two-sided tests; permutation p from {args.n_perm} permutations of Welch's t.", "",
          res.to_markdown(index=False, floatfmt=".4g"), ""]
    with open(args.out, "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    json.dump(cid, open(args.out.replace(".md", "_contaminated_ids.json"), "w"))


if __name__ == "__main__":
    main()
