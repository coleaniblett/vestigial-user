"""Slot effects with bootstrap CIs, one figure and a results table.

    python analyze.py                    # reads results/scores/*.csv (+ results/generations/*_counts.csv)

Slot effect = NLL(text in user slot) - NLL(same text in assistant slot), per text type:
    user text:      user@user - user@asst
    assistant text: asst@user - asst@asst
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402

SLOT_EFFECTS = {"user text": ("user@user", "user@asst"), "assistant text": ("asst@user", "asst@asst")}


def load_scores(score_dir):
    frames = {os.path.basename(p)[:-4]: pd.read_csv(p) for p in glob.glob(os.path.join(score_dir, "*.csv"))}
    if not frames:
        raise SystemExit(f"no score CSVs in {score_dir}")
    stages = [s for s in config.STAGES if s in frames] + sorted(s for s in frames if s not in config.STAGES)
    return {s: frames[s].pivot(index="text_id", columns="condition", values="mean_nll") for s in stages}


def bootstrap_ci(x, n_boot, rng):
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return x.mean(), lo, hi


def summarize(tables, n_boot, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for stage, t in tables.items():
        for cond in t.columns:
            rows.append(dict(stage=stage, quantity=f"NLL {cond}", **dict(zip(["mean", "lo", "hi"], bootstrap_ci(t[cond], n_boot, rng)))))
        for name, (u, a) in SLOT_EFFECTS.items():
            if u in t and a in t:
                rows.append(dict(stage=stage, quantity=f"slot effect, {name}",
                                 **dict(zip(["mean", "lo", "hi"], bootstrap_ci(t[u] - t[a], n_boot, rng)))))
    return pd.DataFrame(rows)


def plot(summary, path):
    stages = list(dict.fromkeys(summary["stage"]))
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond, _, slot in config.CONDITIONS:
        d = summary[summary.quantity == f"NLL {cond}"].set_index("stage").reindex(stages)
        if d["mean"].isna().all():
            continue
        ax.errorbar(x, d["mean"], yerr=[d["mean"] - d["lo"], d["hi"] - d["mean"]], capsize=3, marker="o",
                    linestyle=":" if slot is None else "-", label=cond)
    ax.set_xticks(x, stages)
    ax.set_xlabel("post-training stage")
    ax.set_ylabel("mean per-token NLL of content (nats)")
    ax.set_title("Content NLL by role slot (text@slot), 95% bootstrap CI")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def fmt(r):
    return f"{r['mean']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"


def results_table(summary, gen_dir):
    wide = summary.assign(cell=summary.apply(fmt, axis=1)).pivot(index="quantity", columns="stage", values="cell")
    wide = wide[list(dict.fromkeys(summary["stage"]))]
    order = [f"slot effect, {k}" for k in SLOT_EFFECTS] + [f"NLL {c}" for c, _, _ in config.CONDITIONS]
    wide = wide.reindex([q for q in order if q in wide.index])
    md = ["# Results", "", "Mean over texts, 95% bootstrap CI over texts. Slot effect = NLL(user slot) - NLL(assistant slot).",
          "", wide.to_markdown()]
    counts = sorted(glob.glob(os.path.join(gen_dir, "*_counts.csv")))
    if counts:
        g = pd.concat(pd.read_csv(p) for p in counts).set_index("stage")
        g = g.reindex([s for s in config.STAGES if s in g.index] + [s for s in g.index if s not in config.STAGES])
        md += ["", "## Prefilled user-turn continuations (counts of samples)", "", g.to_markdown()]
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/scores")
    ap.add_argument("--generations", default="results/generations")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    summary = summarize(load_scores(args.scores), args.n_boot, args.seed)
    os.makedirs(args.out, exist_ok=True)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    plot(summary, os.path.join(args.out, "nll_by_stage.png"))
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(results_table(summary, args.generations))
    print(open(os.path.join(args.out, "results.md")).read())


if __name__ == "__main__":
    main()
