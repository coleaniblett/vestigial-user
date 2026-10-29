"""Raw NLL, slot effects and role-specific quantities per text source, with bootstrap CIs.

    python analyze.py                    # reads results/scores/*.csv (+ results/generations/*_counts.csv)

A text set is one (source, kind), e.g. wildchat user or olmo-instruct assistant. Sets are never
pooled. Every text is scored in every slot; per set and stage we report
    raw NLL        mean content NLL in the user, assistant and plain-prose slot
    slot effect    NLL(user slot) - NLL(assistant slot), paired over texts
    role-specific  NLL(own slot) - NLL(plain): user@user - user@plain for user text,
                   asst@asst - asst@plain for assistant text
95% bootstrap CIs are over texts. Writes results/summary.csv, results/results.md and one
figure per text set, results/nll_<source>_<kind>.png (stage on x, one line per slot).
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

SOURCE_ORDER = ["wildchat", "oasst1", "olmo-instruct"]
SLOT_NAMES = {"user": "user slot", "asst": "assistant slot", "plain": "plain (no slot)"}


def load_scores(score_dir):
    frames = {os.path.basename(p)[:-4]: pd.read_csv(p) for p in glob.glob(os.path.join(score_dir, "*.csv"))}
    if not frames:
        raise SystemExit(f"no score CSVs in {score_dir}")
    stages = [s for s in config.STAGES if s in frames] + sorted(s for s in frames if s not in config.STAGES)
    return {s: frames[s] for s in stages}


def text_sets(frames):
    sets = {(src, kind) for df in frames.values() for src, kind in zip(df.source, df.kind)}
    rank = lambda s: (s[1] != "user", SOURCE_ORDER.index(s[0]) if s[0] in SOURCE_ORDER else len(SOURCE_ORDER), s[0])
    return sorted(sets, key=rank)


def set_name(src, kind):
    return f"{src} {kind}"


def bootstrap_ci(x, n_boot, rng):
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return x.mean(), lo, hi


def quantities(kind):
    """[(quantity label, function of the text x condition table)] for one text set."""
    k = config.KIND_ABBR[kind]
    own = "user" if kind == "user" else "asst"
    qs = [(f"NLL {SLOT_NAMES[a]} ({k}@{a})", (lambda t, c=f"{k}@{a}": t[c])) for a, _ in config.SLOTS]
    qs.append(("slot effect (user - assistant slot)", lambda t: t[f"{k}@user"] - t[f"{k}@asst"]))
    qs.append((f"role-specific ({k}@{own} - {k}@plain)", lambda t: t[f"{k}@{own}"] - t[f"{k}@plain"]))
    return qs


def summarize(frames, n_boot, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for stage, df in frames.items():
        for src, kind in text_sets(frames):
            d = df[(df.source == src) & (df.kind == kind)]
            if d.empty:
                continue
            t = d.pivot(index="text_id", columns="condition", values="mean_nll")
            for q, f in quantities(kind):
                m, lo, hi = bootstrap_ci(f(t), n_boot, rng)
                rows.append(dict(stage=stage, source=src, kind=kind, quantity=q, n=len(t), mean=m, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def plot(summary, out_dir):
    stages = list(dict.fromkeys(summary["stage"]))
    x = np.arange(len(stages))
    paths = []
    for (src, kind), g in summary.groupby(["source", "kind"], sort=False):
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for a, slot in config.SLOTS:
            d = g[g.quantity.str.startswith(f"NLL {SLOT_NAMES[a]} (")].set_index("stage").reindex(stages)
            ax.errorbar(x, d["mean"], yerr=[d["mean"] - d["lo"], d["hi"] - d["mean"]], capsize=3, marker="o",
                        linestyle=":" if slot is None else "-", label=SLOT_NAMES[a])
        ax.set_xticks(x, stages)
        ax.set_xlabel("post-training stage")
        ax.set_ylabel("mean per-token NLL of content (nats)")
        ax.set_title(f"{set_name(src, kind)} text (n={g.n.iloc[0]}), 95% bootstrap CI")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, f"nll_{src}_{kind}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def fmt(r):
    return f"{r['mean']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"


def table(summary, select, index):
    d = summary[select(summary)].copy()
    d["text set"] = [set_name(s, k) for s, k in zip(d.source, d.kind)]
    d["cell"] = d.apply(fmt, axis=1)
    order = list(dict.fromkeys(zip(*(d[c] for c in index)) if len(index) > 1 else d[index[0]]))
    wide = d.pivot_table(index=index, columns="stage", values="cell", aggfunc="first")
    wide = wide.reindex(order)[list(dict.fromkeys(summary["stage"]))]
    return wide.reset_index().to_markdown(index=False)


def results_table(summary, gen_dir):
    sets = summary.drop_duplicates(["source", "kind"])
    md = ["# Results", "",
          "Mean over texts, 95% bootstrap CI over texts. Text sources are analyzed separately and never pooled.", "",
          "Texts per set: " + ", ".join(f"{set_name(s, k)} {n}" for s, k, n in zip(sets.source, sets.kind, sets.n)) + ".",
          "", "## Raw mean NLL by text set and slot", "",
          table(summary, lambda d: d.quantity.str.startswith("NLL "), ["text set", "quantity"]),
          "", "## Slot effect: NLL(user slot) - NLL(assistant slot)", "",
          table(summary, lambda d: d.quantity.str.startswith("slot effect"), ["text set"]),
          "", "## Role-specific: NLL(own slot) - NLL(plain)", "",
          "user@user - user@plain for user text; asst@asst - asst@plain for assistant text.", "",
          table(summary, lambda d: d.quantity.str.startswith("role-specific"), ["text set", "quantity"])]
    counts = sorted(glob.glob(os.path.join(gen_dir, "*_counts.csv")))
    if counts:
        g = pd.concat(pd.read_csv(p) for p in counts).set_index("stage")
        g = g.reindex([s for s in config.STAGES if s in g.index] + [s for s in g.index if s not in config.STAGES])
        md += ["", "## Prefilled user-turn continuations (counts of samples)", "",
               "ended_turn: closed the user turn; handoff_to_assistant: then wrote the assistant header (a subset of "
               "ended_turn); assistant_register and repetition_loop: checked on the user portion only.", "",
               g.to_markdown()]
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
    plot(summary, args.out)
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(results_table(summary, args.generations))
    print(open(os.path.join(args.out, "results.md")).read())


if __name__ == "__main__":
    main()
