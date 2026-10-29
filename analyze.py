"""Raw NLL, slot effects and role-specific quantities per text source, with bootstrap CIs.

    python analyze.py                    # reads results/scores/*.csv (+ results/generations/*_counts.csv)

A text set is one (source, kind), e.g. wildchat user or olmo-instruct assistant. Sets are never
pooled. Every text is scored in every slot; per set and stage we report
    raw NLL        mean content NLL in the user, assistant and plain-prose slot
    slot effect    NLL(user slot) - NLL(assistant slot), paired over texts; by default without the first
                   content token (a first-turn assistant header makes the post-trained models expect
                   <|im_start|>, so that token costs 2-14 nats whatever the text), with it alongside
    role-specific  NLL(own slot) - NLL(plain): user@user - user@plain for user text,
                   asst@asst - asst@plain for assistant text
    entropy        mean next-token entropy over content tokens per slot, without and with the first token
95% bootstrap CIs are over texts. Writes results/summary.csv, results/results.md and one
figure per text set, results/nll_<source>_<kind>.png (stage on x, one line per slot).

If score.py saved per-token NLLs (<scores>/<stage>_tokens.parquet), also writes results/position.md
and results/position_<source>_<kind>.png: mean NLL by content-token position per slot and stage,
and the slot difference by position, over texts with at least --positions tokens.
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


EXCL0, ALL = "excl. position 0", "all tokens"


def quantities(kind):
    """[(quantity label, value column, function of the text x condition table)] for one text set. Slot effects
    and entropies come without the first content token (default; after a first-turn assistant header the
    post-trained models expect <|im_start|>, so that token costs 2-14 nats whatever the text) and with it."""
    k = config.KIND_ABBR[kind]
    own = "user" if kind == "user" else "asst"
    qs = [(f"NLL {SLOT_NAMES[a]} ({k}@{a})", "mean_nll", (lambda t, c=f"{k}@{a}": t[c])) for a, _ in config.SLOTS]
    slot_effect = lambda t: t[f"{k}@user"] - t[f"{k}@asst"]
    qs.append((f"slot effect, {EXCL0}", "mean_nll_from1", slot_effect))
    qs.append((f"slot effect, {ALL}", "mean_nll", slot_effect))
    qs.append((f"role-specific ({k}@{own} - {k}@plain)", "mean_nll", lambda t: t[f"{k}@{own}"] - t[f"{k}@plain"]))
    for col, label in (("mean_entropy_from1", EXCL0), ("mean_entropy", ALL)):
        qs += [(f"entropy {SLOT_NAMES[a]} ({k}@{a}), {label}", col, (lambda t, c=f"{k}@{a}": t[c])) for a, _ in config.SLOTS]
    return qs


def summarize(frames, n_boot, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for stage, df in frames.items():
        for src, kind in text_sets(frames):
            d = df[(df.source == src) & (df.kind == kind)]
            if d.empty:
                continue
            tables = {}
            for q, col, f in quantities(kind):
                if col not in d:  # score CSVs from before a column was added
                    continue
                if col not in tables:
                    tables[col] = d.pivot(index="text_id", columns="condition", values=col)
                t = tables[col]
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
          "", "## Slot effect: NLL(user slot) - NLL(assistant slot)", ""]
    if (summary.quantity == f"slot effect, {EXCL0}").any():
        md += ["Default: without the first content token. When an assistant turn directly follows the system turn, "
               "the post-trained checkpoints put nearly all next-token probability on `<|im_start|>`, so the first "
               "content token in the assistant slot costs 2-14 nats whatever the text (see position.md). The "
               "all-tokens version follows for reference.", "",
               f"### Slot effect, {EXCL0} (default)", "",
               table(summary, lambda d: d.quantity == f"slot effect, {EXCL0}", ["text set"]), "",
               f"### Slot effect, {ALL}", ""]
    md += [table(summary, lambda d: d.quantity == f"slot effect, {ALL}", ["text set"]),
          "", "## Role-specific: NLL(own slot) - NLL(plain)", "",
          "user@user - user@plain for user text; asst@asst - asst@plain for assistant text.", "",
          table(summary, lambda d: d.quantity.str.startswith("role-specific"), ["text set", "quantity"])]
    for label, note in ((EXCL0, " (default)"), (ALL, "")):
        sel = lambda d, label=label: d.quantity.str.startswith("entropy ") & d.quantity.str.endswith(label)
        if sel(summary).any():
            md += ["", f"## Mean next-token entropy by text set and slot, {label}{note}", "",
                   "Entropy (nats) of the model's next-token distribution at each content token, averaged per text.", "",
                   table(summary, sel, ["text set", "quantity"])]
    counts = sorted(glob.glob(os.path.join(gen_dir, "*_counts.csv")))
    if counts:
        g = pd.concat(pd.read_csv(p) for p in counts).set_index("stage")
        g = g.reindex([s for s in config.STAGES if s in g.index] + [s for s in g.index if s not in config.STAGES])
        md += ["", "## Prefilled user-turn continuations (counts of samples)", "",
               "ended_turn: closed the user turn; handoff_to_assistant: then wrote the assistant header (a subset of "
               "ended_turn); assistant_register and repetition_loop: checked on the user portion only.", "",
               g.to_markdown()]
    return "\n".join(md) + "\n"


def token_tables(score_dir, stages):
    """{stage: per-token NLL frame} for stages that have a <stage>_tokens.parquet."""
    out = {}
    for s in stages:
        path = os.path.join(score_dir, f"{s}_tokens.parquet")
        if os.path.exists(path):
            out[s] = pd.read_parquet(path)
    return out


def position_arrays(tokens, src, kind, n_pos):
    """(text ids, {slot abbr: array [texts, n_pos]}, {slot abbr: per-text mean beyond n_pos}) for texts
    with at least n_pos content tokens, so every position averages over the same texts."""
    k = config.KIND_ABBR[kind]
    d = tokens[(tokens.source == src) & (tokens.kind == kind)]
    length = d[d.condition == f"{k}@user"].groupby("text_id", observed=True).position.max() + 1
    ids = length[length >= n_pos].index
    d = d[d.text_id.isin(ids)]
    arrays, tails = {}, {}
    for a, _ in config.SLOTS:
        c = d[d.condition == f"{k}@{a}"]
        head = c[c.position < n_pos].pivot(index="text_id", columns="position", values="nll").reindex(ids)
        arrays[a] = head.to_numpy()
        tails[a] = c[c.position >= n_pos].groupby("text_id", observed=True).nll.mean().reindex(ids)
    return ids, arrays, tails


def negligible_from(diff, margin, n_boot, rng):
    """First position p from which the slot difference is negligible: every per-position mean from p on is
    within +-margin (no spike left) and the mean over positions p..end has a 95% CI inside +-margin
    (no steady offset). None if no such p within the positions given."""
    m = diff.mean(axis=0)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    for p in range(diff.shape[1]):
        if np.abs(m[p:]).max() >= margin:
            continue
        lo, hi = np.percentile(diff[:, p:].mean(axis=1)[idx].mean(axis=1), [2.5, 97.5])
        if -margin < lo and hi < margin:
            return p
    return None


def position_analysis(tokens, sets, n_pos, margin, n_boot, seed, out_dir):
    """Mean NLL by content-token position per slot, and the slot difference (user - assistant slot)."""
    rng = np.random.default_rng(seed)
    stages = list(tokens)
    marks = [p for p in (0, 1, 2, 3, 5, 10, 20, n_pos - 1) if p < n_pos]
    neg_rows, pos_rows, tail_rows, ns = [], [], [], {}
    for src, kind in sets:
        name = set_name(src, kind)
        fig, axes = plt.subplots(2, len(stages), figsize=(3.2 * len(stages), 6), sharex=True, sharey="row", squeeze=False)
        for j, stage in enumerate(stages):
            ids, arr, tails = position_arrays(tokens[stage], src, kind, n_pos)
            ns[name] = len(ids)
            if not len(ids):
                continue
            # position 0 can be off any useful scale (post-trained models expect <|im_start|> right after a
            # first-turn assistant header), so panels start at 1 and print position 0's values
            x = np.arange(1, n_pos)
            first = []
            for a, slot in config.SLOTS:
                mean = arr[a].mean(axis=0)
                axes[0, j].plot(x, mean[1:], linestyle=":" if slot is None else "-", label=SLOT_NAMES[a])
                first.append(f"{a} {mean[0]:.1f}")
            axes[0, j].text(0.97, 0.95, "position 0: " + ", ".join(first), transform=axes[0, j].transAxes,
                            ha="right", va="top", fontsize=7)
            diff = arr["user"] - arr["asst"]
            idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
            boot = diff[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            m = diff.mean(axis=0)
            ax = axes[1, j]
            ax.axhspan(-margin, margin, color="0.9", zorder=0)
            ax.axhline(0, color="0.5", lw=0.8)
            ax.fill_between(x, lo[1:], hi[1:], alpha=0.3)
            ax.plot(x, m[1:])
            ax.text(0.97, 0.05, f"position 0: {m[0]:+.2f} [{lo[0]:+.2f}, {hi[0]:+.2f}]", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7)
            axes[0, j].set_title(stage)
            ax.set_xlabel("content token position")
            p = negligible_from(diff, margin, n_boot, rng)
            neg_rows.append(dict(**{"text set": name, "stage": stage}, cell="not within first %d" % n_pos if p is None else str(p)))
            for q in marks:
                pos_rows.append(dict(**{"text set": name, "position": q, "stage": stage},
                                     cell=f"{m[q]:+.3f} [{lo[q]:+.3f}, {hi[q]:+.3f}]"))
            t = (tails["user"] - tails["asst"]).dropna()
            tm, tlo, thi = bootstrap_ci(t, n_boot, rng) if len(t) else (np.nan,) * 3
            whole = diff.mean(axis=1)
            wm, wlo, whi = bootstrap_ci(whole, n_boot, rng)
            tail_rows.append(dict(**{"text set": name, "stage": stage},
                                  cell=f"{wm:+.3f} [{wlo:+.3f}, {whi:+.3f}] / {tm:+.3f} [{tlo:+.3f}, {thi:+.3f}] (n={len(t)})"))
        axes[0, 0].set_ylabel("mean NLL (nats)")
        axes[1, 0].set_ylabel("user slot - assistant slot")
        axes[0, -1].legend(fontsize=7)
        fig.suptitle(f"{name} text, n={ns[name]} texts with >= {n_pos} tokens\n"
                     f"top: mean NLL by position; bottom: user slot - assistant slot, 95% CI, shaded +-{margin}; "
                     "position 0 printed in each panel",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"position_{src}_{kind}.png"), dpi=130)
        plt.close(fig)

    first_rows = []
    for src, kind in sets:
        k = config.KIND_ABBR[kind]
        for stage in stages:
            d = tokens[stage][(tokens[stage].source == src) & (tokens[stage].kind == kind)]
            for p0, label in ((0, "all tokens"), (1, "excluding position 0")):
                per = {a: d[(d.condition == f"{k}@{a}") & (d.position >= p0)].groupby("text_id", observed=True).nll.mean()
                       for a in ("user", "asst")}
                bm, blo, bhi = bootstrap_ci(per["user"] - per["asst"], n_boot, rng)
                first_rows.append(dict(**{"text set": set_name(src, kind), "tokens": label, "stage": stage},
                                       cell=f"{bm:+.3f} [{blo:+.3f}, {bhi:+.3f}]"))

    def wide(rows, index):
        d = pd.DataFrame(rows)
        order = list(dict.fromkeys(zip(*(d[c] for c in index)) if len(index) > 1 else d[index[0]]))
        w = d.pivot_table(index=index, columns="stage", values="cell", aggfunc="first").reindex(order)[stages]
        return w.reset_index().to_markdown(index=False)

    md = ["# Slot difference by token position", "",
          f"Per text set, texts with at least {n_pos} content tokens (so every position averages over the same texts): "
          + ", ".join(f"{k} {v}" for k, v in ns.items()) + ". Slot difference = NLL(user slot) - NLL(assistant "
          "slot) at each content-token position (0 = first token after the role header), paired over texts, 95% "
          "bootstrap CI over texts. Figures: `position_<source>_<kind>.png`.", "",
          "## Position from which the slot difference is negligible", "",
          f"First position p such that every per-position mean slot difference from p to {n_pos - 1} is within "
          f"+-{margin} nats and their mean over positions p..{n_pos - 1} has a 95% CI inside +-{margin} nats.", "",
          wide(neg_rows, ["text set"]), "",
          "## Slot difference at selected positions", "",
          wide(pos_rows, ["text set", "position"]), "",
          f"## Mean slot difference over positions 0..{n_pos - 1} / over positions {n_pos}+", "",
          f"Second value: per-text mean over the tokens after position {n_pos - 1}, for texts longer than {n_pos} tokens.", "",
          wide(tail_rows, ["text set"]), "",
          "## Slot effect with and without the first content token", "",
          "Per-text mean NLL(user slot) - NLL(assistant slot) over all content tokens (as in results.md) and over "
          "positions 1+, all texts of each set.", "",
          wide(first_rows, ["text set", "tokens"]), ""]
    with open(os.path.join(out_dir, "position.md"), "w") as f:
        f.write("\n".join(md))
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/scores")
    ap.add_argument("--generations", default="results/generations")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--positions", type=int, default=40, help="content-token positions in the position analysis")
    ap.add_argument("--margin", type=float, default=0.05, help="nats; slot difference counted as negligible within this")
    args = ap.parse_args()

    frames = load_scores(args.scores)
    summary = summarize(frames, args.n_boot, args.seed)
    os.makedirs(args.out, exist_ok=True)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    plot(summary, args.out)
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(results_table(summary, args.generations))
    print(open(os.path.join(args.out, "results.md")).read())
    tokens = token_tables(args.scores, list(frames))
    if tokens:
        print(position_analysis(tokens, text_sets(frames), args.positions, args.margin, min(args.n_boot, 2000),
                                args.seed, args.out))


if __name__ == "__main__":
    main()
