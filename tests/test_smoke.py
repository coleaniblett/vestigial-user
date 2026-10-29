"""End-to-end smoke test on CPU: prepare_data -> add_replies -> score -> generate -> analyze with SmolLM2-135M-Instruct.

Uses oasst1 pairs without the Dolci exclusion so it stays small and fast. The same model is
scored under two stage labels so analyze.py has more than one stage to plot.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

import config
from render import Rendered, load_jsonl, load_tokenizer, render_all
from score import content_nll, load_model

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
STAGES = ["base", "instruct"]
ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    d = tmp_path_factory.mktemp("smoke")
    common = ["--tokenizer", MODEL]
    run("prepare_data.py", "--pairs", "oasst1", "--extra-user", "none", "--no-exclude-dolci", *common, "--n", "10",
        "--n-prefill", "3", "--out", str(d / "texts.jsonl"), "--prefill-out", str(d / "prefill.jsonl"))
    run("add_replies.py", "--model", MODEL, *common, "--dataset", str(d / "texts.jsonl"), "--prompt-source", "oasst1",
        "--source", "smollm")
    for stage in STAGES:
        run("score.py", "--model", MODEL, "--stage", stage, *common, "--dataset", str(d / "texts.jsonl"),
            "--batch-size", "4", "--out", str(d / "scores"))
        run("generate.py", "--model", MODEL, "--stage", stage, *common, "--prefill", str(d / "prefill.jsonl"),
            "--samples", "2", "--max-new-tokens", "16", "--out", str(d / "generations"))
    # three-turn conversations stitched from the oasst1 pairs, for the role-swap path
    rows = load_jsonl(d / "texts.jsonl")
    users = [r["text"] for r in rows if r["source"] == "oasst1" and r["kind"] == "user"]
    assts = [r["text"] for r in rows if r["source"] == "oasst1" and r["kind"] == "assistant"]
    with open(d / "swap.jsonl", "w", encoding="utf-8") as f:
        for i in range(4):
            f.write(json.dumps(dict(id=f"swap-{i:04d}", p1=users[i], r1=assts[i], p2=users[i + 1])) + "\n")
    run("role_swap.py", "replies", "--model", MODEL, *common, "--dataset", str(d / "swap.jsonl"), "--max-new-tokens", "64")
    run("coin_toss.py", "build", "--n-items", "4", "--out", str(d / "coin_toss.jsonl"))
    for stage in STAGES:
        run("role_swap.py", "score", "--model", MODEL, "--stage", stage, *common, "--dataset", str(d / "swap.jsonl"),
            "--out", str(d / "role_swap"))
        run("coin_toss.py", "score", "--model", MODEL, "--stage", stage, *common, "--dataset", str(d / "coin_toss.jsonl"),
            "--out", str(d / "coin_toss"))
    run("role_swap.py", "analyze", "--scores", str(d / "role_swap"), "--dataset", str(d / "swap.jsonl"),
        "--out-dir", str(d), "--n-boot", "200")
    run("coin_toss.py", "analyze", "--scores", str(d / "coin_toss"), "--dataset", str(d / "coin_toss.jsonl"),
        *common, "--out-dir", str(d), "--n-boot", "200")
    run("analyze.py", "--scores", str(d / "scores"), "--generations", str(d / "generations"),
        "--out", str(d), "--n-boot", "200", "--positions", "10")
    return d


def text_sets(rows):
    return {(r["source"], r["kind"]) for r in rows}


def test_data(pipeline):
    rows = load_jsonl(pipeline / "texts.jsonl")
    assert len({r["id"] for r in rows}) == len(rows)
    assert sum(r["source"] == "oasst1" for r in rows) == 20
    assert text_sets(rows) <= {("oasst1", "user"), ("oasst1", "assistant"), ("smollm", "assistant")}
    for r in rows:
        assert config.MIN_TOKENS <= r["n_tokens"] <= config.MAX_TOKENS
    pairs = {r["pair_id"] for r in rows if r["source"] == "oasst1"}
    assert all(r["pair_id"] in pairs for r in rows if r["source"] == "smollm")
    prefill = load_jsonl(pipeline / "prefill.jsonl")
    assert len(prefill) == 3
    texts = {r["id"]: r["text"] for r in rows}
    assert all(texts[p["text_id"]].startswith(p["prefill"]) for p in prefill)


def test_scores(pipeline):
    rows = load_jsonl(pipeline / "texts.jsonl")
    for stage in STAGES:
        df = pd.read_csv(pipeline / "scores" / f"{stage}.csv")
        assert list(df.columns) == ["text_id", "source", "kind", "condition", "n_tokens", "mean_nll", "mean_nll_from1",
                                    "mean_entropy", "mean_entropy_from1"]
        assert (df.mean_entropy > 0).all() and df.mean_entropy_from1.notna().all()
        assert len(df) == len(rows) * len(config.SLOTS)
        for kind, d in df.groupby("kind"):
            assert set(d.condition) == {c for c, _ in config.conditions(kind)}
        assert df.mean_nll.notna().all() and (df.mean_nll > 0).all()
        # same content tokens in every slot of a text
        assert (df.groupby("text_id").n_tokens.nunique() == 1).all()


def test_generations_and_analysis(pipeline):
    for stage in STAGES:
        lines = load_jsonl(pipeline / "generations" / f"{stage}.jsonl")
        assert len(lines) == 3 * 2
        assert {"ended_turn", "handoff_to_assistant", "assistant_register", "repetition_loop"} <= set(lines[0])
    md = (pipeline / "results.md").read_text()
    assert "Raw mean NLL" in md and "Slot effect" in md and "Role-specific" in md
    for src, kind in text_sets(load_jsonl(pipeline / "texts.jsonl")):
        assert f"{src} {kind}" in md
        assert (pipeline / f"nll_{src}_{kind}.png").stat().st_size > 0


def test_suffix_does_not_change_scores(pipeline):
    """The end-of-turn suffix differs by role but follows the scored span, so it must not matter."""
    tok = load_tokenizer(MODEL)
    model, device = load_model(MODEL, torch.float32)
    full = [((r["id"], c), x) for r in load_jsonl(pipeline / "texts.jsonl")[:4] for c, x in render_all(tok, r).items()]
    cut = [(k, Rendered(x.input_ids[: x.end], x.start, x.end, x.text)) for k, x in full]
    assert any(len(x.input_ids) > x.end for _, x in full)
    a = content_nll(model, device, full, tok.pad_token_id, batch_size=1)
    b = content_nll(model, device, cut, tok.pad_token_id, batch_size=1)
    for k in a:
        torch.testing.assert_close(a[k], b[k], atol=1e-4, rtol=1e-4)


def test_batching_matches_single(pipeline):
    tok = load_tokenizer(MODEL)
    model, device = load_model(MODEL, torch.float32)
    items = [((r["id"], c), x) for r in load_jsonl(pipeline / "texts.jsonl")[:4] for c, x in render_all(tok, r).items()]
    batched = content_nll(model, device, items, tok.pad_token_id, batch_size=8)
    single = content_nll(model, device, items, tok.pad_token_id, batch_size=1)
    for k in single:
        torch.testing.assert_close(batched[k], single[k], atol=1e-4, rtol=1e-4)


def test_token_nll_matches_mean(pipeline):
    for stage in STAGES:
        means = pd.read_csv(pipeline / "scores" / f"{stage}.csv").set_index(["text_id", "condition"])
        tok = pd.read_parquet(pipeline / "scores" / f"{stage}_tokens.parquet")
        g = tok.groupby(["text_id", "condition"], observed=True).nll
        assert len(g) == len(means)
        assert (g.size() == means.n_tokens.reindex(g.size().index)).all()
        assert (g.mean() - means.mean_nll.reindex(g.mean().index)).abs().max() < 1e-5
        e = tok.groupby(["text_id", "condition"], observed=True).entropy.mean()
        assert (e - means.mean_entropy.reindex(e.index)).abs().max() < 1e-5
        late = tok[tok.position >= 1].groupby(["text_id", "condition"], observed=True).nll.mean()
        assert (late - means.mean_nll_from1.reindex(late.index)).abs().max() < 1e-5
    assert "negligible" in (pipeline / "position.md").read_text()


def test_role_swap(pipeline):
    swap = load_jsonl(pipeline / "swap.jsonl")
    n_olmo = sum(r["r1_olmo"] is not None for r in swap)
    assert n_olmo > 0
    for stage in STAGES:
        df = pd.read_csv(pipeline / "role_swap" / f"{stage}.csv")
        three, two = df[df.format == "3turn"], df[df.format == "2turn"]
        assert len(three) == 4 * 2 * 2 and set(three.turn) == {"R1", "P2"}
        assert len(two) == (4 + n_olmo) * 2 and set(two.turn) == {"R1"}
        assert set(two.r1_source) == {"chatgpt", "olmo-instruct"} and set(df.labeling) == {"normal", "flipped"}
        n = df.pivot_table(index=["conv_id", "format", "r1_source", "turn"], columns="labeling", values="n_tokens")
        assert (n.normal == n.flipped).all()
        assert (df.mean_entropy > 0).all()
    md = (pipeline / "role_swap.md").read_text()
    assert "label effect R1 (flipped - normal)" in md and "relative to base" in md
    assert "interaction" in md and "Circularity" in md and "entropy" in md


def test_coin_toss(pipeline):
    for stage in STAGES:
        df = pd.read_csv(pipeline / "coin_toss" / f"{stage}.csv")
        assert len(df) == 8 * 3 * 2 * 6 and (df.logprob < 0).all()
    md = (pipeline / "coin_toss.md").read_text()
    assert "assistant tilt - user tilt" in md and "relative to base" in md
