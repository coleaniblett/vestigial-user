"""End-to-end smoke test on CPU: prepare_data -> score -> generate -> analyze with SmolLM2-135M-Instruct.

Uses the open oasst1 stand-in so it runs without a gated-dataset token. The same model is
scored under two stage labels so analyze.py has more than one stage to plot.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

import config
from render import load_jsonl, load_tokenizer, render_all
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
    run("prepare_data.py", "--dataset", "oasst1", *common, "--n", "10", "--n-prefill", "3",
        "--out", str(d / "texts.jsonl"), "--prefill-out", str(d / "prefill.jsonl"))
    for stage in STAGES:
        run("score.py", "--model", MODEL, "--stage", stage, *common, "--dataset", str(d / "texts.jsonl"),
            "--batch-size", "4", "--out", str(d / "scores"))
        run("generate.py", "--model", MODEL, "--stage", stage, *common, "--prefill", str(d / "prefill.jsonl"),
            "--samples", "2", "--max-new-tokens", "16", "--out", str(d / "generations"))
    run("analyze.py", "--scores", str(d / "scores"), "--generations", str(d / "generations"),
        "--out", str(d), "--n-boot", "200")
    return d


def test_data(pipeline):
    rows = load_jsonl(pipeline / "texts.jsonl")
    assert len(rows) == 10 and len({r["id"] for r in rows}) == 10
    for r in rows:
        assert config.MIN_TOKENS <= r["n_user_tokens"] <= config.MAX_TOKENS
        assert config.MIN_TOKENS <= r["n_assistant_tokens"] <= config.MAX_TOKENS
    prefill = load_jsonl(pipeline / "prefill.jsonl")
    assert len(prefill) == 3
    texts = {r["id"]: r["user"] for r in rows}
    assert all(texts[p["text_id"]].startswith(p["prefill"]) for p in prefill)


def test_scores(pipeline):
    conds = [c for c, _, _ in config.CONDITIONS]
    for stage in STAGES:
        df = pd.read_csv(pipeline / "scores" / f"{stage}.csv")
        assert list(df.columns) == ["text_id", "condition", "n_tokens", "mean_nll"]
        assert len(df) == 10 * len(conds) and set(df.condition) == set(conds)
        assert df.mean_nll.notna().all() and (df.mean_nll > 0).all()
        # same content tokens in every condition of a text type
        n = df.pivot(index="text_id", columns="condition", values="n_tokens")
        assert (n[["user@user", "user@asst", "user@plain"]].nunique(axis=1) == 1).all()
        assert (n[["asst@asst", "asst@user", "asst@plain"]].nunique(axis=1) == 1).all()


def test_generations_and_analysis(pipeline):
    for stage in STAGES:
        lines = load_jsonl(pipeline / "generations" / f"{stage}.jsonl")
        assert len(lines) == 3 * 2
        assert {"ended_turn", "handoff_to_assistant", "assistant_register", "repetition_loop"} <= set(lines[0])
    assert (pipeline / "nll_by_stage.png").stat().st_size > 0
    md = (pipeline / "results.md").read_text()
    assert "slot effect, user text" in md and "slot effect, assistant text" in md


def test_batching_matches_single(pipeline):
    tok = load_tokenizer(MODEL)
    model, device = load_model(MODEL)
    items = [((r["id"], c), x) for r in load_jsonl(pipeline / "texts.jsonl")[:4] for c, x in render_all(tok, r).items()]
    batched = content_nll(model, device, items, tok.pad_token_id, batch_size=8)
    single = content_nll(model, device, items, tok.pad_token_id, batch_size=1)
    for k in single:
        torch.testing.assert_close(batched[k], single[k], atol=1e-4, rtol=1e-4)
