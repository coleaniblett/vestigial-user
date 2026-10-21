"""Span alignment on the OLMo 3 Instruct tokenizer and chat template."""

import os

import pytest

import config
from render import AlignmentError, load_jsonl, load_tokenizer, render, render_all, template_parts

EDGE_CASES = [
    "Hello there.",
    "(parenthetical) opening",
    "123 numbers first, then words.",
    "-dash at the start",
    "'single' and \"double\" quotes",
    "日本語のテキストです。",
    "🎉 emoji first",
    "```python\nprint('hi')\n```",
    "Line one.\n\nLine two after a blank line.",
    "Ends with a colon:",
    "Tabs\tand   spaces inside",
    "ünïcödé àccents",
]


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(config.TEMPLATE_MODEL)


def test_slots_differ_only_in_role_header(tok):
    pu, su = template_parts(tok, "user")
    pa, sa = template_parts(tok, "assistant")
    assert su == sa
    assert pu != pa
    common = os.path.commonprefix([pu, pa])
    assert pu[len(common) :].replace("user", "assistant") == pa[len(common) :]


@pytest.mark.parametrize("preceding", [False, True])
@pytest.mark.parametrize("text", EDGE_CASES)
def test_edge_cases_align(tok, text, preceding):
    row = {"id": "x", "user": text, "assistant": text}
    out = render_all(tok, row, preceding)
    spans = {c: r.input_ids[r.start : r.end] for c, r in out.items()}
    assert len({tuple(s) for s in spans.values()}) == 1
    for r in out.values():
        assert tok.decode(r.input_ids[r.start : r.end], clean_up_tokenization_spaces=False) == text
        assert r.start > 0


def test_rejects_bad_text(tok):
    with pytest.raises(AlignmentError):
        render(tok, " leading space", "user")
    with pytest.raises(AlignmentError):
        render(tok, "has <|im_start|> inside", "user")


@pytest.mark.skipif(not os.path.exists(config.DATA_PATH), reason="no committed sample yet")
@pytest.mark.parametrize("preceding", [False, True])
def test_committed_sample_aligns(tok, preceding):
    rows = load_jsonl(config.DATA_PATH)
    assert len(rows) == config.N_TEXTS
    assert len({r["id"] for r in rows}) == len(rows)
    for row in rows:
        out = render_all(tok, row, preceding)
        assert out["user@user"].end - out["user@user"].start == row["n_user_tokens"]
        assert out["asst@asst"].end - out["asst@asst"].start == row["n_assistant_tokens"]
