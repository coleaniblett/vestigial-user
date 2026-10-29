"""Span alignment on the OLMo 3 Instruct tokenizer and chat template."""

import os
from collections import Counter

import pytest

import config
from generate import flags, stop_ids, turn_markers, user_portion
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
    assert pu != pa
    common = os.path.commonprefix([pu, pa])
    assert pu[len(common) :].replace("user", "assistant") == pa[len(common) :]
    # The OLMo template closes a user turn with <|im_end|>\n and a final assistant turn with
    # <|endoftext|>. The suffix comes after the scored span, so under causal attention it cannot
    # change content NLL (tests/test_smoke.py::test_suffix_does_not_change_scores); it must start
    # with a special/added token so nothing can merge with the end of the content.
    boundary_ids = set(tok.all_special_ids) | set(tok.get_added_vocab().values())
    for suffix in (su, sa):
        ids = tok(suffix, add_special_tokens=False)["input_ids"]
        assert ids and ids[0] in boundary_ids, suffix


@pytest.mark.parametrize("preceding", [False, True])
@pytest.mark.parametrize("text", EDGE_CASES)
def test_edge_cases_align(tok, text, preceding):
    spans = set()
    for kind in ("user", "assistant"):
        out = render_all(tok, {"id": "x", "kind": kind, "text": text}, preceding)
        assert len(out) == len(config.SLOTS)
        spans |= {tuple(r.input_ids[r.start : r.end]) for r in out.values()}
        for r in out.values():
            assert tok.decode(r.input_ids[r.start : r.end], clean_up_tokenization_spaces=False) == text
            assert r.start > 0
    assert len(spans) == 1


def test_generation_stops_only_at_end_of_document(tok):
    """Stopping at <|im_end|> would make a handoff to the assistant unobservable."""
    end_marker, handoff = turn_markers(tok)
    assert end_marker == "<|im_end|>" and handoff.startswith(end_marker) and "assistant" in handoff
    assert stop_ids(tok) == [tok.convert_tokens_to_ids("<|endoftext|>")]
    assert tok.convert_tokens_to_ids(end_marker) not in stop_ids(tok)


def test_rejects_bad_text(tok):
    with pytest.raises(AlignmentError):
        render(tok, " leading space", "user")
    with pytest.raises(AlignmentError):
        render(tok, "has <|im_start|> inside", "user")


@pytest.mark.skipif(not os.path.exists(config.DATA_PATH), reason="no committed sample yet")
@pytest.mark.parametrize("preceding", [False, True])
def test_committed_sample_aligns(tok, preceding):
    rows = load_jsonl(config.DATA_PATH)
    assert len({r["id"] for r in rows}) == len(rows)
    sets = Counter((r["source"], r["kind"]) for r in rows)
    for key in [(config.PAIR_SOURCE, "user"), (config.PAIR_SOURCE, "assistant"), (config.EXTRA_USER_SOURCE, "user")]:
        assert sets[key] == config.N_TEXTS, key
    assert 0 < sets[(config.REPLY_SOURCE, "assistant")] <= config.N_TEXTS
    for row in rows:
        out = render_all(tok, row, preceding)
        assert all(r.end - r.start == row["n_tokens"] for r in out.values())


def test_flags_use_user_portion_only(tok):
    end_marker, handoff = turn_markers(tok)
    loop = "the cat sat on " * 4
    text = f"my cat is sick.{handoff}\nI'd be happy to help! {loop}<|endoftext|>"
    user_text = user_portion(text, tok)
    assert user_text == "my cat is sick."
    f = flags(text, user_text, True, end_marker, handoff)
    assert f == dict(ended_turn=True, handoff_to_assistant=True, assistant_register=False, repetition_loop=False)
    unclosed = f"Sure! I'd be happy to help. {loop}"
    f = flags(unclosed, user_portion(unclosed, tok), False, end_marker, handoff)
    assert f == dict(ended_turn=False, handoff_to_assistant=False, assistant_register=True, repetition_loop=True)


@pytest.mark.parametrize("text", EDGE_CASES)
def test_role_swap_content_identical_across_labelings(tok, text):
    from role_swap import LABELINGS, render_both

    turns = [text, "A reply: " + text, text + " (follow-up)"]
    out = render_both(tok, turns)
    assert set(out) == set(LABELINGS)
    for name, r in out.items():
        assert [tok.decode(r.input_ids[a:b], clean_up_tokenization_spaces=False) for a, b in r.spans] == turns
        roles = LABELINGS[name]
        # each turn's header names its labeled role, and both labelings open with the same default system turn
        for (a, _), role in zip(r.spans, roles):
            assert tok.decode(r.input_ids[:a]).endswith(f"<|im_start|>{role}\n")
    starts = [r.input_ids[: r.spans[0][0] - 3] for r in out.values()]
    assert starts[0] == starts[1] and "system" in tok.decode(starts[0])


def test_coin_toss_prompts(tok, tmp_path):
    import subprocess
    import sys

    from coin_toss import CONDITIONS, OUTCOMES, prompt_ids, variants

    out = tmp_path / "coin.jsonl"
    subprocess.run([sys.executable, "coin_toss.py", "build", "--out", str(out)], check=True)
    rows = load_jsonl(out)
    assert len(rows) == 40 and {r["preferred"] for r in rows} == set(OUTCOMES)
    for r in rows:
        prompts = {c: tok.decode(prompt_ids(tok, r["turns"][c])) for c in CONDITIONS}
        assert all(p.endswith("I flipped the coin and it landed on") for p in prompts.values())
        # the preference sentence is said by exactly the named party, and nobody in "none"
        assert r["preferred"] in r["turns"]["user"][0] and r["preferred"] not in r["turns"]["user"][1]
        assert r["preferred"] in r["turns"]["assistant"][1] and r["preferred"] not in r["turns"]["assistant"][0]
        assert not any(o in t for o in OUTCOMES for t in r["turns"]["none"])
    heads, tails = ({tuple(tok(v, add_special_tokens=False)["input_ids"]) for v in variants(o)} for o in OUTCOMES)
    assert len(heads) == len(tails) == 6 and not heads & tails
