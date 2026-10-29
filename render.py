"""Build each scoring condition from the chat template and locate the content tokens.

Each data row is one text (a source and a kind, user or assistant). It is rendered alone
as the first turn (optionally after one fixed neutral turn), so across slot conditions
only the role header differs. The content is tokenized on its
own and the full sequence must equal prefix + content + suffix tokenized separately;
this rules out merges at the header/footer boundary and makes the content tokens
identical in every condition, plain-prose control included.

    python render.py --tokenizer allenai/Olmo-3-7B-Instruct --n 2   # dump examples per source to docs/
"""

import argparse
import json
from dataclasses import dataclass

from transformers import AutoTokenizer

import config

SENTINEL = "CONTENT"


class AlignmentError(ValueError):
    pass


@dataclass
class Rendered:
    input_ids: list[int]
    start: int  # content occupies input_ids[start:end]
    end: int
    text: str  # the full rendered string


def load_tokenizer(name):
    return AutoTokenizer.from_pretrained(name)


def special_strings(tok):
    return {s for s in tok.all_special_tokens + list(tok.get_added_vocab()) if s}


def check_text(tok, text):
    """Raise AlignmentError if a text cannot be scored cleanly with this tokenizer."""
    if text != text.strip():
        raise AlignmentError("text has leading/trailing whitespace")
    for s in special_strings(tok):
        if s in text:
            raise AlignmentError(f"text contains special token {s!r}")


def messages_for(slot, content, preceding=False):
    msgs = []
    if preceding:
        other = "assistant" if slot == "user" else "user"
        msgs.append({"role": other, "content": config.NEUTRAL_TURN[other]})
    msgs.append({"role": slot, "content": content})
    return msgs


def template_parts(tok, slot, preceding=False):
    """Rendered string before and after the scored turn's content."""
    s = tok.apply_chat_template(messages_for(slot, SENTINEL, preceding), tokenize=False)
    if s.count(SENTINEL) != 1:
        raise AlignmentError("template does not place content verbatim exactly once")
    prefix, suffix = s.split(SENTINEL)
    return prefix, suffix


def doc_start_ids(tok):
    """Token(s) that open a plain-prose document, so its first content token is scored."""
    vocab = tok.get_vocab()
    if "<|endoftext|>" in vocab:  # document separator (OLMo, SmolLM); their bos may be a chat token
        return [vocab["<|endoftext|>"]]
    return [tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id]


def encode(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


def render(tok, text, slot, preceding=False):
    """slot: "user", "assistant", or None for plain prose."""
    check_text(tok, text)
    content = encode(tok, text)
    if tok.decode(content, clean_up_tokenization_spaces=False) != text:
        raise AlignmentError("decoding the content tokens does not reproduce the text")

    if slot is None:
        ids = doc_start_ids(tok) + content
        return Rendered(ids, len(ids) - len(content), len(ids), tok.decode(ids))

    prefix, suffix = template_parts(tok, slot, preceding)
    full = prefix + text + suffix
    if tok.apply_chat_template(messages_for(slot, text, preceding), tokenize=False) != full:
        raise AlignmentError("template modifies the content")
    pre, suf = encode(tok, prefix), encode(tok, suffix)
    ids = encode(tok, full)
    if ids != pre + content + suf:
        raise AlignmentError("tokens merge across the content boundary")
    start, end = len(pre), len(pre) + len(content)
    assert tok.decode(ids[start:end], clean_up_tokenization_spaces=False) == text
    return Rendered(ids, start, end, full)


def render_all(tok, row, preceding=False):
    """One text (a data row) in every slot; asserts its content tokens match across slots."""
    out = {cond: render(tok, row["text"], slot, preceding) for cond, slot in config.conditions(row["kind"])}
    spans = [r.input_ids[r.start : r.end] for r in out.values()]
    assert all(s == spans[0] for s in spans), f"content tokens differ across slots ({row['id']})"
    return out


@dataclass
class RenderedConversation:
    input_ids: list[int]
    spans: list[tuple[int, int]]  # content of turn i occupies input_ids[start:end]
    text: str


def render_conversation(tok, contents, roles):
    """Several turns with the given roles (default system turn included). Each content is tokenized
    on its own and the full sequence must equal the template pieces and contents tokenized
    separately, so a content's tokens do not depend on the roles around it."""
    marks = [f"<<<{SENTINEL}{i}>>>" for i in range(len(contents))]
    msgs = lambda texts: [{"role": r, "content": t} for r, t in zip(roles, texts)]
    s = tok.apply_chat_template(msgs(marks), tokenize=False)
    pieces, rest = [], s
    for m in marks:
        if rest.count(m) != 1:
            raise AlignmentError("template does not place each content verbatim exactly once")
        before, rest = rest.split(m)
        pieces.append(before)
    pieces.append(rest)

    ids, spans, full = [], [], ""
    for i, text in enumerate(contents):
        check_text(tok, text)
        content = encode(tok, text)
        if tok.decode(content, clean_up_tokenization_spaces=False) != text:
            raise AlignmentError("decoding the content tokens does not reproduce the text")
        ids += encode(tok, pieces[i])
        spans.append((len(ids), len(ids) + len(content)))
        ids += content
        full += pieces[i] + text
    ids += encode(tok, pieces[-1])
    full += pieces[-1]
    if tok.apply_chat_template(msgs(contents), tokenize=False) != full:
        raise AlignmentError("template modifies the content")
    if encode(tok, full) != ids:
        raise AlignmentError("tokens merge across a content boundary")
    return RenderedConversation(ids, spans, full)


def render_prefill(tok, sentence, preceding=False):
    """User turn opened with `sentence` and left unterminated, for generation."""
    check_text(tok, sentence)
    prefix, _ = template_parts(tok, "user", preceding)
    return encode(tok, prefix + sentence)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def dump_examples(tok, rows, path, preceding=False):
    lines = [
        "# Rendered examples",
        "",
        f"Tokenizer/template: `{tok.name_or_path}`; preceding neutral turn: {preceding}.",
        "Scored content is wrapped in ⟦ ⟧ (markers added for display only).",
        "",
    ]
    for row in rows:
        lines.append(f"## {row['id']} ({row['source']} {row['kind']} text)\n")
        for cond, r in render_all(tok, row, preceding).items():
            before = tok.decode(r.input_ids[: r.start])
            content = tok.decode(r.input_ids[r.start : r.end])
            after = tok.decode(r.input_ids[r.end :])
            lines += [
                f"### {cond} ({r.end - r.start} content tokens of {len(r.input_ids)})",
                "",
                "````",  # four backticks: texts may contain ``` code fences
                f"{before}⟦{content}⟧{after}",
                "````",
                "",
            ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
    ap.add_argument("--dataset", default=config.DATA_PATH)
    ap.add_argument("--n", type=int, default=2, help="examples per source and kind")
    ap.add_argument("--preceding-turn", action="store_true", default=config.PRECEDING_TURN)
    ap.add_argument("--out", default="docs/rendered_examples.md")
    args = ap.parse_args()
    tok = load_tokenizer(args.tokenizer)
    rows, count = [], {}
    for row in load_jsonl(args.dataset):
        key = (row["source"], row["kind"])
        if count.get(key, 0) < args.n:
            count[key] = count.get(key, 0) + 1
            rows.append(row)
    dump_examples(tok, rows, args.out, args.preceding_turn)
    print(f"wrote {args.out}")
