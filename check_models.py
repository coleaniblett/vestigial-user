"""Verify model IDs, licenses and chat templates via the HF API (no weights downloaded).

    python check_models.py      # writes docs/models.md

Downloads the Instruct tokenizer + config, fetches only the tokenizer files of the
other checkpoints, and checks the three post-trained checkpoints share one chat template
and all four share one tokenizer. Both checks are semantic: templates are compared by what
they render for a plain user/assistant exchange (the files differ in a tool-handling
null check), tokenizers by vocab + merges + added tokens (the files differ in how merges
are serialized and in the names/ids of a few reserved tokens).
"""

import hashlib
import json

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig, AutoTokenizer

import config

DEMO = [{"role": "user", "content": "USER"}, {"role": "assistant", "content": "ASSISTANT"}]


def chat_template(repo, files):
    if "chat_template.jinja" in files:
        return open(hf_hub_download(repo, "chat_template.jinja")).read()
    return json.load(open(hf_hub_download(repo, "tokenizer_config.json"))).get("chat_template")


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12] if s else None


def tokenizer_core(path):
    """(vocab, merges, added tokens) with merges normalized to pairs regardless of serialization."""
    t = json.load(open(path))
    merges = [tuple(m) if isinstance(m, list) else tuple(m.split(" ")) for m in t["model"]["merges"]]
    added = {(a["id"], a["content"]) for a in t.get("added_tokens", [])}
    return t["model"]["vocab"], merges, added


def diff_vocab(a, b):
    return sorted((k, a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k))


def main():
    api = HfApi()
    lines = ["# Model checks", "", "| stage | repo | sha | license | gated | weights (GB) | vocab | chat template sha | tokenizer.json sha |",
             "|---|---|---|---|---|---|---|---|---|"]
    templates, tok_hashes, cores = {}, {}, {}
    for stage, repo in config.MODELS.items():
        info = api.model_info(repo, files_metadata=True)
        files = [s.rfilename for s in info.siblings]
        size = sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors")) / 1e9
        license_ = (info.card_data or {}).get("license") if info.card_data else None
        cfg = json.load(open(hf_hub_download(repo, "config.json")))
        templates[stage] = chat_template(repo, files)
        if "tokenizer.json" in files:
            path = hf_hub_download(repo, "tokenizer.json")
            tok_hashes[stage] = sha(open(path).read())
            cores[stage] = tokenizer_core(path)
        else:
            tok_hashes[stage] = cores[stage] = None
        lines.append(f"| {stage} | `{repo}` | {info.sha[:10]} | {license_} | {info.gated} | {size:.1f} | "
                     f"{cfg.get('vocab_size')} | {sha(templates[stage])} | {tok_hashes[stage]} |")

    tok = AutoTokenizer.from_pretrained(config.TEMPLATE_MODEL)
    AutoConfig.from_pretrained(config.TEMPLATE_MODEL)

    # Chat templates: byte equality, then rendering equality on a plain exchange (no tools).
    post = ["sft", "dpo", "instruct"]
    same_bytes = len({templates[s] for s in post}) == 1
    renders = {s: tok.apply_chat_template(DEMO, chat_template=templates[s], tokenize=False) for s in post}
    same_render = len(set(renders.values())) == 1
    lines += ["", "## Chat templates", "",
              f"Post-trained checkpoints share one chat template file: **{same_bytes}**.",
              f"Post-trained templates render a plain user/assistant exchange identically: **{same_render}**."]
    if not same_bytes:
        import difflib
        for other in sorted({templates[s] for s in post} - {templates["instruct"]}):
            who = ", ".join(s for s in post if templates[s] == other)
            d = difflib.unified_diff(templates["instruct"].splitlines(), other.splitlines(), "instruct", who, lineterm="", n=0)
            lines += ["", f"Template file of {who} vs instruct (unified diff):", "", "```diff", *list(d)[2:], "```"]
    lines.append(f"\nBase has its own chat template: {templates['base'] is not None}; it is scored with the Instruct template.")

    # Tokenizers: byte equality, then vocab/merges/added-token equality.
    ref = cores["instruct"]
    same_tok_bytes = len(set(tok_hashes.values())) == 1
    same_merges = all(c[1] == ref[1] for c in cores.values())
    vocab_diffs = {s: diff_vocab(c[0], ref[0]) for s, c in cores.items()}
    added_diffs = {s: (c[2] - ref[2], ref[2] - c[2]) for s, c in cores.items()}
    same_vocab = all(not d for d in vocab_diffs.values())
    lines += ["", "## Tokenizers", "",
              f"All four share one tokenizer.json file: **{same_tok_bytes}**.",
              f"All four share one merge list: **{same_merges}**.",
              f"All four share one vocab: **{same_vocab}**."]
    for s, d in vocab_diffs.items():
        if d:
            lines += ["", f"`{s}` vocab entries differing from Instruct (token, {s} id, instruct id):", ""]
            lines += [f"- `{k}`: {a} vs {b}" for k, a, b in d]
    # Differences among reserved/added tokens (never present in scored text, see render.check_text)
    # are harmless; a differing id for an ordinary vocab entry is not.
    reserved = {name for c in cores.values() for _, name in c[2]}
    only_reserved = all(k in reserved for d in vocab_diffs.values() for k, _, _ in d)
    lines.append(f"\nVocab differences are confined to reserved/added tokens: **{only_reserved}**.")

    demo = tok.apply_chat_template(DEMO, tokenize=False)
    lines += ["", f"Special tokens: bos={tok.bos_token!r} eos={tok.eos_token!r} pad={tok.pad_token!r}", "",
              "Instruct template rendering of a user + assistant exchange:", "", "```", demo, "```", "",
              "Instruct chat template:", "", "```jinja", templates["instruct"] or "", "```", ""]
    with open("docs/models.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[:20]))
    assert same_render, "post-trained chat templates render differently"
    assert same_merges and only_reserved, "tokenizers differ beyond reserved tokens"


if __name__ == "__main__":
    main()
