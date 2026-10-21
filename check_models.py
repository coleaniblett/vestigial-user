"""Verify model IDs, licenses and chat templates via the HF API (no weights downloaded).

    python check_models.py      # writes docs/models.md

Downloads the Instruct tokenizer + config, fetches only the tokenizer files of the
other checkpoints, and checks the three post-trained checkpoints share one chat template.
"""

import hashlib
import json

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig, AutoTokenizer

import config


def chat_template(repo, files):
    if "chat_template.jinja" in files:
        return open(hf_hub_download(repo, "chat_template.jinja")).read()
    return json.load(open(hf_hub_download(repo, "tokenizer_config.json"))).get("chat_template")


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12] if s else None


def main():
    api = HfApi()
    lines = ["# Model checks", "", "| stage | repo | sha | license | gated | weights (GB) | vocab | chat template sha | tokenizer.json sha |",
             "|---|---|---|---|---|---|---|---|---|"]
    templates, tok_hashes = {}, {}
    for stage, repo in config.MODELS.items():
        info = api.model_info(repo, files_metadata=True)
        files = [s.rfilename for s in info.siblings]
        size = sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors")) / 1e9
        license_ = (info.card_data or {}).get("license") if info.card_data else None
        cfg = json.load(open(hf_hub_download(repo, "config.json")))
        templates[stage] = chat_template(repo, files)
        tok_hashes[stage] = sha(open(hf_hub_download(repo, "tokenizer.json")).read()) if "tokenizer.json" in files else None
        lines.append(f"| {stage} | `{repo}` | {info.sha[:10]} | {license_} | {info.gated} | {size:.1f} | "
                     f"{cfg.get('vocab_size')} | {sha(templates[stage])} | {tok_hashes[stage]} |")

    post = ["sft", "dpo", "instruct"]
    same = len({templates[s] for s in post}) == 1
    lines += ["", f"Post-trained checkpoints share one chat template: **{same}**.",
              f"Base has its own chat template: {templates['base'] is not None}; it is scored with the Instruct template.",
              f"All four share one tokenizer.json: **{len(set(tok_hashes.values())) == 1}**."]

    tok = AutoTokenizer.from_pretrained(config.TEMPLATE_MODEL)
    AutoConfig.from_pretrained(config.TEMPLATE_MODEL)
    demo = tok.apply_chat_template([{"role": "user", "content": "USER"}, {"role": "assistant", "content": "ASSISTANT"}],
                                   tokenize=False)
    lines += ["", f"Special tokens: bos={tok.bos_token!r} eos={tok.eos_token!r} pad={tok.pad_token!r}", "",
              "Instruct template rendering of a user + assistant exchange:", "", "```", demo, "```", "",
              "Instruct chat template:", "", "```jinja", templates["instruct"] or "", "```", ""]
    with open("docs/models.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[:12]))
    assert same, "post-trained checkpoints have different chat templates"


if __name__ == "__main__":
    main()
