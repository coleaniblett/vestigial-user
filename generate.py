"""Sample continuations of prefilled (unterminated) user turns, for one model.

    python generate.py --model instruct

Writes <out>/<stage>.jsonl (one line per sample, with counter flags) and
<out>/<stage>_counts.csv (totals per flag).

Flags per sample:
    ended_turn            closed the user turn (<|im_end|> or end of document)
    handoff_to_assistant  wrote the assistant header after closing the turn (a subset of ended_turn)
    assistant_register    assistant phrases in the user portion (text before the turn closes)
    repetition_loop       a word 4-gram repeated 3+ times in the user portion
A post-trained model usually finishes the user's message, closes the turn and then answers as
the assistant, so register and repetition are checked on the user portion only.
"""

import argparse
import csv
import json
import os
import re
from collections import Counter

import torch

import config
from render import SENTINEL, load_jsonl, load_tokenizer, render_prefill, template_parts
from score import load_model, resolve

ASSISTANT_REGISTER = re.compile(
    r"\b(I'd be (happy|glad) to|I would be (happy|glad) to|Certainly[,!.]|Sure[,!]|Of course[,!.]"
    r"|As an AI|I'm sorry, but|I hope this helps|Here (is|are) (a|an|the|some)\b|Great question"
    r"|Let me know if|feel free to)",
    re.I,
)


def turn_markers(tok):
    """End-of-user-turn string and the user->assistant handoff string from the template."""
    _, user_suffix = template_parts(tok, "user")
    s = tok.apply_chat_template(
        [{"role": "user", "content": SENTINEL}, {"role": "assistant", "content": SENTINEL + "2"}], tokenize=False
    )
    handoff = s.split(SENTINEL)[1]
    return user_suffix.strip(), handoff.strip()


def stop_ids(tok):
    """End-of-document token, the one stop token for every stage. Post-trained checkpoints
    also stop on <|im_end|> by default, which would hide any handoff to the assistant."""
    vocab = tok.get_vocab()
    return [vocab["<|endoftext|>"]] if "<|endoftext|>" in vocab else [tok.eos_token_id]


def repetition_loop(text, n=4, k=3):
    words = text.split()
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return any(c >= k for c in grams.values())


def user_portion(text, tok):
    """Continuation text before the model closes (or abandons) the user turn."""
    cut = min((text.find(t) for t in ("<|im_end|>", "<|im_start|>", tok.eos_token) if t and t in text), default=len(text))
    return text[:cut]


def flags(text, user_text, stopped_on_eos, end_marker, handoff):
    return dict(
        ended_turn=bool(stopped_on_eos or (end_marker and end_marker in text)),
        handoff_to_assistant=bool(handoff and handoff in text),
        assistant_register=bool(ASSISTANT_REGISTER.search(user_text)),
        repetition_loop=repetition_loop(user_text),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage")
    ap.add_argument("--tokenizer", default=config.TEMPLATE_MODEL)
    ap.add_argument("--prefill", default=config.PREFILL_PATH)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--samples", type=int, default=config.GEN_SAMPLES)
    ap.add_argument("--max-new-tokens", type=int, default=config.GEN_MAX_NEW_TOKENS)
    ap.add_argument("--temperature", type=float, default=config.GEN_TEMPERATURE)
    ap.add_argument("--preceding-turn", action="store_true", default=config.PRECEDING_TURN)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", default="results/generations")
    args = ap.parse_args()

    model_id, stage = resolve(args.model, args.stage)
    tok = load_tokenizer(args.tokenizer)
    end_marker, handoff = turn_markers(tok)
    model, device = load_model(model_id)
    eos_ids = stop_ids(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    torch.manual_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    totals, records = Counter(), []
    for item in load_jsonl(args.prefill)[: args.limit]:
        ids = torch.tensor([render_prefill(tok, item["prefill"], args.preceding_turn)], device=device)
        gen = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            do_sample=True,
            temperature=args.temperature,
            top_p=1.0,
            top_k=0,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.samples,
            eos_token_id=eos_ids,
            pad_token_id=pad_id,
        )
        for s, seq in enumerate(gen[:, ids.shape[1] :].tolist()):
            cut = next((i for i, t in enumerate(seq) if t in eos_ids), None)
            stopped = cut is not None
            text = tok.decode(seq[: cut + 1] if stopped else seq, skip_special_tokens=False)
            user_text = user_portion(text, tok)
            f = flags(text, user_text, stopped, end_marker, handoff)
            totals.update({k: int(v) for k, v in f.items()})
            totals["n"] += 1
            records.append(dict(prefill_id=item["id"], sample=s, prefill=item["prefill"], continuation=text,
                                user_portion=user_text, **f))

    with open(os.path.join(args.out, f"{stage}.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, f"{stage}_counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        keys = ["n", "ended_turn", "handoff_to_assistant", "assistant_register", "repetition_loop"]
        w.writerow(["stage", *keys])
        w.writerow([stage, *(totals[k] for k in keys)])
    print(f"{model_id}: {dict(totals)}")


if __name__ == "__main__":
    main()
