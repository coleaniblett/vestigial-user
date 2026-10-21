# vestigial-user

**Question.** Does post-training make a model treat the assistant slot differently from the user slot, and at which stage?
We look at one dimension, *vestigial regression* (Shiller 2026): the model losing the ability to write as the user.

**Measurement.** We take real user and assistant text and compute per-token NLL with the text placed in each role slot. We do this at each post-training stage of the OLMo 3 7B Instruct line:

| stage | checkpoint |
|---|---|
| base | `allenai/Olmo-3-1025-7B` |
| sft | `allenai/Olmo-3-7B-Instruct-SFT` |
| dpo | `allenai/Olmo-3-7B-Instruct-DPO` |
| instruct (RL) | `allenai/Olmo-3-7B-Instruct` |

`check_models.py` verifies IDs, licenses, sizes, and chat templates through the HF API and writes `docs/models.md`.

## Status

The code is in place, but `data/texts.jsonl`, `data/prefill.jsonl` and `docs/` are **not yet generated**. huggingface.co was unreachable from the build environment, so the model checks, data sampling and both test files are still pending. Run the first five commands under *Running on the GPU* to produce them. `source` in each row records which dataset the committed sample came from.

## Design

- **Data** (`prepare_data.py` → `data/texts.jsonl`, `data/prefill.jsonl`). We sample 300 English conversations and drop any flagged toxic or redacted. From each we keep the first user turn and the first assistant reply. Each text must be 20–300 tokens by the OLMo tokenizer and render cleanly (see below). The default dataset is `allenai/WildChat-1M`, which is gated and needs `HF_TOKEN` with its terms accepted. `--dataset oasst1` switches to `OpenAssistant/oasst1` (root prompt plus its top-ranked reply). Each row records its `source`. The prefill set is the first sentence of 20 sampled user turns.
- **Conditions** (`config.CONDITIONS`). Each text is scored alone as the first turn, so the role header is the only difference between slot conditions:

  | condition | text | slot |
  |---|---|---|
  | `user@user` | user | user |
  | `asst@asst` | assistant | assistant |
  | `user@asst` | user | assistant (swap) |
  | `asst@user` | assistant | user (swap) |
  | `user@plain`, `asst@plain` | either | none: plain prose after `<\|endoftext\|>` (control) |

  `--preceding-turn` (off by default; `config.PRECEDING_TURN`) puts one fixed neutral turn of the opposite role before the scored turn, since a lone assistant turn is itself unusual input. This adds a second difference between slots (the preceding turn's text), so treat it as a robustness check.
- **One template for every stage.** All stages, base included, are scored with the Instruct tokenizer and chat template, so token sequences are identical across stages. Base is the symmetric baseline. Caveat: the chat special tokens may be poorly trained in base.
- **Rendering** (`render.py`). The content is tokenized on its own. The full rendered sequence must equal `tokens(prefix) + tokens(content) + tokens(suffix)`, which rules out merges at the header boundary. As a result, the content tokens are identical across all conditions, the control included. Decoding the span must reproduce the text exactly. Texts that fail these checks are filtered out at sampling time. `python render.py` dumps examples to `docs/rendered_examples.md`.
- **Scoring** (`score.py`). Batched, right-padded per-token NLL over content tokens only. Runs in bf16 on GPU and fp32 on CPU. Writes `results/scores/<stage>.csv` with columns `text_id, condition, n_tokens, mean_nll`.
- **Generation** (`generate.py`). Opens a user turn with each prefill sentence, leaves it unterminated, and samples 3 × 120 tokens at T=0.7. It counts four flags per sample: `ended_turn` (closed the user turn), `handoff_to_assistant` (wrote the assistant header), `assistant_register` (assistant phrases such as "I'd be happy to"), and `repetition_loop` (any word 4-gram repeated 3 or more times).
- **Analysis** (`analyze.py`). The slot effect is NLL(user slot) − NLL(assistant slot), computed per text type:
  - user text: `user@user − user@asst`
  - assistant text: `asst@user − asst@asst`

  95% bootstrap CIs are taken over texts (paired). Writes `results/nll_by_stage.png` (stage on x, one line per condition), `results/results.md`, and `results/summary.csv`.

## Running on the GPU (single 80 GB card)

```bash
uv sync                                   # lockfile: uv.lock
export HF_TOKEN=...                       # WildChat terms accepted on this account
uv run python check_models.py             # IDs, licenses, shared chat template -> docs/models.md
uv run python prepare_data.py             # only if data/texts.jsonl is not committed yet
uv run python render.py                   # -> docs/rendered_examples.md; eyeball it
uv run pytest                             # span alignment (OLMo tokenizer) + CPU smoke test (SmolLM2-135M)
bash run_gpu.sh                           # score + generate per checkpoint, one process each, then analyze
OUT=results_preceding bash run_gpu.sh --preceding-turn   # optional robustness run
```

`run_gpu.sh` runs each checkpoint in its own process, so GPU memory is released between them. The checkpoints are roughly 15 GB each in bf16; set `CLEAR_CACHE=1` to delete each one from the HF cache after use. Scoring 300 texts × 6 conditions takes a few minutes per model.

## What the result pattern would mean

- **Base:** both slot effects should be near zero, since base has no learned role asymmetry. This is the reference point.
- **Assistant text:** if post-training is learning the assistant role, `asst@asst` should fall and the assistant-text slot effect should become clearly positive.
- **User text (the vestigial-regression signal):** if the model still models users in the user slot, the user-text slot effect should be negative (`user@user < user@asst`). Vestigial regression would show up in three ways:
  - `user@user` rises across stages relative to base and relative to `user@plain`.
  - The user-text slot effect shrinks to zero or flips positive. A flip means user text is *more* expected in the assistant slot, so the model treats every slot as its own.
  - Prefilled user turns increasingly end early, hand off to the assistant, or slip into assistant register.
- **Which stage:** the stage where the change first appears is the answer. SFT is the prime suspect because its loss masks user tokens, so user modeling drifts untrained. If the change appears at DPO or RL instead, preference/RL optimization is doing it.
- **Controls:** if `*@plain` moves as much as the slot conditions, the change is general distribution drift, not role-specific. The role-specific part is `user@user − user@plain`.
