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

`check_models.py` verifies IDs, licenses, sizes, chat templates and tokenizers through the HF API and writes `docs/models.md`. The template files of SFT/DPO and Instruct differ only in a tool-handling null check and render identically without tools; the four tokenizer.json files differ only in merge serialization and in the names of a few reserved tokens, so the check compares what they render and encode rather than file hashes.

## Design

- **Data** (`prepare_data.py` → `data/texts.jsonl`, `data/prefill.jsonl`; `add_replies.py`). Each row of `texts.jsonl` is one text with a `source` and a `kind` (user or assistant). Sources are analyzed separately and never pooled:

  | source | kind | n | what |
  |---|---|---|---|
  | `wildchat` | user, assistant | 300 pairs | first user turn and first assistant reply (ChatGPT) of English `allenai/WildChat-1M` conversations, toxic/redacted dropped |
  | `oasst1` | user | 300 | English `OpenAssistant/oasst1` root prompts, toxicity < 0.5 |
  | `olmo-instruct` | assistant | ≤ 300 | OLMo 3 7B Instruct's greedy reply to each `wildchat` user turn, cut at 300 new tokens; cut replies are kept and marked `truncated` (causal attention: their NLL is that of the reply's first 300 tokens) |

  **Held out from post-training.** OLMo 3's SFT data contains most of WildChat (86% of an unfiltered sample's user turns appear verbatim) and some oasst1, so `prepare_data.py` skips any conversation whose user turn appears in the Dolci-Instruct SFT, DPO or RL data (`check_contamination.py`: exact match after whitespace normalization; RL prompts are parsed out of their `user: ...` strings and also matched on WildChat's `conversation_hash`), then keeps sampling (seed 0) until 300 clean texts per source pass the other filters. Each text must be 20–300 tokens by the OLMo tokenizer and render cleanly (see below). `allenai/WildChat-1M` is no longer gated. Rows record `pair_id` (links a WildChat user turn, its reply and OLMo's reply) and `source_id`. The prefill set is the first sentence of 20 sampled WildChat user turns (distinct sentences; different conversations can share an opening). `--pairs oasst1 --extra-user none --no-exclude-dolci` gives the small oasst1-only set the smoke test uses.
- **Conditions** (`config.SLOTS`). Every text is scored in every slot; the condition name is `<kind>@<slot>`, e.g. `user@asst` is user text in the assistant slot. Each text is scored alone as the first turn, so the role header is the only difference between slot conditions before the scored span (the Instruct template also prepends a fixed default system turn, identical in every slot condition). After the span the template closes a user turn with `<|im_end|>\n` and a final assistant turn with `<|endoftext|>`; this suffix follows the scored tokens, so under causal attention it cannot change their NLL (checked in the smoke test).

  | slot | user text | assistant text |
  |---|---|---|
  | user | `user@user` | `asst@user` (swap) |
  | assistant | `user@asst` (swap) | `asst@asst` |
  | none: plain prose after `<\|endoftext\|>` (control) | `user@plain` | `asst@plain` |

  `--preceding-turn` (off by default; `config.PRECEDING_TURN`) puts one fixed neutral turn of the opposite role before the scored turn, since a lone assistant turn is itself unusual input. This adds a second difference between slots (the preceding turn's text), so treat it as a robustness check.
- **One template for every stage.** All stages, base included, are scored with the Instruct tokenizer and chat template, so token sequences are identical across stages. Base is the symmetric baseline. Caveat: the chat special tokens may be poorly trained in base.
- **Rendering** (`render.py`). The content is tokenized on its own. The full rendered sequence must equal `tokens(prefix) + tokens(content) + tokens(suffix)`, which rules out merges at the header boundary. As a result, the content tokens are identical across all conditions, the control included. Decoding the span must reproduce the text exactly. Texts that fail these checks are filtered out at sampling time. `python render.py` dumps examples to `docs/rendered_examples.md`.
- **Scoring** (`score.py`). Batched, right-padded per-token NLL over content tokens only. Runs in fp32 everywhere (a 7B model takes ~33 GB on GPU): in bf16, per-text NLL moves by up to 0.05 nats depending on batch composition, comparable to the effects of interest. Generation runs in bf16. Writes `results/scores/<stage>.csv` with columns `text_id, source, kind, condition, n_tokens, mean_nll, mean_nll_from1, mean_entropy, mean_entropy_from1` (`_from1`: without the first content token; entropy: of the next-token distribution at each content token), and the per-token values to `results/scores/<stage>_tokens.parquet` (`text_id, source, kind, condition, position, nll, entropy`; position 0 is the first content token).
- **Generation** (`generate.py`). Opens a user turn with each prefill sentence, leaves it unterminated, and samples 3 × 120 tokens at T=0.7. It counts four flags per sample: `ended_turn` (closed the user turn), `handoff_to_assistant` (then wrote the assistant header; a subset of `ended_turn`), `assistant_register` (assistant phrases such as "I'd be happy to") and `repetition_loop` (any word 4-gram repeated 3 or more times). The last two are checked on the user portion only, the text before the turn closes: a post-trained model usually finishes the user's message, closes the turn and answers as the assistant, and that reply is not the user's voice.
- **Analysis** (`analyze.py`). Per text set (source × kind) and stage, with 95% bootstrap CIs over texts (paired across slots):
  - raw mean NLL in each slot (user, assistant, plain);
  - slot effect = NLL(user slot) − NLL(assistant slot): `user@user − user@asst` for user text, `asst@user − asst@asst` for assistant text. By default over content positions 1+, with the all-tokens version alongside: when an assistant turn directly follows the system turn, the post-trained checkpoints put nearly all next-token probability on `<|im_start|>` (1.000 SFT, 0.960 DPO, 0.881 Instruct), so the first content token in the assistant slot costs 2–14 nats whatever the text;
  - role-specific = NLL(own slot) − NLL(plain): `user@user − user@plain` for user text, `asst@asst − asst@plain` for assistant text (all tokens);
  - mean next-token entropy per slot, over positions 1+ (default) and over all tokens.

  Writes `results/results.md`, `results/summary.csv` and one figure per text set, `results/nll_<source>_<kind>.png` (stage on x, one line per slot).

- **Position analysis** (`analyze.py`, from the per-token NLLs). Per text set and stage, over the texts with at least 40 content tokens (so every position averages over the same texts): mean NLL at each of the first 40 content-token positions in each slot, and the slot difference NLL(user slot) − NLL(assistant slot) by position with 95% bootstrap CIs. The slot difference counts as negligible from the first position p such that every per-position mean from p to 39 is within ±0.05 nats and their mean over positions p..39 has a 95% CI inside ±0.05 nats (`--positions`, `--margin`). Writes `results/position.md` and `results/position_<source>_<kind>.png`.
- **Role-swapped conversations** (`role_swap.py` → `data/swap.jsonl`, `results/role_swap/<stage>.csv`, `results/role_swap.md`). 300 held-out English WildChat conversations with at least three turns: user P1, assistant R1, user P2, each 20–300 tokens, skipped if P1, P2 or the conversation hash is in the Dolci data. Each is rendered under two labelings that keep every content token identical (checked as for the slot conditions): normal (user P1, assistant R1, user P2) and flipped (assistant P1, user R1, assistant P2), both with the default system turn. Only R1 and P2 are scored. A two-turn format (P1, R1 only) scores R1 from two sources on the same conversations: ChatGPT's original reply and OLMo 3 7B Instruct's greedy reply to P1 (`role_swap.py replies`; max 300 new tokens, cut replies kept and flagged), giving the source × label interaction (OLMo label effect − ChatGPT label effect). At the Instruct stage the OLMo replies are that checkpoint's own greedy output, so their normal-label NLL is low by construction. Mean next-token entropy of each scored turn is reported too. Label effect = NLL(flipped) − NLL(normal) per turn, 95% bootstrap CI over conversations; base measures what the flipped labeling's oddness costs on its own, so later stages are also reported relative to base, paired over conversations. The swap is its own context control, so `run_gpu.sh` skips it under `--preceding-turn`.

- **Coin-toss spillover** (`coin_toss.py` → `data/coin_toss.jsonl`, `results/coin_toss/<stage>.csv`, `results/coin_toss.md`). 40 three-turn conversations (20 items × heads/tails as the preferred outcome): user opener about settling something with a fair coin, neutral assistant reply, and a user turn ending "I flipped the coin and it landed on", left open. A first-person preference sentence with a reason sits in the assistant's reply, in the user's opener, or nowhere; nothing else differs. P(outcome) sums the probabilities of the case and spacing variants (`" heads"`, `"heads"`, `" Heads"`, `"Heads"`, `" HEADS"`, `"HEADS"`, same for tails) as token sequences after the prompt. Reports log-odds toward the preferred outcome per condition (the "none" mean is 0 by the pair design), assistant tilt − user tilt, and its change relative to base, with bootstrap CIs over conversations.

## Running on the GPU (single 48 GB+ card)

```bash
uv sync                                   # lockfile: uv.lock
uv run python check_models.py             # IDs, licenses, shared chat template -> docs/models.md
uv run python prepare_data.py             # optional: data/ is committed; rerun reproduces it (streams WildChat, loads Dolci)
uv run python add_replies.py              # optional: adds the olmo-instruct replies to data/texts.jsonl (GPU)
uv run python check_contamination.py      # overlap of data/texts.jsonl with the Dolci SFT/DPO/RL data
uv run python role_swap.py prepare        # optional: data/swap.jsonl is committed; rerun reproduces it
uv run python role_swap.py replies        # optional: adds OLMo Instruct's replies to data/swap.jsonl (GPU)
uv run python coin_toss.py build          # optional: data/coin_toss.jsonl is committed; rerun reproduces it
uv run python render.py                   # -> docs/rendered_examples.md; eyeball it
uv run pytest                             # span alignment (OLMo tokenizer) + CPU smoke test (SmolLM2-135M)
bash run_gpu.sh                           # score + generate + role swap + coin toss per checkpoint, then analyze
OUT=results_preceding bash run_gpu.sh --preceding-turn   # optional robustness run
```

`run_gpu.sh` runs each checkpoint in its own process, so GPU memory is released between them. The checkpoints are roughly 15 GB each in bf16; set `CLEAR_CACHE=1` to delete each one from the HF cache after use. Scoring ~1,200 texts × 3 slots takes about five minutes per model in fp32.

## What the result pattern would mean

- **Base:** both slot effects should be near zero, since base has no learned role asymmetry. This is the reference point.
- **Assistant text:** if post-training is learning the assistant role, `asst@asst` should fall and the assistant-text slot effect should become clearly positive.
- **User text (the vestigial-regression signal):** if the model still models users in the user slot, the user-text slot effect should be negative (`user@user < user@asst`). Vestigial regression would show up in three ways:
  - `user@user` rises across stages relative to base and relative to `user@plain`.
  - The user-text slot effect shrinks to zero or flips positive. A flip means user text is *more* expected in the assistant slot, so the model treats every slot as its own.
  - Prefilled user turns increasingly end early, hand off to the assistant, or slip into assistant register.
- **Which stage:** the stage where the change first appears is the answer. SFT is the prime suspect because its loss masks user tokens, so user modeling drifts untrained. If the change appears at DPO or RL instead, preference/RL optimization is doing it.
- **Controls:** if `*@plain` moves as much as the slot conditions, the change is general distribution drift, not role-specific. The role-specific part is `user@user − user@plain`.
