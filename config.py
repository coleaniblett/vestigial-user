"""Shared settings. CLI flags on each script override these."""

# OLMo 3 7B Instruct line, in post-training order.
MODELS = {
    "base": "allenai/Olmo-3-1025-7B",
    "sft": "allenai/Olmo-3-7B-Instruct-SFT",
    "dpo": "allenai/Olmo-3-7B-Instruct-DPO",
    "instruct": "allenai/Olmo-3-7B-Instruct",
}
STAGES = list(MODELS)

# Every stage (base included) is scored with this tokenizer and chat template,
# so token sequences are identical across stages.
TEMPLATE_MODEL = "allenai/Olmo-3-7B-Instruct"

# Text sources. Each row of data/texts.jsonl is one text with a source and a kind
# ("user" or "assistant"); sources are analyzed separately and never pooled.
#   wildchat       first user turn + first assistant reply of WildChat conversations
#   oasst1         oasst1 root prompts (user text only)
#   olmo-instruct  OLMo 3 7B Instruct's greedy reply to each WildChat user turn (add_replies.py)
# The pair source can be switched to "oasst1" (prompt + top-ranked reply), as the smoke test does.
PAIR_SOURCE = "wildchat"
EXTRA_USER_SOURCE = "oasst1"
DATASET_IDS = {"wildchat": "allenai/WildChat-1M", "oasst1": "OpenAssistant/oasst1"}
# Drop any conversation whose user turn appears in the Dolci-Instruct SFT/DPO/RL data.
EXCLUDE_DOLCI = True

N_TEXTS = 300  # per source
MIN_TOKENS, MAX_TOKENS = 20, 300
N_PREFILL = 20
SEED = 0

REPLY_SOURCE = "olmo-instruct"
REPLY_MODEL = "instruct"
REPLY_MAX_NEW_TOKENS = 300

DATA_PATH = "data/texts.jsonl"
PREFILL_PATH = "data/prefill.jsonl"

# Optional fixed neutral turn placed before the scored turn (off by default).
# When on, the scored turn follows a turn of the opposite role, so each slot
# sees ordinary alternation: a user-slot text follows NEUTRAL_TURN["assistant"],
# an assistant-slot text follows NEUTRAL_TURN["user"].
PRECEDING_TURN = False
NEUTRAL_TURN = {"user": "Hello.", "assistant": "Hello! How can I help you today?"}

# Every text is scored in every slot. Condition name = <kind>@<slot>, e.g. user@asst is user text
# in the assistant slot. slot None = plain prose after <|endoftext|> (control).
KIND_ABBR = {"user": "user", "assistant": "asst"}
SLOTS = [("user", "user"), ("asst", "assistant"), ("plain", None)]


def conditions(kind):
    """[(condition name, slot)] for a text of this kind."""
    return [(f"{KIND_ABBR[kind]}@{abbr}", slot) for abbr, slot in SLOTS]


GEN_MAX_NEW_TOKENS = 120
GEN_TEMPERATURE = 0.7
GEN_SAMPLES = 3
