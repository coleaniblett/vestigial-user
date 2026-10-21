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

# Dataset switch: "wildchat" (default, gated: needs HF_TOKEN with terms accepted)
# or "oasst1" (open stand-in).
DATASET = "wildchat"
DATASET_IDS = {"wildchat": "allenai/WildChat-1M", "oasst1": "OpenAssistant/oasst1"}

N_TEXTS = 300
MIN_TOKENS, MAX_TOKENS = 20, 300
N_PREFILL = 20
SEED = 0

DATA_PATH = "data/texts.jsonl"
PREFILL_PATH = "data/prefill.jsonl"

# Optional fixed neutral turn placed before the scored turn (off by default).
# When on, the scored turn follows a turn of the opposite role, so each slot
# sees ordinary alternation: a user-slot text follows NEUTRAL_TURN["assistant"],
# an assistant-slot text follows NEUTRAL_TURN["user"].
PRECEDING_TURN = False
NEUTRAL_TURN = {"user": "Hello.", "assistant": "Hello! How can I help you today?"}

# Scoring conditions: (condition, which text, which slot). slot None = plain prose.
CONDITIONS = [
    ("user@user", "user", "user"),
    ("asst@asst", "assistant", "assistant"),
    ("user@asst", "user", "assistant"),
    ("asst@user", "assistant", "user"),
    ("user@plain", "user", None),
    ("asst@plain", "assistant", None),
]

GEN_MAX_NEW_TOKENS = 120
GEN_TEMPERATURE = 0.7
GEN_SAMPLES = 3
