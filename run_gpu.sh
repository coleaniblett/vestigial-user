#!/usr/bin/env bash
# Score and generate with each checkpoint in turn, one process per model so GPU memory
# is fully released between them. Extra args go to score.py and generate.py, e.g. --preceding-turn.
# The role-swap (data/swap.jsonl) and coin-toss (data/coin_toss.jsonl) scoring run only without --preceding-turn:
# they set their own conversational context.
#   bash run_gpu.sh
#   OUT=results_preceding bash run_gpu.sh --preceding-turn
set -euo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-results}
STAGES=${STAGES:-"base sft dpo instruct"}

SWAP=1
for a in "$@"; do [[ "$a" == --preceding-turn ]] && SWAP=0; done
[[ -f data/swap.jsonl && -f data/coin_toss.jsonl ]] || SWAP=0

for stage in $STAGES; do
  echo "=== $stage"
  uv run python score.py --model "$stage" --out "$OUT/scores" "$@"
  uv run python generate.py --model "$stage" --out "$OUT/generations" "$@"
  if [[ $SWAP == 1 ]]; then
    uv run python role_swap.py score --model "$stage" --out "$OUT/role_swap"
    uv run python coin_toss.py score --model "$stage" --out "$OUT/coin_toss"
  fi
  # weights are ~15 GB per checkpoint; set CLEAR_CACHE=1 to delete each after use
  if [[ "${CLEAR_CACHE:-0}" == 1 ]]; then
    uv run python -c "import config, shutil, huggingface_hub as h; \
      [shutil.rmtree(r.repo_path) for r in h.scan_cache_dir().repos if r.repo_id == config.MODELS['$stage'] and r.repo_id != config.TEMPLATE_MODEL]"
  fi
done
uv run python analyze.py --scores "$OUT/scores" --generations "$OUT/generations" --out "$OUT"
if [[ $SWAP == 1 ]]; then
  uv run python role_swap.py analyze --scores "$OUT/role_swap" --out-dir "$OUT"
  uv run python coin_toss.py analyze --scores "$OUT/coin_toss" --out-dir "$OUT"
fi
