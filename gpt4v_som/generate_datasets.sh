#!/usr/bin/env bash
# Generate all SFT/RL datasets from VisualWebArena render trajectories.
# Run from the repository root:
#   bash gpt4v_som/generate_datasets.sh
#
# Conventions:
#   - idx < 100  → eval slice  (first 100 examples reserved for evaluation)
#   - idx >= 100 → training slice
#   - sft_*      → PASS-only trajectories (supervised fine-tuning)
#   - rl_*       → PASS + FAIL trajectories (reward-learning / RL)
#   - *_image_ignored → screenshot replaced with 1×1 PNG placeholder
#   - *_noex     → no few-shot examples in prompt (cleaner loss signal)

set -euo pipefail

SCRIPT="gpt4v_som/create_sft_dataset.py"
OUT="gpt4v_som/dataset"

echo "=== sft_training (PASS, idx >= 100, full images) ==="
python3 "$SCRIPT" \
  --min-render-id 100 \
  --success_only \
  -o "$OUT/sft_training.jsonl"

echo "=== sft_training_eval (PASS, idx < 100, full images) ==="
python3 "$SCRIPT" \
  --min-render-id 0 --max-render-id 100 \
  --success_only \
  -o "$OUT/sft_training_eval.jsonl"

echo "=== sft_training_image_ignored (PASS, idx >= 100, images replaced) ==="
python3 "$SCRIPT" \
  --min-render-id 100 \
  --success_only \
  --ignore-images \
  -o "$OUT/sft_training_image_ignored.jsonl"

echo "=== sft_training_image_ignored_eval (PASS, idx < 100, images replaced) ==="
python3 "$SCRIPT" \
  --min-render-id 0 --max-render-id 100 \
  --success_only \
  --ignore-images \
  -o "$OUT/sft_training_image_ignored_eval.jsonl"

echo "=== rl_training (PASS+FAIL, idx >= 100, full images) ==="
python3 "$SCRIPT" \
  --min-render-id 100 \
  -o "$OUT/rl_training.jsonl"

echo "=== rl_training_eval (PASS+FAIL, idx < 100, full images) ==="
python3 "$SCRIPT" \
  --min-render-id 0 --max-render-id 100 \
  -o "$OUT/rl_training_eval.jsonl"

echo "=== rl_training_image_ignored (PASS+FAIL, idx >= 100, images replaced) ==="
python3 "$SCRIPT" \
  --min-render-id 100 \
  --ignore-images \
  -o "$OUT/rl_training_image_ignored.jsonl"

echo "=== rl_training_image_ignored_eval (PASS+FAIL, idx < 100, images replaced) ==="
python3 "$SCRIPT" \
  --min-render-id 0 --max-render-id 100 \
  --ignore-images \
  -o "$OUT/rl_training_image_ignored_eval.jsonl"

echo "=== sft_training_noex (PASS, idx >= 100, full images, no few-shot) ==="
python3 "$SCRIPT" \
  --min-render-id 100 \
  --success_only \
  --no-examples \
  -o "$OUT/sft_training_noex.jsonl"

echo "=== sft_training_eval_noex (PASS, idx < 100, full images, no few-shot) ==="
python3 "$SCRIPT" \
  --min-render-id 0 --max-render-id 100 \
  --success_only \
  --no-examples \
  -o "$OUT/sft_training_eval_noex.jsonl"

echo "=== rl_training_noex (PASS+FAIL, idx >= 100, full images, no few-shot) ==="
python3 "$SCRIPT" \
  --min-render-id 100 \
  --no-examples \
  -o "$OUT/rl_training_noex.jsonl"

echo "=== rl_training_eval_noex (PASS+FAIL, idx < 100, full images, no few-shot) ==="
python3 "$SCRIPT" \
  --min-render-id 0 --max-render-id 100 \
  --no-examples \
  -o "$OUT/rl_training_eval_noex.jsonl"

echo ""
echo "Done. Datasets written to $OUT/"
ls -lh "$OUT/"*.jsonl 2>/dev/null || true
