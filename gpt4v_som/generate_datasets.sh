#!/usr/bin/env bash
# Generate all SFT/RL datasets from VisualWebArena render trajectories.
# Run from the repository root:
#   bash gpt4v_som/generate_datasets.sh
#
# 16 files = {sft,rl} × {image_resized,image_ignored} × {noex,ex} × {train,eval}
#
# Conventions:
#   - idx < 100  → eval slice  (_eval suffix, last)
#   - idx >= 100 → training slice
#   - sft_*      → PASS-only trajectories (supervised fine-tuning)
#   - rl_*       → PASS + FAIL trajectories (reward-learning / RL)
#   - *_image_resized → screenshots resized so longest side ≤ 720 px (LANCZOS)
#   - *_image_ignored → screenshot replaced with 1×1 PNG placeholder
#   - *_noex     → no few-shot examples in prompt (cleaner loss signal)
#                  (no suffix = few-shot examples included)

set -euo pipefail

SCRIPT="gpt4v_som/create_sft_dataset.py"
OUT="gpt4v_som/dataset"

# ---------------------------------------------------------------------------
# SFT + image_resized
# ---------------------------------------------------------------------------

echo "=== sft_image_resized (PASS, idx >= 100, 720px, with examples) ==="
python3 "$SCRIPT" \
  --min-render-id 100 \
  --success_only \
  --image-resize 720 \
  -o "$OUT/sft_image_resized.jsonl"

echo "=== sft_image_resized_eval (PASS, idx < 100, 720px, with examples) ==="
python3 "$SCRIPT" \
  --min-render-id 0 --max-render-id 100 \
  --success_only \
  --image-resize 720 \
  -o "$OUT/sft_image_resized_eval.jsonl"

echo "=== sft_image_resized_noex (PASS, idx >= 100, 720px, no few-shot) ==="
python3 "$SCRIPT" \
  --min-render-id 100 \
  --success_only \
  --no-examples \
  --image-resize 720 \
  -o "$OUT/sft_image_resized_noex.jsonl"

echo "=== sft_image_resized_noex_eval (PASS, idx < 100, 720px, no few-shot) ==="
python3 "$SCRIPT" \
  --min-render-id 0 --max-render-id 100 \
  --success_only \
  --no-examples \
  --image-resize 720 \
  -o "$OUT/sft_image_resized_noex_eval.jsonl"

# ---------------------------------------------------------------------------
# SFT + image_ignored
# ---------------------------------------------------------------------------

echo "=== sft_image_ignored (PASS, idx >= 100, images replaced, with examples) ==="
python3 "$SCRIPT" \
  --min-render-id 100 \
  --success_only \
  --ignore-images \
  -o "$OUT/sft_image_ignored.jsonl"

echo "=== sft_image_ignored_eval (PASS, idx < 100, images replaced, with examples) ==="
python3 "$SCRIPT" \
  --min-render-id 0 --max-render-id 100 \
  --success_only \
  --ignore-images \
  -o "$OUT/sft_image_ignored_eval.jsonl"

echo "=== sft_image_ignored_noex (PASS, idx >= 100, images replaced, no few-shot) ==="
python3 "$SCRIPT" \
  --min-render-id 100 \
  --success_only \
  --ignore-images \
  --no-examples \
  -o "$OUT/sft_image_ignored_noex.jsonl"

echo "=== sft_image_ignored_noex_eval (PASS, idx < 100, images replaced, no few-shot) ==="
python3 "$SCRIPT" \
  --min-render-id 0 --max-render-id 100 \
  --success_only \
  --ignore-images \
  --no-examples \
  -o "$OUT/sft_image_ignored_noex_eval.jsonl"

# ---------------------------------------------------------------------------
# RL + image_resized
# ---------------------------------------------------------------------------

echo "=== rl_image_resized (PASS+FAIL, idx >= 100, 720px, with examples) ==="
python3 "$SCRIPT" \
  --min-render-id 100 \
  --image-resize 720 \
  -o "$OUT/rl_image_resized.jsonl"

echo "=== rl_image_resized_eval (PASS+FAIL, idx < 100, 720px, with examples) ==="
python3 "$SCRIPT" \
  --min-render-id 0 --max-render-id 100 \
  --image-resize 720 \
  -o "$OUT/rl_image_resized_eval.jsonl"

echo "=== rl_image_resized_noex (PASS+FAIL, idx >= 100, 720px, no few-shot) ==="
python3 "$SCRIPT" \
  --min-render-id 100 \
  --no-examples \
  --image-resize 720 \
  -o "$OUT/rl_image_resized_noex.jsonl"

echo "=== rl_image_resized_noex_eval (PASS+FAIL, idx < 100, 720px, no few-shot) ==="
python3 "$SCRIPT" \
  --min-render-id 0 --max-render-id 100 \
  --no-examples \
  --image-resize 720 \
  -o "$OUT/rl_image_resized_noex_eval.jsonl"

# ---------------------------------------------------------------------------
# RL + image_ignored
# ---------------------------------------------------------------------------

echo "=== rl_image_ignored (PASS+FAIL, idx >= 100, images replaced, with examples) ==="
python3 "$SCRIPT" \
  --min-render-id 100 \
  --ignore-images \
  -o "$OUT/rl_image_ignored.jsonl"

echo "=== rl_image_ignored_eval (PASS+FAIL, idx < 100, images replaced, with examples) ==="
python3 "$SCRIPT" \
  --min-render-id 0 --max-render-id 100 \
  --ignore-images \
  -o "$OUT/rl_image_ignored_eval.jsonl"

echo "=== rl_image_ignored_noex (PASS+FAIL, idx >= 100, images replaced, no few-shot) ==="
python3 "$SCRIPT" \
  --min-render-id 100 \
  --ignore-images \
  --no-examples \
  -o "$OUT/rl_image_ignored_noex.jsonl"

echo "=== rl_image_ignored_noex_eval (PASS+FAIL, idx < 100, images replaced, no few-shot) ==="
python3 "$SCRIPT" \
  --min-render-id 0 --max-render-id 100 \
  --ignore-images \
  --no-examples \
  -o "$OUT/rl_image_ignored_noex_eval.jsonl"

echo ""
echo "Done. Datasets written to $OUT/"
ls -lh "$OUT/"*.jsonl 2>/dev/null || true
