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
#   - sft_*      → PASS-only, prompt+assistant (SFT format, create_sft_dataset.py)
#   - rl_*       → PASS+FAIL, prompt-only + ground_truth (RFT format, create_rl_dataset.py)
#   - *_image_resized → screenshots resized so longest side ≤ 720 px (LANCZOS)
#   - *_image_ignored → screenshot replaced with 1×1 PNG placeholder
#   - *_noex     → no few-shot examples in prompt (cleaner loss signal)
#                  (no suffix = few-shot examples included)

set -euo pipefail

SCRIPT="gpt4v_som/create_sft_dataset.py"
RL_SCRIPT="gpt4v_som/create_rl_dataset.py"
OUT="gpt4v_som/dataset"

# Pillow is required for --image-resize. Install it if missing.
python3 -c "from PIL import Image" 2>/dev/null \
  || pip3 install -q Pillow 2>/dev/null \
  || sudo apt-get install -y -q python3-pil

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
# RL + image_resized  (RFT format: prompt-only, ground_truth=action|||0/1)
# ---------------------------------------------------------------------------

echo "=== rl_image_resized (PASS+FAIL, idx >= 100, 720px, with examples) ==="
python3 "$RL_SCRIPT" \
  --min-render-id 100 \
  --image-resize 720 \
  -o "$OUT/rl_image_resized.jsonl"

echo "=== rl_image_resized_eval (PASS+FAIL, idx < 100, 720px, with examples) ==="
python3 "$RL_SCRIPT" \
  --min-render-id 0 --max-render-id 100 \
  --image-resize 720 \
  -o "$OUT/rl_image_resized_eval.jsonl"

echo "=== rl_image_resized_noex (PASS+FAIL, idx >= 100, 720px, no few-shot) ==="
python3 "$RL_SCRIPT" \
  --min-render-id 100 \
  --no-examples \
  --image-resize 720 \
  -o "$OUT/rl_image_resized_noex.jsonl"

echo "=== rl_image_resized_noex_eval (PASS+FAIL, idx < 100, 720px, no few-shot) ==="
python3 "$RL_SCRIPT" \
  --min-render-id 0 --max-render-id 100 \
  --no-examples \
  --image-resize 720 \
  -o "$OUT/rl_image_resized_noex_eval.jsonl"

# ---------------------------------------------------------------------------
# RL + image_ignored  (RFT format: prompt-only, ground_truth=action|||0/1)
# ---------------------------------------------------------------------------

echo "=== rl_image_ignored (PASS+FAIL, idx >= 100, images replaced, with examples) ==="
python3 "$RL_SCRIPT" \
  --min-render-id 100 \
  --ignore-images \
  -o "$OUT/rl_image_ignored.jsonl"

echo "=== rl_image_ignored_eval (PASS+FAIL, idx < 100, images replaced, with examples) ==="
python3 "$RL_SCRIPT" \
  --min-render-id 0 --max-render-id 100 \
  --ignore-images \
  -o "$OUT/rl_image_ignored_eval.jsonl"

echo "=== rl_image_ignored_noex (PASS+FAIL, idx >= 100, images replaced, no few-shot) ==="
python3 "$RL_SCRIPT" \
  --min-render-id 100 \
  --ignore-images \
  --no-examples \
  -o "$OUT/rl_image_ignored_noex.jsonl"

echo "=== rl_image_ignored_noex_eval (PASS+FAIL, idx < 100, images replaced, no few-shot) ==="
python3 "$RL_SCRIPT" \
  --min-render-id 0 --max-render-id 100 \
  --ignore-images \
  --no-examples \
  -o "$OUT/rl_image_ignored_noex_eval.jsonl"

echo ""
echo "Done. Datasets written to $OUT/"
ls -lh "$OUT/"*.jsonl 2>/dev/null || true
