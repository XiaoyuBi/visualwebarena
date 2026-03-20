# Running VisualWebArena with Hyperbolic (Qwen2.5-VL)

Set up [Hyperbolic](https://hyperbolic.xyz) and run the Qwen accessibility-tree evaluation.

---

## 1. Get a Hyperbolic API key

1. Sign up at **[https://hyperbolic.xyz](https://hyperbolic.xyz)**.
2. In the dashboard, create or copy an **API key**.

---

## 2. Environment variables

```bash
export OPENAI_BASE_URL="https://api.hyperbolic.xyz/v1"
export OPENAI_API_KEY="<your_hyperbolic_api_key>"
```

---

## 3. Run a single task (run.py)

```bash
export OPENAI_BASE_URL="https://api.hyperbolic.xyz/v1"
export OPENAI_API_KEY="<your_hyperbolic_api_key>"

python run.py \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_3s.json \
  --test_start_idx 0 \
  --test_end_idx 1 \
  --result_dir results_qwen_hyperbolic \
  --test_config_base_dir=config_files/vwa/test_classifieds \
  --model "Qwen/Qwen2.5-VL-7B-Instruct" \
  --observation_type accessibility_tree
```

---

## 4. Run top-100 script: `qwen_accessibility_tree_top100.sh`

Runs the first 100 tasks on Classifieds, Shopping, and Reddit. Default model is **`Qwen/Qwen2.5-VL-7B-Instruct`**.

**Default run:**

```bash
export OPENAI_BASE_URL="https://api.hyperbolic.xyz/v1"
export OPENAI_API_KEY="<your_hyperbolic_api_key>"

bash scripts/qwen_accessibility_tree_top100.sh
```

**Add a result_dir suffix** (e.g. to avoid overwriting):

```bash
bash scripts/qwen_accessibility_tree_top100.sh _hyperbolic
```

Result dirs: `results_qwen_actree_classifieds_top100_hyperbolic`, `results_qwen_actree_shopping_top100_hyperbolic`, `results_qwen_actree_reddit_top100_hyperbolic`.

**Override model:**

```bash
export VWA_MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
bash scripts/qwen_accessibility_tree_top100.sh _72b
```
