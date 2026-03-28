# Running Qwen2.5-VL-7B with Image + Caps + SoM on VisualWebArena

## Context

Run Qwen2.5-VL-7B-Instruct on VisualWebArena using **Image + Caps + SoM** observation mode via an **OpenAI-compatible API** (e.g. **Hyperbolic**). Instance 1 (web environment) is already running. Each of the 3 websites (Classifieds, Shopping, Reddit) runs the first 100 tasks.

You can either:

- **`qwen_som_top100.sh`** — repeat all 3 sites **3 times** for run-to-run variance, default prompt `p_som_cot_id_actree_3s.json`.
- **`qwen_som_top100_trajectory_hints_once.sh`** — **one pass per site** (no multi-run loop), prompt **`p_som_cot_id_actree_3s_trajectory_hints.json`** (extra rules 6–10: invalid actions, URL revisits, loops, premature stop, stop when satisfied).

---

## Step 1: Create AWS Instance 2 (Evaluation Server)

- **AMI**: Ubuntu 22.04 LTS (Canonical) — avoid 24.04
- **Instance type**: `r6i.xlarge` (32 GiB RAM, 4 vCPU) — CPU-only sufficient (Qwen on API, BLIP-2 on CPU)
- **Storage**: 50+ GiB (gp3)
- **Security group**: Allow SSH (22)

---

## Step 2: Install Environment

```bash
ssh -i /path/to/key.pem ubuntu@<instance2-public-dns>

# System packages + Python 3.11
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Playwright system dependencies
sudo apt install -y \
  libatk1.0-0 libatk-bridge2.0-0 libcups2 libatspi2.0-0 \
  libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
  libpango-1.0-0 libasound2 libnss3 libnspr4 libdrm2 libxkbcommon0

# Clone repo
cd ~
git clone <your-repo-url> visualwebarena
cd visualwebarena
# Optional: git checkout <branch-or-tag>   # use the branch you run experiments on

# Python venv + deps
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install "transformers>=4.36.0" "tokenizers>=0.15.0"
pip install "datasets>=2.18.0" "torch>=2.4.0"

# Playwright + project install
playwright install chromium
playwright install-deps
pip install -e .
python -c "import nltk; nltk.download('punkt')"
```

---

## Step 3: Set Environment Variables

```bash
HOSTNAME="<Instance-1-Public-IP>"   # your running web environment
export DATASET=visualwebarena
export CLASSIFIEDS="http://${HOSTNAME}:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
export SHOPPING="http://${HOSTNAME}:7770"
export REDDIT="http://${HOSTNAME}:9999"
export WIKIPEDIA="http://${HOSTNAME}:8888"
export HOMEPAGE="http://${HOSTNAME}:4399"

# LLM API (agent rollout), e.g. Hyperbolic
export OPENAI_BASE_URL="https://api.hyperbolic.xyz/v1"
export OPENAI_API_KEY="<your-hyperbolic-api-key>"

# Optional: string_match LLM grading uses GPT-4 via OpenAI official API when set
# (see llms/providers/openai_utils.py eval_client). If unset, eval shares OPENAI_*.
# export EVAL_OPENAI_API_KEY="sk-..."
```

---

## Step 4: Generate Test Configs and Cookies

```bash
cd ~/visualwebarena
source venv/bin/activate
python scripts/generate_test_data.py
bash prepare.sh

# Verify connectivity
curl -I http://${HOSTNAME}:9980
curl -I http://${HOSTNAME}:7770
curl -I http://${HOSTNAME}:9999
```

---

## Step 5: Run Shell Scripts

Both scripts pass **`--viewport_height 2048 --max_obs_length 3840 --max_images 4`** to `run.py` (same as in-repo defaults for SoM runs).

### A. Three runs per site (variance) — default prompt

**Script**: `scripts/qwen_som_top100.sh`

Runs Image + Caps + SoM for tasks 0–99 on all 3 sites, **3 full passes** (run 1/2/3). Instruction: `agent/prompts/jsons/p_som_cot_id_actree_3s.json`.

```bash
export OPENAI_BASE_URL="https://api.hyperbolic.xyz/v1"
export OPENAI_API_KEY="<your-hyperbolic-api-key>"

bash scripts/qwen_som_top100.sh              # default
bash scripts/qwen_som_top100.sh _hyperbolic   # optional result_dir suffix

# Override model:
export VWA_MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
bash scripts/qwen_som_top100.sh _72b
```

### B. One run per site — trajectory-hints prompt

**Script**: `scripts/qwen_som_top100_trajectory_hints_once.sh`

Same task range and flags, but **each site runs once** (no outer `for run in 1 2 3` loop). Instruction: `agent/prompts/jsons/p_som_cot_id_actree_3s_trajectory_hints.json`.

```bash
bash scripts/qwen_som_top100_trajectory_hints_once.sh
bash scripts/qwen_som_top100_trajectory_hints_once.sh _hints_run1
```

Result directories:

- `results_qwen_som_classifieds_top100_trajectory_hints<suffix>`
- `results_qwen_som_shopping_top100_trajectory_hints<suffix>`
- `results_qwen_som_reddit_top100_trajectory_hints<suffix>`

---

## Step 6: Check Results

Each `result_dir` contains:

- `results.csv` — per-task scores (`task_id`, `score`, `trajectory_length`, difficulties)
- `render_*.html` — visual trajectory replays

### Mean score (trajectory-hints single-run script)

```python
import pandas as pd

for site in ["classifieds", "shopping", "reddit"]:
    df = pd.read_csv(f"results_qwen_som_{site}_top100_trajectory_hints/results.csv")
    print(f"{site}: mean score = {df['score'].mean():.4f}, n = {len(df)}")
```

Adjust the folder name if you used a suffix (e.g. `..._trajectory_hints_hints_run1`).

### Mean and variance (`qwen_som_top100.sh`, 3 runs)

```python
import pandas as pd
import numpy as np

for site in ["classifieds", "shopping", "reddit"]:
    scores = []
    for run in [1, 2, 3]:
        df = pd.read_csv(f"results_qwen_som_{site}_top100_run{run}/results.csv")
        scores.append(df["score"].mean())
    print(f"{site}: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}, runs={scores}")
```

---

## Implementation Notes

**Scripts:**

1. `scripts/qwen_som_top100.sh` — 3× passes for variance; prompt `p_som_cot_id_actree_3s.json`.
2. `scripts/qwen_som_top100_trajectory_hints_once.sh` — one pass per site; prompt `p_som_cot_id_actree_3s_trajectory_hints.json`.

**Behavior (no local code changes required for a standard run):**

- `image_som` observation with BLIP-2 captioning loaded in `run.py` via `image_utils.get_captioning_fn` and passed into `construct_agent`.
- Qwen2.5-VL treated as multimodal when the model name contains `qwen` and `vl` and the prompt uses `MultimodalCoTPromptConstructor` (`agent/agent.py`).
- SoM prompts: `agent/prompts/jsons/p_som_cot_id_actree_3s.json` (baseline); `p_som_cot_id_actree_3s_trajectory_hints.json` (extra rules in `intro`).
- OpenAI-compatible client: `OPENAI_API_KEY` + optional `OPENAI_BASE_URL` in `llms/providers/openai_utils.py`; optional `EVAL_OPENAI_API_KEY` for GPT-4 string eval.

---

## Verification

After a run:

1. Each expected `results.csv` exists and has 100 rows (tasks 0–99).
2. For **`qwen_som_top100.sh`**, scores often differ across runs (default `temperature=1.0` in `run.py`).
3. `render_*.html` shows SoM-style annotated screenshots when observation type is `image_som`.
