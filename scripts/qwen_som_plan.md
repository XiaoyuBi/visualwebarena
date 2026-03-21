# Running Qwen2.5-VL-7B with Image + Caps + SoM on VisualWebArena

## Context

Run Qwen2.5-VL-7B-Instruct on VisualWebArena using **Image + Caps + SoM** observation mode via the **Hyperbolic API**. Instance 1 (web environment) is already running. Each of the 3 websites (Classifieds, Shopping, Reddit) runs the first 100 tasks, repeated 3 times for variance.

---

## Step 1: Create AWS Instance 2 (Evaluation Server)

- **AMI**: Ubuntu 22.04 LTS (Canonical) — avoid 24.04
- **Instance type**: `r6i.xlarge` (32 GiB RAM, 4 vCPU) — CPU-only sufficient (Qwen on Hyperbolic, BLIP-2 on CPU)
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
git checkout qwen_1.1

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

# Hyperbolic API
export OPENAI_BASE_URL="https://api.hyperbolic.xyz/v1"
export OPENAI_API_KEY="<your-hyperbolic-api-key>"
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

## Step 5: Create and Run Shell Script

**Script**: `scripts/qwen_som_top100.sh`

This script runs Image + Caps + SoM for the first 100 tasks on all 3 websites, with 3 runs each. Based on the existing pattern in `scripts/qwen_accessibility_tree_top100.sh` and `scripts/run_classifieds_som.sh`.

**Usage:**
```bash
export OPENAI_BASE_URL="https://api.hyperbolic.xyz/v1"
export OPENAI_API_KEY="<your-hyperbolic-api-key>"

bash scripts/qwen_som_top100.sh              # default
bash scripts/qwen_som_top100.sh _hyperbolic  # with suffix

# Override model:
export VWA_MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
bash scripts/qwen_som_top100.sh _72b
```

---

## Step 6: Check Results and Compute Variance

Each `result_dir` contains:
- `results.csv` — per-task scores (task_id, score, trajectory_length, difficulties)
- `render_*.html` — visual trajectory replays

Compute mean and variance:

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

**Files created:**
1. `scripts/qwen_som_top100.sh` — the shell script

**No code modifications needed** — all required functionality already exists:
- `image_som` observation type with BLIP-2 captioning (`run.py:266-276`)
- Qwen VL detection as vision model (`agent/agent.py:119-127`)
- `MultimodalCoTPromptConstructor` for SoM prompts (`agent/prompts/jsons/p_som_cot_id_actree_3s.json`)
- OpenAI-compatible API with custom base URL (`llms/providers/openai_utils.py:15-16`)

## Verification

After the script runs, verify:
1. Each `results_qwen_som_*_run{1,2,3}/results.csv` exists and has 100 rows
2. Scores vary between runs (temperature=1.0 ensures this)
3. Visual trajectories in `render_*.html` show annotated screenshots with SoM bounding boxes
