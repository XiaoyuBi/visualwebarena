# VisualWebArena: AWS Instance Setup & Run Guide

This guide covers creating an AWS EC2 instance, installing the environment, and running VisualWebArena evaluation (e.g. `run.py` with BLIP-2).

---

## 1. AWS Instance Creation

### 1.1 Choose an AMI

- **Recommended**: **Ubuntu 22.04 LTS** (official Canonical image).
  - In EC2 → Launch Instance → **Choose AMI**.
  - Search for **"Ubuntu 22.04 LTS"** and select **"Ubuntu 22.04 LTS - Jammy"** (Provider: **Canonical Group Limited**).
  - Avoid Ubuntu 24.04 for this setup; package names differ and Playwright `install-deps` may fail.
- If you use a Marketplace AMI, complete the **Subscribe** step (Canonical’s Ubuntu is $0 for the image).

### 1.2 Instance Type

- **For CPU-only (BLIP-2 on CPU)**: **r6i.xlarge** (32 GiB RAM, 4 vCPU). Enough memory for BLIP-2 + browser.
- **For GPU (faster BLIP-2)**: **g4dn.xlarge** or **g5.xlarge** (NVIDIA GPU). Use an AMI with NVIDIA drivers/CUDA if you choose GPU.

### 1.3 Storage

- Set root volume to at least **50 GiB** (gp3) to hold OS, repo, venv, and BLIP-2 weights (~15–20 GB).

### 1.4 Key Pair & Security Group

- Attach a **key pair** (e.g. `pem_key.pem`) and download the `.pem` file locally.
- **Security group**: allow SSH (22) from your IP. If you run websites (Classifieds, Shopping, etc.) on the same instance, also allow inbound ports **4399, 7770, 9980, 9999, 8888** as needed.

### 1.5 Launch and Connect

- Launch the instance, then connect:
  ```bash
  ssh -i /path/to/pem_key.pem ubuntu@<instance-public-DNS>
  ```
- Optional: add the host to `~/.ssh/config` for easier access (see [Cursor Remote-SSH](#optional-cursor-remote-ssh) below).

---

## 2. Environment Installation (on the Instance)

All commands below assume you are SSH’d into the instance as `ubuntu`.

### 2.1 System Packages and Python 3.11

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

### 2.2 Playwright System Dependencies (Ubuntu 22.04)

```bash
sudo apt install -y \
  libatk1.0-0 libatk-bridge2.0-0 libcups2 libatspi2.0-0 \
  libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
  libpango-1.0-0 libasound2 libnss3 libnspr4 libdrm2 libxkbcommon0
```

### 2.3 Clone Repo and Create venv

```bash
cd ~
git clone https://github.com/<your-username>/visualwebarena.git
cd visualwebarena
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 2.4 Fix BLIP-2 / Dependencies (transformers, tokenizers, torch, datasets)

```bash
pip install "transformers>=4.36.0" "tokenizers>=0.15.0"
pip install "datasets>=2.18.0" "torch>=2.4.0"
```

### 2.5 Playwright Browser and Project Install

```bash
playwright install chromium
playwright install-deps
pip install -e .
```

---

## 3. Configuration and Cookies

### 3.1 Environment Variables

Set URLs to your websites. If **websites run on the same instance**, use `localhost`:

```bash
HOSTNAME="localhost"  # Set your instance or server hostname here
export DATASET=visualwebarena
export CLASSIFIEDS="http://${HOSTNAME}:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
export SHOPPING="http://${HOSTNAME}:7770"
export REDDIT="http://${HOSTNAME}:9999"
export WIKIPEDIA="http://${HOSTNAME}:8888"
export HOMEPAGE="http://${HOSTNAME}:4399"
export OPENAI_API_KEY=sk-<your-openai-key>
```

If websites run on **another host**, replace `localhost` with that host’s public DNS or IP (e.g. `http://ec2-xx-xx-xx-xx.compute.amazonaws.com:9980`).

Optional: append these to `~/.bashrc` and run `source ~/.bashrc` so they persist across sessions.

### 3.2 Generate Config and Login Cookies

```bash
cd ~/visualwebarena
source venv/bin/activate
python scripts/generate_test_data.py
bash prepare.sh
```

Ensure the instance can reach the URLs above (e.g. Docker sites are up if using localhost).

---

## 4. Run Test

### 4.1 Full Evaluation (one Classifieds task)

```bash
cd ~/visualwebarena
source venv/bin/activate
# Set env vars if not in .bashrc (see 3.1)

python run.py \
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json \
  --test_start_idx 0 \
  --test_end_idx 1 \
  --result_dir results \
  --test_config_base_dir=config_files/vwa/test_classifieds \
  --model gpt-4o \
  --action_set_tag som \
  --observation_type image_som
```

- Trajectory and logs: `results/`, e.g. `results/render_0.html`, `results/config.json`.
- Success/failure is reported in logs (e.g. `[Result] (PASS)` or `(FAIL)`).

### 4.2 Demo (no website backend required)

```bash
python run_demo.py \
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json \
  --start_url "https://www.example.com" \
  --intent "Click the more information link." \
  --result_dir demo_test \
  --model gpt-4o \
  --action_set_tag som \
  --observation_type image_som
```

Output is under `demo_test/` (e.g. `demo_test/render_0.html`). No automatic PASS/FAIL; check the trajectory manually.

---

## 5. Optional: Cursor Remote-SSH

To open the AWS repo inside Cursor over SSH:

1. On your machine, edit `~/.ssh/config`:
   ```
   Host <your-host-alias>
       HostName <instance-public-DNS>
       User ubuntu
       IdentityFile "/path/to/your/key.pem"
   ```
   Replace `<instance-public-DNS>` with your instance’s public DNS (e.g. from EC2 console). Use the full path to your `.pem` file. **Quote the path if it contains spaces.**

2. In Cursor: **Cmd+Shift+P** → **Remote-SSH: Connect to Host...** → choose the host alias you defined.

3. After connecting: **File → Open Folder** → `/home/ubuntu/visualwebarena`.

---

## 6. Checklist Summary

| Step | Action |
|------|--------|
| 1 | Create EC2: Ubuntu 22.04 LTS AMI, r6i.xlarge (or g4dn/g5 for GPU), 50 GiB storage, key pair |
| 2 | SSH in and install Python 3.11, Playwright system deps, clone repo, venv, `pip install -r requirements.txt` |
| 3 | Upgrade transformers, tokenizers, datasets, torch; `playwright install` + `install-deps`; `pip install -e .` |
| 4 | Set DATASET, CLASSIFIEDS, SHOPPING, REDDIT, WIKIPEDIA, HOMEPAGE, OPENAI_API_KEY |
| 5 | `python scripts/generate_test_data.py` and `bash prepare.sh` |
| 6 | Run `run.py` (or `run_demo.py`) and inspect `results/` or `demo_test/` |

---

## References

- Main project README: [README.md](../README.md)
- Docker/website setup: [environment_docker/README.md](../environment_docker/README.md)
