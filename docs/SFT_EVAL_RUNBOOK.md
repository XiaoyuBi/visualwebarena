# SFT Evaluation Runbook (Browser + Evaluator)

This runbook documents how to:
- set up the browser instance
- set up the evaluator instance
- configure environment variables
- launch three parallel jobs (Classifieds / Shopping / Reddit, top-100, one run)

All sensitive or deployment-specific values below use placeholders.

## 1) Placeholders To Fill In

Replace these before running commands:
- `<BROWSER_HOST>`: browser host DNS/IP (serves web apps)
- `<EVALUATOR_HOST>`: evaluator host DNS/IP (runs `run.py`)
- `<SSH_KEY_PATH>`: path to your `.pem` file
- `<REPO_PATH>`: path to repo on remote host (usually `~/visualwebarena`)
- `<FIREWORKS_API_KEY>`: Fireworks key (`fw_...`)
- `<OPENAI_API_KEY>`: OpenAI key for evaluator model
- `<SFT_MODEL_ROUTE>`: SFT model route in `model#deployment` format
- `<EVAL_MODEL_NAME>`: evaluator model (example: `gpt-5`)

Example `SFT_MODEL_ROUTE` format:
`accounts/<ACCOUNT>/models/<MODEL_NAME>#accounts/<ACCOUNT>/deployments/<DEPLOYMENT_ID>`

## 2) Browser Instance Setup (`<BROWSER_HOST>`)

### 2.1 SSH In

```bash
ssh -i "<SSH_KEY_PATH>" ubuntu@<BROWSER_HOST>
```

### 2.2 Verify services are up

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

Expected: containers/services for classifieds, shopping, forum (reddit), wikipedia are running.

### 2.3 Verify login URLs do not redirect to stale host

```bash
curl -sSI "http://<BROWSER_HOST>:9980/index.php?page=login" | sed -n '1,15p'
curl -sSI "http://<BROWSER_HOST>:7770/customer/account/login/" | sed -n '1,15p'
```

Expected:
- status `200 OK` (or stable expected response)
- no `Location:` pointing to old/incorrect host

### 2.4 If redirects are wrong, fix site base URLs

Use your service-specific config method (container env/config/db).  
After changes, flush cache/restart service and re-check headers above.

## 3) Evaluator Instance Setup (`<EVALUATOR_HOST>`)

### 3.1 SSH In

```bash
ssh -i "<SSH_KEY_PATH>" ubuntu@<EVALUATOR_HOST>
```

### 3.2 Activate environment

```bash
cd <REPO_PATH>
source venv/bin/activate
```

### 3.3 Configure env vars

Append to `~/.bashrc_vwa`:

```bash
cat >> ~/.bashrc_vwa <<'EOF'
export OPENAI_BASE_URL=https://api.fireworks.ai/inference/v1
export OPENAI_API_KEY=<FIREWORKS_API_KEY>

export EVAL_OPENAI_API_KEY=<OPENAI_API_KEY>
export EVAL_OPENAI_MODEL=<EVAL_MODEL_NAME>

export CLASSIFIEDS=http://<BROWSER_HOST>:9980
export SHOPPING=http://<BROWSER_HOST>:7770
export REDDIT=http://<BROWSER_HOST>:9999
export WIKIPEDIA=http://<BROWSER_HOST>:8888
EOF

source ~/.bashrc_vwa
```

### 3.4 Sanity-check env values (masked)

```bash
python3 - <<'PY'
import os
for k in ["OPENAI_BASE_URL","OPENAI_API_KEY","EVAL_OPENAI_API_KEY","EVAL_OPENAI_MODEL","CLASSIFIEDS","SHOPPING","REDDIT","WIKIPEDIA"]:
    v = os.environ.get(k, "")
    if "KEY" in k and v:
        print(f"{k}={v[:8]}***{v[-4:]}")
    else:
        print(f"{k}={v or '<unset>'}")
PY
```

## 4) Trigger 3 Parallel Jobs (One Run, Top-100)

Run on `<EVALUATOR_HOST>`:

```bash
cd <REPO_PATH>
source ~/.bashrc_vwa
source venv/bin/activate

TS=$(date +%Y%m%d_%H%M%S)
BATCH="results/sft_once_${TS}"
mkdir -p "$BATCH"

INSTR="agent/prompts/jsons/p_som_cot_id_actree_3s.json"
MODEL="<SFT_MODEL_ROUTE>"

nohup python run.py \
  --instruction_path "$INSTR" \
  --test_start_idx 0 --test_end_idx 100 \
  --model "$MODEL" \
  --result_dir "$BATCH/classifieds_top100" \
  --test_config_base_dir config_files/vwa/test_classifieds \
  --action_set_tag som --observation_type image_som \
  --captioning_model Salesforce/blip2-flan-t5-xl \
  --viewport_height 2048 --max_obs_length 3840 --max_images 4 \
  > "$BATCH/classifieds.log" 2>&1 &

nohup python run.py \
  --instruction_path "$INSTR" \
  --test_start_idx 0 --test_end_idx 100 \
  --model "$MODEL" \
  --result_dir "$BATCH/shopping_top100" \
  --test_config_base_dir config_files/vwa/test_shopping \
  --action_set_tag som --observation_type image_som \
  --captioning_model Salesforce/blip2-flan-t5-xl \
  --viewport_height 2048 --max_obs_length 3840 --max_images 4 \
  > "$BATCH/shopping.log" 2>&1 &

nohup python run.py \
  --instruction_path "$INSTR" \
  --test_start_idx 0 --test_end_idx 100 \
  --model "$MODEL" \
  --result_dir "$BATCH/reddit_top100" \
  --test_config_base_dir config_files/vwa/test_reddit \
  --action_set_tag som --observation_type image_som \
  --captioning_model Salesforce/blip2-flan-t5-xl \
  --viewport_height 2048 --max_obs_length 3840 --max_images 4 \
  > "$BATCH/reddit.log" 2>&1 &

echo "BATCH=$BATCH"
```

## 5) Monitor Progress

### 5.1 Process liveness

```bash
pgrep -af "$BATCH/classifieds_top100"
pgrep -af "$BATCH/shopping_top100"
pgrep -af "$BATCH/reddit_top100"
```

### 5.2 Results growth

```bash
for s in classifieds shopping reddit; do
  d="$BATCH/${s}_top100"
  echo "== $s =="
  if [ -f "$d/results.csv" ]; then
    wc -l "$d/results.csv"
    tail -3 "$d/results.csv"
  else
    echo "results.csv not yet"
  fi
done
```

### 5.3 Error signatures

```bash
tail -80 "$BATCH/classifieds.log"
tail -80 "$BATCH/shopping.log"
tail -80 "$BATCH/reddit.log"
```

Look for:
- login/navigation timeout
- evaluator model unavailable
- multimodal prompt/image argument mismatch
- missing NLTK resources

## 6) Recovery Playbook

- **Only one site fails**: stop and relaunch only that site’s process.
- **Login URL timeout**: fix browser-site base URL/redirect config first.
- **Evaluator model errors**: verify `EVAL_OPENAI_MODEL` and key access.
- **Render exists but no CSV row**: task may have failed after render; rerun strategy should account for this.

## 7) Notes

- `launch_eval.sh` is intentionally removed; this runbook is the source of truth.
- Keep all secrets in env vars, not in repo scripts.
