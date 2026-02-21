#!/bin/bash
### Run Qwen VL with accessibility_tree on the first 100 tasks of each VWA site type:
### classifieds, shopping, reddit.
### Requires: export OPENAI_BASE_URL=https://openrouter.ai/api/v1 and OPENAI_API_KEY=<your_openrouter_key>

model="qwen/qwen-2.5-vl-7b-instruct"
instruction_path="agent/prompts/jsons/p_cot_id_actree_3s.json"
observation_type="accessibility_tree"
action_set_tag="id_accessibility_tree"
start_idx=0
end_idx=100

# --- Classifieds (top 100) ---
result_dir="results_qwen_actree_classifieds_top100"
bash prepare.sh
python run.py \
  --instruction_path "$instruction_path" \
  --test_start_idx $start_idx \
  --test_end_idx $end_idx \
  --model "$model" \
  --result_dir "$result_dir" \
  --test_config_base_dir=config_files/vwa/test_classifieds \
  --action_set_tag "$action_set_tag" \
  --observation_type "$observation_type"

# --- Shopping (top 100) ---
result_dir="results_qwen_actree_shopping_top100"
bash prepare.sh
python run.py \
  --instruction_path "$instruction_path" \
  --test_start_idx $start_idx \
  --test_end_idx $end_idx \
  --model "$model" \
  --result_dir "$result_dir" \
  --test_config_base_dir=config_files/vwa/test_shopping \
  --action_set_tag "$action_set_tag" \
  --observation_type "$observation_type"

# --- Reddit (top 100) ---
result_dir="results_qwen_actree_reddit_top100"
bash prepare.sh
python run.py \
  --instruction_path "$instruction_path" \
  --test_start_idx $start_idx \
  --test_end_idx $end_idx \
  --model "$model" \
  --result_dir "$result_dir" \
  --test_config_base_dir=config_files/vwa/test_reddit \
  --action_set_tag "$action_set_tag" \
  --observation_type "$observation_type"
