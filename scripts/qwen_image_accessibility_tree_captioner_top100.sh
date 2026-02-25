#!/bin/bash
### Run Qwen VL with image + accessibility_tree + captioner on the first 100 tasks
### of each VWA site type: classifieds, shopping, reddit.
### Requires: export OPENAI_BASE_URL and OPENAI_API_KEY (OpenRouter or Hyperbolic direct).
### Optional: $1 = result_dir suffix (e.g. _hyperbolic → results_qwen_img_actree_cap_classifieds_top100_hyperbolic).
### Optional: export VWA_MODEL to override model.

result_dir_suffix="${1:-}"
model="${VWA_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
instruction_path="agent/prompts/jsons/p_multimodal_cot_id_actree_3s.json"
observation_type="accessibility_tree_with_captioner"
action_set_tag="id_accessibility_tree"
captioning_model="Salesforce/blip2-flan-t5-xl"
start_idx=0
end_idx=100

# --- Classifieds (top 100) ---
result_dir="results_qwen_img_actree_cap_classifieds_top100${result_dir_suffix}"
bash prepare.sh
python run.py \
  --instruction_path "$instruction_path" \
  --test_start_idx $start_idx \
  --test_end_idx $end_idx \
  --model "$model" \
  --result_dir "$result_dir" \
  --test_config_base_dir=config_files/vwa/test_classifieds \
  --action_set_tag "$action_set_tag" \
  --observation_type "$observation_type" \
  --captioning_model "$captioning_model"

# --- Shopping (top 100) ---
result_dir="results_qwen_img_actree_cap_shopping_top100${result_dir_suffix}"
bash prepare.sh
python run.py \
  --instruction_path "$instruction_path" \
  --test_start_idx $start_idx \
  --test_end_idx $end_idx \
  --model "$model" \
  --result_dir "$result_dir" \
  --test_config_base_dir=config_files/vwa/test_shopping \
  --action_set_tag "$action_set_tag" \
  --observation_type "$observation_type" \
  --captioning_model "$captioning_model"

# --- Reddit (top 100) ---
result_dir="results_qwen_img_actree_cap_reddit_top100${result_dir_suffix}"
bash prepare.sh
python run.py \
  --instruction_path "$instruction_path" \
  --test_start_idx $start_idx \
  --test_end_idx $end_idx \
  --model "$model" \
  --result_dir "$result_dir" \
  --test_config_base_dir=config_files/vwa/test_reddit \
  --action_set_tag "$action_set_tag" \
  --observation_type "$observation_type" \
  --captioning_model "$captioning_model"
