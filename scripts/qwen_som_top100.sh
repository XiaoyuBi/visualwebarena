#!/bin/bash
### Run Qwen VL with Image + Caps + SoM on the first 100 tasks of each VWA site,
### repeated 3 times for variance computation.
### Requires: export OPENAI_BASE_URL and OPENAI_API_KEY (Hyperbolic).
### Optional: $1 = result_dir suffix (e.g. _hyperbolic).
### Optional: export VWA_MODEL to override model.

result_dir_suffix="${1:-}"
model="${VWA_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
instruction_path="agent/prompts/jsons/p_som_cot_id_actree_3s.json"
captioning_model="Salesforce/blip2-flan-t5-xl"
observation_type="image_som"
action_set_tag="som"
start_idx=0
end_idx=100

for run in 1 2 3; do
  echo "===== Run ${run}/3 ====="

  # --- Classifieds ---
  result_dir="results_qwen_som_classifieds_top100_run${run}${result_dir_suffix}"
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
    --captioning_model "$captioning_model" \
    --viewport_height 2048 --max_obs_length 3840 --max_images 4

  # --- Shopping ---
  result_dir="results_qwen_som_shopping_top100_run${run}${result_dir_suffix}"
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
    --captioning_model "$captioning_model" \
    --viewport_height 2048 --max_obs_length 3840 --max_images 4

  # --- Reddit ---
  result_dir="results_qwen_som_reddit_top100_run${run}${result_dir_suffix}"
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
    --captioning_model "$captioning_model" \
    --viewport_height 2048 --max_obs_length 3840 --max_images 4

done

echo "===== All runs complete ====="
