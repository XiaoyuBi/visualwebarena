#!/bin/bash

## Run Shopping idx 0-100 with gpt-5-mini: SoM, accessibility_tree, and image

# 1. SoM (image_som observation)
model="gpt-5-mini"
test_start_idx=0
test_end_idx=100
result_dir="results/shopping/shopping_gpt5mini_som_0_100_1"
instruction_path="agent/prompts/jsons/p_som_cot_id_actree_3s.json"
python run.py \
  --instruction_path $instruction_path \
  --test_start_idx $test_start_idx \
  --test_end_idx $test_end_idx \
  --model $model \
  --result_dir $result_dir \
  --test_config_base_dir config_files/vwa/test_shopping \
  --repeating_action_failure_th 5 --viewport_height 2048 --max_obs_length 3840 --max_steps 15 \
  --action_set_tag som --observation_type image_som

# 2. accessibility_tree only
model="gpt-5-mini"
test_start_idx=0
test_end_idx=100
result_dir="results/shopping/shopping_gpt5mini_actree_0_100_1"
instruction_path="agent/prompts/jsons/p_cot_id_actree_3s.json"
python run.py \
  --instruction_path $instruction_path \
  --test_start_idx $test_start_idx \
  --test_end_idx $test_end_idx \
  --model $model \
  --result_dir $result_dir \
  --test_config_base_dir config_files/vwa/test_shopping \
  --repeating_action_failure_th 5 --viewport_height 2048 --max_obs_length 3840 --max_steps 15 \
  --action_set_tag id_accessibility_tree --observation_type accessibility_tree

# 3. image (observation_type image)
model="gpt-5-mini"
test_start_idx=0
test_end_idx=100
result_dir="results/shopping/shopping_gpt5mini_image_0_100_1"
instruction_path="agent/prompts/jsons/p_som_cot_id_actree_3s.json"
python run.py \
  --instruction_path $instruction_path \
  --test_start_idx $test_start_idx \
  --test_end_idx $test_end_idx \
  --model $model \
  --result_dir $result_dir \
  --test_config_base_dir config_files/vwa/test_shopping \
  --repeating_action_failure_th 5 --viewport_height 2048 --max_obs_length 3840 --max_steps 15 \
  --action_set_tag som --observation_type image
