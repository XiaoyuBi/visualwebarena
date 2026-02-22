#!/bin/bash

## Run Classifieds idx 98-100 with gpt-5-mini and SoM (image_som observation)
model="gpt-5-mini"
result_dir="classifieds_gpt5mini_som_98_100"
instruction_path="agent/prompts/jsons/p_som_cot_id_actree_3s.json"

test_start_idx=98
test_end_idx=101

python run.py \
  --instruction_path $instruction_path \
  --test_start_idx $test_start_idx \
  --test_end_idx $test_end_idx \
  --model $model \
  --result_dir $result_dir \
  --test_config_base_dir config_files/vwa/test_classifieds \
  --repeating_action_failure_th 5 --viewport_height 2048 --max_obs_length 3840 \
  --action_set_tag som --observation_type image_som
