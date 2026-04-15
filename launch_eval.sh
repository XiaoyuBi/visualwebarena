#!/bin/bash
source ~/visualwebarena/venv/bin/activate
source ~/.bashrc_vwa
cd ~/visualwebarena

MODEL="accounts/xiaoyubi-kyywdxfo90t/models/sft-vwa-image-resized-2epoch#accounts/xiaoyubi-kyywdxfo90t/deployments/qwen25vl7b-sft"
COMMON="--instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json --test_start_idx 0 --test_end_idx 100 --action_set_tag som --observation_type image_som --captioning_model Salesforce/blip2-flan-t5-xl --viewport_height 2048 --max_obs_length 3840 --max_images 4"

nohup python run.py $COMMON --model "$MODEL" --result_dir results_sft_classifieds_top100 --test_config_base_dir config_files/vwa/test_classifieds > logs_classifieds.log 2>&1 &
nohup python run.py $COMMON --model "$MODEL" --result_dir results_sft_shopping_top100 --test_config_base_dir config_files/vwa/test_shopping > logs_shopping.log 2>&1 &
nohup python run.py $COMMON --model "$MODEL" --result_dir results_sft_reddit_top100 --test_config_base_dir config_files/vwa/test_reddit > logs_reddit.log 2>&1 &

echo "Launched 3 processes"
sleep 3
ps aux | grep "python run.py" | grep -v grep | wc -l
