#!/bin/bash
### Re-run failed tasks from Run 1 first, then continue with Run 2 and Run 3.
### Requires: export OPENAI_BASE_URL and OPENAI_API_KEY (Hyperbolic).

result_dir_suffix="${1:-}"
model="${VWA_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
instruction_path="agent/prompts/jsons/p_som_cot_id_actree_3s.json"
captioning_model="Salesforce/blip2-flan-t5-xl"
observation_type="image_som"
action_set_tag="som"
start_idx=0
end_idx=100

common_args="--instruction_path $instruction_path \
  --test_start_idx $start_idx --test_end_idx $end_idx \
  --model $model \
  --action_set_tag $action_set_tag --observation_type $observation_type \
  --captioning_model $captioning_model \
  --viewport_height 2048 --max_obs_length 3840 --max_images 4"

echo "===== Re-running failed tasks from Run 1 ====="

for site in classifieds shopping reddit; do
  result_dir="results_qwen_som_${site}_top100_run1${result_dir_suffix}"
  if [ -d "$result_dir" ] && [ -f "$result_dir/results.csv" ]; then
    echo "--- Re-running failed: $site Run 1 ---"
    bash prepare.sh
    bash scripts/rerun_failed.sh "$result_dir" \
      $common_args \
      --test_config_base_dir=config_files/vwa/test_${site}
  else
    echo "--- $site Run 1: no results dir, running fresh ---"
    result_dir="results_qwen_som_${site}_top100_run1${result_dir_suffix}"
    bash prepare.sh
    python run.py \
      $common_args \
      --result_dir "$result_dir" \
      --test_config_base_dir=config_files/vwa/test_${site}
  fi
done

echo "===== Run 1 re-run complete. Continuing with Run 2 and Run 3 ====="

for run in 2 3; do
  echo "===== Run ${run}/3 ====="

  for site in classifieds shopping reddit; do
    result_dir="results_qwen_som_${site}_top100_run${run}${result_dir_suffix}"
    echo "--- ${site} Run ${run} ---"
    bash prepare.sh
    python run.py \
      $common_args \
      --result_dir "$result_dir" \
      --test_config_base_dir=config_files/vwa/test_${site}
  done
done

echo "===== All runs complete ====="
