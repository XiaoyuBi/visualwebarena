#!/bin/bash
### Re-run failed tasks (those with render HTML but no results.csv entry).
### Removes the HTML files for failed tasks so run.py's get_unfinished() picks them up.
### Usage: bash scripts/rerun_failed.sh <result_dir> [run.py args...]
### Example: bash scripts/rerun_failed.sh results_qwen_som_classifieds_top100_run1 \
###            --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json ...

result_dir="$1"
shift

if [ ! -d "$result_dir" ]; then
  echo "ERROR: result_dir '$result_dir' does not exist"
  exit 1
fi

csv_file="$result_dir/results.csv"
if [ ! -f "$csv_file" ]; then
  echo "ERROR: No results.csv in $result_dir"
  exit 1
fi

# Get scored task IDs from results.csv
scored_ids=$(tail -n +2 "$csv_file" | cut -d, -f1 | sort -n)

# Get all task IDs that have HTML files
html_ids=$(ls "$result_dir"/render_*.html 2>/dev/null | sed 's/.*render_//;s/\.html//' | sort -n)

# Find failed task IDs (have HTML but not in CSV)
failed_count=0
for id in $html_ids; do
  if ! echo "$scored_ids" | grep -qx "$id"; then
    echo "Removing render for failed task $id"
    rm -f "$result_dir/render_${id}.html"
    rm -f "$result_dir/traces/${id}.zip"
    failed_count=$((failed_count + 1))
  fi
done

echo "Removed $failed_count failed task renders. Re-running..."
python run.py --result_dir "$result_dir" "$@"
