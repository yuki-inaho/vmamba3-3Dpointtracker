#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p result
exec 9>result/run_kd_additional_r2.lock
flock -n 9 || { echo 'This additional-training job is already running.' >&2; exit 1; }
source scripts/cudnn_env.sh
export HF_HUB_OFFLINE=1
run_dir=result/refiner_kd_improved_20261002/pilot_R2_fullcache_s42
eval_dir=result/refiner_kd_improved_20261002/monitor_full_pilot_R2_s42
args=()
if [[ -f "$run_dir/training_status.json" ]] && [[ $(jq -r .status "$run_dir/training_status.json") == complete ]]; then
  echo 'Training already complete; continuing with evaluation.'
else
  if [[ -d "$run_dir" ]]; then
    [[ -f "$run_dir/last.pt" ]] || { echo 'Existing run has no resumable checkpoint; inspect it before proceeding.' >&2; exit 1; }
    args+=(--resume "$run_dir/last.pt")
  fi
  date '+%Y-%m-%d %H:%M:%S %Z%z'
  uv run --no-sync python scripts/train_refiner_staged.py \
    --mode pilot --method staged \
    --architecture vssd_local_global_128x2_v2_train \
    --config configs/distill_refiner_local_global.yaml \
    --input-manifest result/refiner_kd_improved_20261002/inputs_w16_train/input_manifest.json \
    --monitor-manifest result/refiner_kd_improved_20261002/inputs_w16_validation/input_manifest.json \
    --out-dir "$run_dir" --seed 42 --microbatch 32 --early-stop "${args[@]}"
fi
best_checkpoint=$(jq -er '.best_checkpoint | select(type == "string" and length > 0)' "$run_dir/training_status.json")
if [[ -e "$eval_dir/accuracy_report.json" ]]; then
  echo "An evaluation report already exists at $eval_dir; preserving it."
  exit 0
fi
date '+%Y-%m-%d %H:%M:%S %Z%z'
uv run --no-sync python scripts/evaluate_student_refiner.py \
  --config result/refiner_kd_20261002/selected_config.yaml \
  --student-dir "$run_dir" --mode monitor --provider cuda \
  --checkpoint "$best_checkpoint" \
  --evaluation-inputs result/refiner_kd_20261002/evaluation_inputs/monitor/evaluation_inputs.json \
  --out-dir "$eval_dir"
date '+%Y-%m-%d %H:%M:%S %Z%z'
echo 'Additional R2 training and full-frame monitor evaluation finished.'
