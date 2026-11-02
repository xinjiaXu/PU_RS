#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bcd_root="$repo_root/building-change-detection"
data_root="${DATA_ROOT:-/home/xxj/Data}"
gpu_id="${GPU_ID:-0}"
workers="${NUM_WORKERS:-4}"
batch_size="${BATCH_SIZE:-4}"

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s {levircd|levircdplus|cdd|whucd|sysucd}\n' "$0" >&2
  exit 2
fi

case "$1" in
  levircd) name="LEVIR-CD"; root="$data_root/LEVIR_CD"; image_a=A; image_b=B; label=label ;;
  levircdplus) name="LEVIR-CD+"; root="$data_root/LEVIR_CD_PLUS/cut_data"; image_a=A; image_b=B; label=label ;;
  cdd) name="CDD"; root="$data_root/CDD"; image_a=A; image_b=B; label=OUT ;;
  whucd) name="WHU-CD"; root="$data_root/WHUCD"; image_a=A; image_b=B; label=label ;;
  sysucd) name="SYSU-CD"; root="$data_root/SYSU_CD"; image_a=time1; image_b=time2; label=label ;;
  *) printf 'Unknown dataset key: %s\n' "$1" >&2; exit 2 ;;
esac

for folder in "$image_a" "$image_b" "$label"; do
  [[ -d "$root/test/$folder" ]] || { printf 'Missing test data directory: %s\n' "$root/test/$folder" >&2; exit 1; }
done

checkpoint="$repo_root/PU_RS_work_dirs/$1/$name/PVT-v2/Seg_epoch_best.pth"
[[ -f "$checkpoint" ]] || { printf 'Checkpoint not found: %s\nTrain this dataset first.\n' "$checkpoint" >&2; exit 1; }

cd "$bcd_root"
CUDA_VISIBLE_DEVICES="$gpu_id" python test.py \
  --data_name "$name" --test-root "$root/test" \
  --image-dir-a "$image_a" --image-dir-b "$image_b" --label-dir "$label" \
  --save_path "$repo_root/PU_RS_work_dirs/$1" --checkpoint "$checkpoint" \
  --output-dir "$repo_root/PU_RS_work_dirs/$1/test_predictions" \
  --num-workers "$workers" --batchsize "$batch_size"
