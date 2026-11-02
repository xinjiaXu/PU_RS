#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bcd_root="$repo_root/building-change-detection"
data_root="${DATA_ROOT:-/home/xxj/Data}"
gpu_id="${GPU_ID:-0}"
workers="${NUM_WORKERS:-2}"
batch_size="${BATCH_SIZE:-8}"
epochs="${EPOCHS:-200}"

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

for split in train val; do
  for folder in "$image_a" "$image_b" "$label"; do
    [[ -d "$root/$split/$folder" ]] || { printf 'Missing data directory: %s\n' "$root/$split/$folder" >&2; exit 1; }
  done
done

weight="$bcd_root/pretrained_model/pvt_v2_b4.pth"
[[ -f "$weight" ]] || { printf 'Missing PVT-v2-b4 pretrained weight: %s\n' "$weight" >&2; exit 1; }

cd "$bcd_root"
CUDA_VISIBLE_DEVICES="$gpu_id" python train.py \
  --data_name "$name" \
  --train-root "$root/train" --val-root "$root/val" \
  --image-dir-a "$image_a" --image-dir-b "$image_b" --label-dir "$label" \
  --save_path "$repo_root/PU_RS_work_dirs/$1" \
  --num-workers "$workers" --batchsize "$batch_size" --epoch "$epochs"
