#!/bin/bash
# GP only, report congestion/HPWL trade-off.  usage: ./scripts/run_gpsweep.sh <design> <tag> <gpu> [extra args...]
set -e
d=$1; tag=$2; gpu=$3; shift 3; td=${TD:-0.7}
X=$(cd "$(dirname "$0")/.." && pwd); C=${XPLACE_CONTAINER:?set XPLACE_CONTAINER to the docker container that mounts ~/Xplace at /workspace}
docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $C bash -c \
 "cd /workspace && python3 main.py --custom_json xplace_work/contest_${d}_L4xw936.json \
  --load_from_raw True --seed 1 --target_density ${td} --output_dir ${d}_${tag} --design_name ${d} --gpu ${gpu} \
  --global_placement True --legalization False --detail_placement False --write_global_placement True \
  --use_m3_snet_density True --m3_stripe_density 0.5 --m3_stripe_density_min 0.15 \
  --adaptive_congest_rc 1.43 --adaptive_m3_stripe_density_min 0.1 --stop_overflow 0.07 \
  --drv_checkpoint fcn_trainer/results/dynamic_v19_aes2048_dropaoi/best_model.pt \
  --drv_pred_adaptive True --drv_phase1_vis_freq 0 --exp false \
  --drv_density_update_overflow 0.25 --drv_density_update_step 0.05 --drv_density_update_freq 10 \
  --drv_density_hot_threshold 0.08 \
  --deterministic True --drv_ap_json fcn_trainer/L_modified_m2_ap.json $*" > /tmp/gp_${d}_${tag}.log 2>&1
echo "[done] $d $tag"
