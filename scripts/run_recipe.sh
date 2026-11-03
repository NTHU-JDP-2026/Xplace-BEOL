#!/bin/bash
# 最佳配方 q97t85p02: GP(td0.85,q0.97,inflate) -> PA 2% on gp.def -> upload
# usage: ./scripts/run_recipe.sh <design> <gpu>
set -e
d=$1; gpu=${2:-0}
X=$(cd "$(dirname "$0")/.." && pwd); S=${CAD_SSH:?set CAD_SSH=user@host}
W=${CAD_FLOW:?set CAD_FLOW=<flow dir on the CAD host>}/iccad_test
C=${XPLACE_CONTAINER:?set XPLACE_CONTAINER to the docker container that mounts ~/Xplace at /workspace}
cat > $X/xplace_work/contest_${d}_L4xw936.json <<JSON
{ "benchmark": "xplace_work", "design_name": "${d}",
  "lefs": ["/workspace/xplace_work/asap7_tech_4x_201209.lef","/workspace/xplace_work/L_modified.lef"],
  "def": "/workspace/xplace_work/${d}_L4xw936.def" }
JSON
echo "[1/3] GP (td0.85, q0.97, inflate)"
docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $C bash -c \
 "cd /workspace && python3 main.py --custom_json xplace_work/contest_${d}_L4xw936.json \
  --load_from_raw True --seed 1 --deterministic True --target_density 0.85 --gpu ${gpu} \
  --output_dir ${d}_rcp --design_name ${d} \
  --global_placement True --legalization False --detail_placement False --write_global_placement True \
  --use_cell_inflate true --route_cong_quantile 0.97 --min_area_inc 0.01 --use_route_force false \
  --use_m3_snet_density True --m3_stripe_density 0.5 --m3_stripe_density_min 0.15 \
  --adaptive_congest_rc 1.43 --adaptive_m3_stripe_density_min 0.1 --stop_overflow 0.07 \
  --drv_checkpoint fcn_trainer/results/dynamic_v19_aes2048_dropaoi/best_model.pt \
  --drv_pred_adaptive True --drv_phase1_vis_freq 0 --exp false \
  --drv_density_update_overflow 0.25 --drv_density_update_step 0.05 --drv_density_update_freq 10 \
  --drv_density_hot_threshold 0.08 --drv_ap_json fcn_trainer/L_modified_m2_ap.json" \
  > /tmp/gp_${d}_rcp.log 2>&1
grep -E "Select best GR|GP Stop" /tmp/gp_${d}_rcp.log | tail -2
echo "[2/3] PA 2% on gp.def"
docker exec $C bash -c "cd /workspace && python3 tool/pin_access_prepass.py \
  --gp_def result/${d}_rcp/placement_${d}_gp.def --pdn_def xplace_work/${d}_L4xw936.def \
  --lefs xplace_work/L_modified.lef --reports data/pin_scarcity/L_modified_pin_scarcity_report.txt \
  --area_budget 0.02 --score_sigma 1.0 --min_sep_sites 8 --min_sep_rows 1 \
  --stripe_margin_sites 1 --max_shift_sites 24 --max_shift_rows 2 \
  --out_def xplace_work/${d}_rcp_pa.def" 2>&1 | grep -E 'frozen|self-check'
echo "[3/3] upload"
scp -q -o BatchMode=yes $X/xplace_work/${d}_rcp_pa.def $S:$W/xplace_rcp_${d}.def
echo "[done] $d"
