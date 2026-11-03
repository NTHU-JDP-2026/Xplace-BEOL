#!/bin/bash
# scarcity-weighted GP + PA + upload.  usage: TD= Q= PA= ALPHA= ./scripts/run_sc.sh <design> <gpu>
set -e
d=$1; gpu=${2:-0}; td=${TD:-0.85}; q=${Q:-0.97}; pa=${PA:-0.02}; a=${ALPHA:-1.0}
X=$(cd "$(dirname "$0")/.." && pwd); S=${CAD_SSH:?set CAD_SSH=user@host}
W=${CAD_FLOW:?set CAD_FLOW=<flow dir on the CAD host>}/iccad_test
C=${XPLACE_CONTAINER:?set XPLACE_CONTAINER to the docker container that mounts ~/Xplace at /workspace}
docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $C bash -c \
 "cd /workspace && python3 main.py --custom_json xplace_work/contest_${d}_L4xw936.json \
  --load_from_raw True --seed 1 --deterministic True --target_density ${td} --gpu ${gpu} \
  --output_dir ${d}_sc --design_name ${d} \
  --global_placement True --legalization False --detail_placement False --write_global_placement True \
  --use_cell_inflate true --route_cong_quantile ${q} --min_area_inc 0.01 --use_route_force false \
  --inflate_scarcity_report /workspace/data/pin_scarcity/L_modified_pin_scarcity_report.txt --inflate_scarcity_alpha ${a} \
  --use_m3_snet_density True --m3_stripe_density 0.5 --m3_stripe_density_min 0.15 \
  --adaptive_congest_rc 1.43 --adaptive_m3_stripe_density_min 0.1 --stop_overflow 0.07 \
  --drv_checkpoint fcn_trainer/results/dynamic_v19_aes2048_dropaoi/best_model.pt \
  --drv_pred_adaptive True --drv_phase1_vis_freq 0 --exp false \
  --drv_density_update_overflow 0.25 --drv_density_update_step 0.05 --drv_density_update_freq 10 \
  --drv_density_hot_threshold 0.08 --drv_ap_json fcn_trainer/L_modified_m2_ap.json" > /tmp/gp_${d}_sc.log 2>&1
echo "[$d] $(grep -oE 'GR WL: [0-9]+, GR #Vias: [0-9]+, #EstShorts: [0-9]+' /tmp/gp_${d}_sc.log | tail -1)"
docker exec $C bash -c "cd /workspace && python3 tool/pin_access_prepass.py \
  --gp_def result/${d}_sc/placement_${d}_gp.def --pdn_def xplace_work/${d}_L4xw936.def \
  --lefs xplace_work/L_modified.lef --reports data/pin_scarcity/L_modified_pin_scarcity_report.txt \
  --area_budget ${pa} --score_sigma 1.0 --min_sep_sites 8 --min_sep_rows 1 \
  --stripe_margin_sites 1 --max_shift_sites 24 --max_shift_rows 2 \
  --out_def xplace_work/${d}_sc_pa.def" 2>&1 | grep -E 'frozen'
scp -q -o BatchMode=yes $X/xplace_work/${d}_sc_pa.def $S:$W/xplace_sc_${d}.def
echo "[done] $d"
