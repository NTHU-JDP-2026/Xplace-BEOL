#!/bin/bash
# GP(td0.7 + route force, adaptive cong threshold) -> LG+DP -> PA(5%) -> upload
# usage: ./scripts/run_congfix.sh <design> <tag> [extra GP args...]
set -e
d=$1; tag=$2; shift 2
X=$(cd "$(dirname "$0")/.." && pwd); S=${CAD_SSH:?set CAD_SSH=user@host}
W=${CAD_FLOW:?set CAD_FLOW=<flow dir on the CAD host>}/iccad_test
C=${XPLACE_CONTAINER:?set XPLACE_CONTAINER to the docker container that mounts ~/Xplace at /workspace}

cat > $X/xplace_work/contest_${d}_L4xw936.json <<JSON
{ "benchmark": "xplace_work", "design_name": "${d}",
  "lefs": ["/workspace/xplace_work/asap7_tech_4x_201209.lef","/workspace/xplace_work/L_modified.lef"],
  "def": "/workspace/xplace_work/${d}_L4xw936.def" }
JSON

echo "[1/4] Xplace GP (route force, adaptive cong threshold)"
docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $C bash -c "cd /workspace && python3 main.py --custom_json xplace_work/contest_${d}_L4xw936.json \
  --load_from_raw True --seed 1 --target_density 0.7 --output_dir ${d}_${tag} --design_name ${d} \
  --global_placement True --legalization False --detail_placement False --write_global_placement True \
  --use_m3_snet_density True --m3_stripe_density 0.5 --m3_stripe_density_min 0.15 \
  --adaptive_congest_rc 1.43 --adaptive_m3_stripe_density_min 0.1 --stop_overflow 0.07 \
  --drv_checkpoint fcn_trainer/results/dynamic_v19_aes2048_dropaoi/best_model.pt \
  --drv_pred_adaptive True --drv_phase1_vis_freq 0 --exp false \
  --drv_density_update_overflow 0.25 --drv_density_update_step 0.05 --drv_density_update_freq 10 \
  --drv_density_hot_threshold 0.08 \
  --deterministic True --drv_ap_json fcn_trainer/L_modified_m2_ap.json $*" \
  2>&1 | tee /tmp/gp_${d}_${tag}.log | grep -E 'Congestion threshold|cgMap max|OvflNets|End route|GP Stop|exact HPWL|WARN|nan|Error'

echo "[2/4] LG + DP"
cat > $X/xplace_work/contest_${d}_${tag}_gp.json <<JSON
{ "benchmark": "xplace_work", "design_name": "${d}",
  "lefs": ["/workspace/xplace_work/asap7_tech_4x_201209.lef","/workspace/xplace_work/L_modified.lef"],
  "def": "/workspace/result/${d}_${tag}/placement_${d}_gp.def" }
JSON
docker exec $C bash -c "cd /workspace && python3 main.py --custom_json xplace_work/contest_${d}_${tag}_gp.json \
  --load_from_raw True --seed 1 --target_density 0.7 --output_dir ${d}_${tag}_dp --design_name ${d} \
  --global_placement False --legalization True --detail_placement True --write_placement True \
  --deterministic True" 2>&1 | grep -E '#Mov|Finish Legalization|Write placement|HPWL'

echo "[3/4] PA pre-pass (5%)"
docker exec $C bash -c "cd /workspace && python3 tool/pin_access_prepass.py \
  --gp_def result/${d}_${tag}_dp/placement_${d}_dp.def --pdn_def xplace_work/${d}_L4xw936.def \
  --lefs xplace_work/L_modified.lef --reports data/pin_scarcity/L_modified_pin_scarcity_report.txt \
  --score_sigma 1.0 --area_budget 0.05 --stripe_margin_sites 1 --min_sep_sites 8 --min_sep_rows 1 \
  --max_shift_sites 24 --max_shift_rows 2 --out_def xplace_work/${d}_${tag}_pa.def \
  --out_json result/${d}_${tag}_dp/pa.json" 2>&1 | grep -E 'frozen|self-check'

echo "[4/4] upload"
scp -q -o BatchMode=yes $X/xplace_work/${d}_${tag}_pa.def $S:$W/xplace_${tag}_${d}.def
echo "[done] $d $tag"
