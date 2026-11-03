#!/bin/bash
# L library 4x：Xplace tuned GP + PA pre-pass（完全沿用舊設計的配置）
# usage: ./scripts/run_xplace_L4x.sh <design>
set -e
d=$1
X=$(cd "$(dirname "$0")/.." && pwd)
S=${CAD_SSH:?set CAD_SSH=user@host}
W=${CAD_FLOW:?set CAD_FLOW=<flow dir on the CAD host>}/iccad_test
CTR=${XPLACE_CONTAINER:?set XPLACE_CONTAINER to the docker container that mounts ~/Xplace at /workspace}

echo "[1/4] 取回 Innovus L4xmod 的 placed DEF"
scp -q -o BatchMode=yes $S:$W/save_${d}_L4xmod/${d}_placed.def $X/xplace_work/${d}_L4x.def
cat > $X/xplace_work/contest_${d}_L4x.json <<JSON
{
    "benchmark": "xplace_work",
    "design_name": "${d}",
    "lefs": [
        "/workspace/xplace_work/asap7_tech_4x_201209.lef",
        "/workspace/xplace_work/L_modified.lef"
    ],
    "def": "/workspace/xplace_work/${d}_L4x.def"
}
JSON

echo "[2/4] Xplace tuned GP"
docker exec $CTR bash -c "cd /workspace && python3 main.py \
  --custom_json xplace_work/contest_${d}_L4x.json \
  --load_from_raw True --seed 1 --target_density 0.85 \
  --output_dir ${d}_L4x --design_name ${d} \
  --write_global_placement True --detail_placement False --legalization False \
  --use_m3_snet_density True --m3_stripe_density 0.5 --m3_stripe_density_min 0.15 \
  --adaptive_congest_rc 1.43 --adaptive_m3_stripe_density_min 0.1 --stop_overflow 0.07 \
  --drv_checkpoint fcn_trainer/results/dynamic_v19_aes2048_dropaoi/best_model.pt \
  --drv_pred_adaptive True --drv_phase1_vis_freq 0 --exp false \
  --drv_density_update_overflow 0.25 --drv_density_update_step 0.05 \
  --drv_density_update_freq 10 --drv_density_hot_threshold 0.08 \
  --use_route_force false --use_cell_inflate true --min_area_inc 0.005 --deterministic True \
  --drv_ap_json fcn_trainer/L_modified_m2_ap.json" 2>&1 | grep -E 'GP Stop|exact HPWL'

echo "[3/4] PA pre-pass（4x：位移上限與間隔也 x4）"
docker exec $CTR bash -c "cd /workspace && python3 tool/pin_access_prepass.py \
  --gp_def result/${d}_L4x/placement_${d}_gp.def \
  --pdn_def xplace_work/${d}_L4x.def \
  --lefs xplace_work/L_modified.lef \
  --reports data/pin_scarcity/L_modified_pin_scarcity_report.txt \
  --score_sigma 1.0 --area_budget 0.10 --stripe_margin_sites 1 \
  --min_sep_sites 8 --min_sep_rows 1 --max_shift_sites 24 --max_shift_rows 2 \
  --out_def result/${d}_L4x/placement_${d}_pa.def \
  --out_json result/${d}_L4x/placement_${d}_pa.json" 2>&1 | grep -E 'prepass\] (M3|scarcity|movable|frozen|displacement|self-check)'

echo "[4/4] 上傳"
scp -q -o BatchMode=yes $X/result/${d}_L4x/placement_${d}_gp.def $S:$W/xplace_L4x_${d}.def
scp -q -o BatchMode=yes $X/result/${d}_L4x/placement_${d}_pa.def $S:$W/xplace_L4xpa_${d}.def
echo "[done] $d"
