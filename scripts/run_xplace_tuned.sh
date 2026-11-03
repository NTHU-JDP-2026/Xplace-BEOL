#!/bin/bash
# 用「你原本 run_xplace.sh 的完整調校配置」跑 ICCAD design（AP json 換成 R）
# usage: ./scripts/run_xplace_tuned.sh <design> <tagprefix>
set -e
d=$1; P=$2
X=$(cd "$(dirname "$0")/.." && pwd)
S=${CAD_SSH:?set CAD_SSH=user@host}
W=${CAD_FLOW:?set CAD_FLOW=<flow dir on the CAD host>}/iccad_test
CTR=${XPLACE_CONTAINER:?set XPLACE_CONTAINER to the docker container that mounts ~/Xplace at /workspace}

echo "[1/3] 取回 Innovus ${P}mod 的 placed DEF（沿用同一個 floorplan/PDN）"
scp -q -o BatchMode=yes $S:$W/save_${d}_${P}mod/${d}_placed.def $X/xplace_work/${d}_${P}.def
cat > $X/xplace_work/contest_${d}_${P}.json <<JSON
{
    "benchmark": "xplace_work",
    "design_name": "${d}",
    "lefs": [
        "/workspace/xplace_work/asap7_tech_1x_201209_fixed.lef",
        "/workspace/xplace_work/R_modified.lef"
    ],
    "def": "/workspace/xplace_work/${d}_${P}.def"
}
JSON

echo "[2/3] Xplace GP — tuned（DRV model + M3 stripe density + cell inflate + R 的 AP json）"
docker exec $CTR bash -c "cd /workspace && python3 main.py \
  --custom_json xplace_work/contest_${d}_${P}.json \
  --load_from_raw True --seed 1 --target_density 0.85 \
  --output_dir ${d}_${P}_tuned --design_name ${d} \
  --write_global_placement True --detail_placement False --legalization False \
  --use_m3_snet_density True --m3_stripe_density 0.5 --m3_stripe_density_min 0.15 \
  --adaptive_congest_rc 1.43 --adaptive_m3_stripe_density_min 0.1 \
  --stop_overflow 0.07 \
  --drv_checkpoint fcn_trainer/results/dynamic_v19_aes2048_dropaoi/best_model.pt \
  --drv_pred_adaptive True --drv_phase1_vis_freq 0 --exp false \
  --drv_density_update_overflow 0.25 --drv_density_update_step 0.05 \
  --drv_density_update_freq 10 --drv_density_hot_threshold 0.08 \
  --use_route_force false --use_cell_inflate true --min_area_inc 0.005 \
  --deterministic True \
  --drv_ap_json fcn_trainer/R_modified_m2_ap.json" 2>&1 | grep -E 'GP Stop|exact HPWL|Write placement|drv|inflate|ERROR' | tail -12

echo "[3/3] 上傳"
scp -q -o BatchMode=yes $X/result/${d}_${P}_tuned/placement_${d}_gp.def $S:$W/xplace_tuned_${d}_${P}.def
echo "[done] $d tuned"
