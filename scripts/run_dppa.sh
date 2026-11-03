#!/bin/bash
# GP(已存在) → LG+DP → PA → 上傳   （L 4x）
# usage: ./scripts/run_dppa.sh <design>
set -e
d=$1
X=$(cd "$(dirname "$0")/.." && pwd); S=${CAD_SSH:?set CAD_SSH=user@host}
W=${CAD_FLOW:?set CAD_FLOW=<flow dir on the CAD host>}/iccad_test
C=${XPLACE_CONTAINER:?set XPLACE_CONTAINER to the docker container that mounts ~/Xplace at /workspace}

echo "[1/3] LG + DP（吃 papath 那次的 GP 結果，不含 PA）"
cp $X/result/${d}_papath/placement_${d}_gp.def $X/xplace_work/${d}_gponly.def
cat > $X/xplace_work/contest_${d}_gponly.json <<JSON
{ "benchmark": "xplace_work", "design_name": "${d}",
  "lefs": ["/workspace/xplace_work/asap7_tech_4x_201209.lef","/workspace/xplace_work/L_modified.lef"],
  "def": "/workspace/xplace_work/${d}_gponly.def" }
JSON
docker exec $C bash -c "cd /workspace && python3 main.py --custom_json xplace_work/contest_${d}_gponly.json \
  --load_from_raw True --seed 1 --target_density 0.7 --output_dir ${d}_dponly --design_name ${d} \
  --global_placement False --legalization True --detail_placement True --write_placement True \
  --deterministic True" 2>&1 | grep -E '#Mov|Finish Legalization|Write placement'

echo "[2/3] PA pre-pass（在 DP 之後）"
docker exec $C bash -c "cd /workspace && python3 tool/pin_access_prepass.py \
  --gp_def result/${d}_dponly/placement_${d}_dp.def --pdn_def xplace_work/${d}_L4xw936.def \
  --lefs xplace_work/L_modified.lef --reports data/pin_scarcity/L_modified_pin_scarcity_report.txt \
  --score_sigma 1.0 --area_budget 0.10 --stripe_margin_sites 1 --min_sep_sites 8 --min_sep_rows 1 \
  --max_shift_sites 24 --max_shift_rows 2 --out_def xplace_work/${d}_dp_pa.def \
  --out_json result/${d}_dponly/dp_pa.json" 2>&1 | grep -E 'frozen|self-check'

echo "[3/3] 上傳"
scp -q -o BatchMode=yes $X/xplace_work/${d}_dp_pa.def $S:$W/xplace_dppa_${d}.def
echo "[done] $d"
