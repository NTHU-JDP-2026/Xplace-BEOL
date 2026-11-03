#!/bin/bash
# usage: ./scripts/run_xplace_util.sh <design> <tagprefix>     例: ./scripts/run_xplace_util.sh hidden1 u40
# 只跑 R_modified；Xplace target_density 由 XP_TARGET_DENSITY 控制（預設 0.8）
set -e
d=$1; P=$2
X=$(cd "$(dirname "$0")/.." && pwd)
S=${CAD_SSH:?set CAD_SSH=user@host}
W=${CAD_FLOW:?set CAD_FLOW=<flow dir on the CAD host>}/iccad_test
CTR=${XPLACE_CONTAINER:?set XPLACE_CONTAINER to the docker container that mounts ~/Xplace at /workspace}
TD=${XP_TARGET_DENSITY:-0.8}

echo "[1/4] 取回 Innovus ${P}mod 的 placed DEF"
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

echo "[2/4] Xplace GP (target_density=$TD)"
docker exec $CTR bash -c "cd /workspace && python3 main.py --custom_json xplace_work/contest_${d}_${P}.json \
  --load_from_raw True --seed 1 --target_density $TD --output_dir ${d}_${P} --design_name ${d} \
  --write_global_placement True --detail_placement False --legalization False --deterministic True" 2>&1 | grep -E 'GP Stop|exact HPWL|Write placement'

echo "[3/4] PA pre-pass"
docker exec $CTR bash -c "cd /workspace && python3 tool/pin_access_prepass.py \
  --gp_def result/${d}_${P}/placement_${d}_gp.def \
  --pdn_def xplace_work/${d}_${P}.def \
  --lefs xplace_work/R_modified.lef \
  --reports data/pin_scarcity/R_modified_pin_scarcity_report.txt \
  --score_sigma 1.0 --area_budget 0.10 --stripe_margin_sites 1 \
  --min_sep_sites 8 --min_sep_rows 1 --max_shift_sites 24 --max_shift_rows 2 \
  --out_def result/${d}_${P}/placement_${d}_pa.def \
  --out_json result/${d}_${P}/placement_${d}_pa.json" 2>&1 | grep -E 'prepass'

echo "[4/4] 上傳"
scp -q -o BatchMode=yes $X/result/${d}_${P}/placement_${d}_gp.def $S:$W/xplace_gp_${d}_${P}.def
scp -q -o BatchMode=yes $X/result/${d}_${P}/placement_${d}_pa.def $S:$W/xplace_pa_${d}_${P}.def
echo "[done] $d ($P)"
