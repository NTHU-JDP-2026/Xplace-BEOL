#!/bin/bash
# 針對 iccad_test 的 util-0.6 實驗跑 Xplace：GP(origin) / GP(modified) / GP+PA(modified)
# usage: ./scripts/run_xplace_u60.sh <design>
set -e
d=$1
X=$(cd "$(dirname "$0")/.." && pwd)
S=${CAD_SSH:?set CAD_SSH=user@host}
W=${CAD_FLOW:?set CAD_FLOW=<flow dir on the CAD host>}/iccad_test
CTR=${XPLACE_CONTAINER:?set XPLACE_CONTAINER to the docker container that mounts ~/Xplace at /workspace}

echo "[1/5] 取回 Innovus 的 placed DEF"
scp -q -o BatchMode=yes $S:$W/save_${d}_w378orig/${d}_placed.def $X/xplace_work/${d}_unplaced.def

for lef in R_origin R_modified; do
  cat > $X/xplace_work/contest_${d}_${lef}.json <<JSON
{
    "benchmark": "xplace_work",
    "design_name": "${d}",
    "lefs": [
        "/workspace/xplace_work/asap7_tech_1x_201209_fixed.lef",
        "/workspace/xplace_work/${lef}.lef"
    ],
    "def": "/workspace/xplace_work/${d}_unplaced.def"
}
JSON
done

echo "[2/5] Xplace GP (R_origin)"
docker exec $CTR bash -c "cd /workspace && python3 main.py --custom_json xplace_work/contest_${d}_R_origin.json \
  --load_from_raw True --seed 1 --target_density 0.85 --output_dir ${d}_u60_orig --design_name ${d} \
  --write_global_placement True --detail_placement False --legalization False --deterministic True" 2>&1 | grep -E 'GP Stop|exact HPWL|Write placement'

echo "[3/5] Xplace GP (R_modified)"
docker exec $CTR bash -c "cd /workspace && python3 main.py --custom_json xplace_work/contest_${d}_R_modified.json \
  --load_from_raw True --seed 1 --target_density 0.85 --output_dir ${d}_u60_mod --design_name ${d} \
  --write_global_placement True --detail_placement False --legalization False --deterministic True" 2>&1 | grep -E 'GP Stop|exact HPWL|Write placement'

echo "[4/5] PA pre-pass (R_modified)"
docker exec $CTR bash -c "cd /workspace && python3 tool/pin_access_prepass.py \
  --gp_def result/${d}_u60_mod/placement_${d}_gp.def \
  --pdn_def xplace_work/${d}_unplaced.def \
  --lefs xplace_work/R_modified.lef \
  --reports data/pin_scarcity/R_modified_pin_scarcity_report.txt \
  --score_sigma 1.0 --area_budget 0.10 --stripe_margin_sites 1 \
  --min_sep_sites 8 --min_sep_rows 1 --max_shift_sites 24 --max_shift_rows 2 \
  --out_def result/${d}_u60_mod/placement_${d}_pa.def \
  --out_json result/${d}_u60_mod/placement_${d}_pa.json" 2>&1 | grep -E 'prepass'

echo "[5/5] 上傳 3 份 DEF"
scp -q -o BatchMode=yes $X/result/${d}_u60_orig/placement_${d}_gp.def $S:$W/xplace_gp_${d}_orig.def
scp -q -o BatchMode=yes $X/result/${d}_u60_mod/placement_${d}_gp.def  $S:$W/xplace_gp_${d}_mod.def
scp -q -o BatchMode=yes $X/result/${d}_u60_mod/placement_${d}_pa.def  $S:$W/xplace_pa_${d}_mod.def
echo "[done] $d"
