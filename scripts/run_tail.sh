#!/bin/bash
# LG+DP -> PA(5%) -> upload, starting from an existing GP def.  usage: ./scripts/run_tail.sh <design> <tag> <gpu>
set -e
d=$1; tag=$2; gpu=${3:-0}; td=${TD:-0.7}
X=$(cd "$(dirname "$0")/.." && pwd); S=${CAD_SSH:?set CAD_SSH=user@host}
W=${CAD_FLOW:?set CAD_FLOW=<flow dir on the CAD host>}/iccad_test
C=${XPLACE_CONTAINER:?set XPLACE_CONTAINER to the docker container that mounts ~/Xplace at /workspace}
cat > $X/xplace_work/contest_${d}_${tag}_gp.json <<JSON
{ "benchmark": "xplace_work", "design_name": "${d}",
  "lefs": ["/workspace/xplace_work/asap7_tech_4x_201209.lef","/workspace/xplace_work/L_modified.lef"],
  "def": "/workspace/result/${d}_${tag}/placement_${d}_gp.def" }
JSON
echo "[1/3] LG + DP"
docker exec $C bash -c "cd /workspace && python3 main.py --custom_json xplace_work/contest_${d}_${tag}_gp.json \
  --load_from_raw True --seed 1 --target_density ${td} --output_dir ${d}_${tag}_dp --design_name ${d} --gpu ${gpu} \
  --global_placement False --legalization True --detail_placement True --write_placement True \
  --deterministic True" 2>&1 | grep -E '#Mov|Finish Legalization|Write placement|HPWL'
echo "[2/3] PA pre-pass (5%)"
docker exec $C bash -c "cd /workspace && python3 tool/pin_access_prepass.py \
  --gp_def result/${d}_${tag}_dp/placement_${d}_dp.def --pdn_def xplace_work/${d}_L4xw936.def \
  --lefs xplace_work/L_modified.lef --reports data/pin_scarcity/L_modified_pin_scarcity_report.txt \
  --score_sigma 1.0 --area_budget 0.05 --stripe_margin_sites 1 --min_sep_sites 8 --min_sep_rows 1 \
  --max_shift_sites 24 --max_shift_rows 2 --out_def xplace_work/${d}_${tag}_pa.def \
  --out_json result/${d}_${tag}_dp/pa.json" 2>&1 | grep -E 'frozen|self-check'
echo "[3/3] upload"
scp -q -o BatchMode=yes $X/xplace_work/${d}_${tag}_pa.def $S:$W/xplace_${tag}_${d}.def
echo "[done] $d $tag"
