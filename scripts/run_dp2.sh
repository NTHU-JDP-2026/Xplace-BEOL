#!/bin/bash
# PA 之後再跑一次 Xplace LG+DP（尊重 PA 的 FIXED），然後上傳
# usage: ./scripts/run_dp2.sh <design> <pa_tag> <gpu>   ; TD=<target_density>
set -e
d=$1; pa=$2; gpu=${3:-0}; td=${TD:-0.7}
X=$(cd "$(dirname "$0")/.." && pwd); S=${CAD_SSH:?set CAD_SSH=user@host}
W=${CAD_FLOW:?set CAD_FLOW=<flow dir on the CAD host>}/iccad_test
C=${XPLACE_CONTAINER:?set XPLACE_CONTAINER to the docker container that mounts ~/Xplace at /workspace}
cat > $X/xplace_work/contest_${d}_${pa}_pa.json <<JSON
{ "benchmark": "xplace_work", "design_name": "${d}",
  "lefs": ["/workspace/xplace_work/asap7_tech_4x_201209.lef","/workspace/xplace_work/L_modified.lef"],
  "def": "/workspace/xplace_work/${d}_${pa}_pa.def" }
JSON
docker exec $C bash -c "cd /workspace && python3 main.py --custom_json xplace_work/contest_${d}_${pa}_pa.json \
  --load_from_raw True --seed 1 --target_density ${td} --output_dir ${d}_${pa}_dp2 --design_name ${d} --gpu ${gpu} \
  --global_placement False --legalization True --detail_placement True --write_placement True \
  --deterministic True" 2>&1 | grep -E '#Mov|#Fix|Input solution|Finish Legalization|After DP|Write placement'
scp -q -o BatchMode=yes $X/result/${d}_${pa}_dp2/placement_${d}_dp.def $S:$W/xplace_${pa}dp2_${d}.def
echo "[done] $d ${pa}dp2"
