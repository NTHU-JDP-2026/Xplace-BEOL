#!/bin/bash
# PA pre-pass only, from an existing DP def -> upload.
# usage: ./scripts/run_pa.sh <design> <src_tag> <new_tag> <budget> [extra prepass args...]
set -e
d=$1; src=$2; tag=$3; budget=$4; shift 4
X=$(cd "$(dirname "$0")/.." && pwd); S=${CAD_SSH:?set CAD_SSH=user@host}
W=${CAD_FLOW:?set CAD_FLOW=<flow dir on the CAD host>}/iccad_test
C=${XPLACE_CONTAINER:?set XPLACE_CONTAINER to the docker container that mounts ~/Xplace at /workspace}
docker exec $C bash -c "cd /workspace && python3 tool/pin_access_prepass.py \
  --gp_def result/${d}_${src}_dp/placement_${d}_dp.def --pdn_def xplace_work/${d}_L4xw936.def \
  --lefs xplace_work/L_modified.lef --reports data/pin_scarcity/L_modified_pin_scarcity_report.txt \
  --score_sigma 1.0 --area_budget ${budget} --stripe_margin_sites 1 --min_sep_rows 1 \
  --max_shift_sites 24 --max_shift_rows 2 $* \
  --out_def xplace_work/${d}_${tag}_pa.def --out_json result/${d}_${src}_dp/pa_${tag}.json" 2>&1 | grep -E 'frozen|self-check'
scp -q -o BatchMode=yes $X/xplace_work/${d}_${tag}_pa.def $S:$W/xplace_${tag}_${d}.def
echo "[done] $d $tag budget=$budget $*"
