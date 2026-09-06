#!/bin/bash
# Xplace global placement + pin-access pre-pass.
#
# ./run_xplace.sh <design>
#   1. GP  : main.py inside the docker container (unchanged flags)
#   2. pre-pass : pull the pin-access-critical cells (ranked by the LEF-derived
#            pin-scarcity score) off the vertical M3 stripes, spread them apart,
#            snap them to site/row and mark them FIXED
#            (see formulation/pin_access_prepass.md)
#
# Outputs in result/<design>/ :
#   placement_<design>_gp_raw.def  the raw GP solution (pre-pass input)
#   placement_<design>_pa.def      the pre-pass result
#   placement_<design>_pa.json     frozen-cell list + displacement stats
#   placement_<design>_gp.def      == _pa.def, so ./run_innovus_routing.sh works
#
# Knobs (env vars). The defaults are the setting that cut total DRC on all six
# measured designs (seed/sha256/misty/cast/camellia/aes; 693 -> 391 violations,
# -43.6%): 10% area budget, 8-site separation, 24-site/2-row displacement cap:
#   PA_PREPASS=on|off        on = GP + pre-pass, off = plain GP
#   PA_AREA_BUDGET=0.10  PA_MIN_SEP_SITES=8  PA_MIN_SEP_ROWS=1
#   PA_MAX_SHIFT_SITES=24  PA_MAX_SHIFT_ROWS=2  PA_SCORE_SIGMA=1.0
#   PA_STRIPE_MARGIN=1
#   PA_LEFS=...  PA_REPORTS=...  DOCKER_CTR=compassionate_pascal

design_name=$1
if [ -z "$design_name" ]; then
  echo "usage: ./run_xplace.sh <design>"
  exit 1
fi

PA_PREPASS=${PA_PREPASS:-on}
PA_AREA_BUDGET=${PA_AREA_BUDGET:-0.10}
PA_MIN_SEP_SITES=${PA_MIN_SEP_SITES:-8}
PA_MIN_SEP_ROWS=${PA_MIN_SEP_ROWS:-1}
PA_MAX_SHIFT_SITES=${PA_MAX_SHIFT_SITES:-24}
PA_MAX_SHIFT_ROWS=${PA_MAX_SHIFT_ROWS:-2}
PA_SCORE_SIGMA=${PA_SCORE_SIGMA:-1.0}
PA_STRIPE_MARGIN=${PA_STRIPE_MARGIN:-1}
PA_LEFS=${PA_LEFS:-xplace_work/L_modified.lef,xplace_work/SL_modified.lef}
PA_REPORTS=${PA_REPORTS:-L_1x_pin_scarcity_report.txt,SL_1x_pin_scarcity_report.txt,R_1x_pin_scarcity_report.txt}
DOCKER_CTR=${DOCKER_CTR:-compassionate_pascal}

RES=result/${design_name}
GP=${RES}/placement_${design_name}_gp.def
RAW=${RES}/placement_${design_name}_gp_raw.def
PA=${RES}/placement_${design_name}_pa.def

python3 tool/generate_contest_json.py "$design_name"


# ---------------------------------------------------------------- 1) global placement
docker exec ${DOCKER_CTR} bash -c \
"cd /workspace && rm -f ${GP} ${RES}/placement_${design_name}_dp.def ${RAW} ${PA} ${RES}/placement_${design_name}_pa.json && python3 main.py \
  --custom_json xplace_work/contest.json \
  --load_from_raw True \
  --seed 1 \
  --target_density 0.85 \
  --output_dir ${design_name} \
  --write_global_placement True \
  --detail_placement False \
  --legalization False \
  --use_m3_snet_density True \
  --m3_stripe_density 0.5 \
  --m3_stripe_density_min 0.15 \
  --adaptive_congest_rc 1.43 \
  --adaptive_m3_stripe_density_min 0.1 \
  --design_name ${design_name} \
  --stop_overflow 0.07 \
  --drv_checkpoint fcn_trainer/results/dynamic_v19_aes2048_dropaoi/best_model.pt \
  --drv_pred_adaptive True \
  --drv_force_weight 0.35 \
  --drv_phase1_vis_freq 0 \
  --exp false \
  --drv_density_update_overflow 0.25 \
  --drv_density_update_step 0.05 \
  --drv_density_update_freq 10  \
  --drv_density_hot_threshold 0.08 \
  --use_route_force false \
  --use_cell_inflate true \
  --min_area_inc 0.005 \
  --deterministic True \
  --drv_ap_json fcn_trainer/SL_modified_m2_ap.json,fcn_trainer/L_modified_m2_ap.json" 2>&1


# ---------------------------------------------------------------- 2) pin-access pre-pass
if [ "${PA_PREPASS}" = "off" ]; then
  echo "[prepass] PA_PREPASS=off -- keeping the plain GP solution in ${GP}"
  exit 0
fi

if [ ! -f "${GP}" ]; then
  echo "[prepass] ERROR: ${GP} not found -- global placement did not finish."
  exit 1
fi

echo "[prepass] scarcity-ranked selection: area_budget=${PA_AREA_BUDGET} min_sep_sites=${PA_MIN_SEP_SITES} max_shift=${PA_MAX_SHIFT_SITES}sites/${PA_MAX_SHIFT_ROWS}row"

# Run inside the container: result/ is written by root there, and the tool only
# needs the python standard library.
docker exec ${DOCKER_CTR} bash -c \
"cd /workspace && cp ${GP} ${RAW} && python3 tool/pin_access_prepass.py \
  --gp_def ${RAW} \
  --pdn_def xplace_work/${design_name}_unplaced.def \
  --lefs ${PA_LEFS} \
  --reports ${PA_REPORTS} \
  --score_sigma ${PA_SCORE_SIGMA} \
  --area_budget ${PA_AREA_BUDGET} \
  --stripe_margin_sites ${PA_STRIPE_MARGIN} \
  --min_sep_sites ${PA_MIN_SEP_SITES} \
  --min_sep_rows ${PA_MIN_SEP_ROWS} \
  --max_shift_sites ${PA_MAX_SHIFT_SITES} \
  --max_shift_rows ${PA_MAX_SHIFT_ROWS} \
  --out_def ${PA} \
  --out_json ${RES}/placement_${design_name}_pa.json" 2>&1
if [ $? -ne 0 ]; then
  echo "[prepass] FAILED (see the self-check line above) -- ${GP} still holds the raw GP solution."
  exit 1
fi

# Publish for ./run_innovus_routing.sh, which reads placement_<design>_gp.def
docker exec ${DOCKER_CTR} bash -c "cd /workspace && cp ${PA} ${GP}"
echo "[prepass] done. ${GP} now holds the pre-pass solution"
echo "[prepass]       (raw GP kept in ${RAW}; frozen-cell list in ${RES}/placement_${design_name}_pa.json)"
echo "[prepass] next: ./run_innovus_routing.sh ${design_name}"
