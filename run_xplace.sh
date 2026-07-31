#!/bin/bash

design_name=$1

python3 tool/generate_contest_json.py "$design_name"



# docker exec compassionate_pascal bash -c \
# "cd /workspace && python3 main.py \
# --custom_json xplace_work/contest.json \
# --load_from_raw True \
# --target_density 1 \
# --output_dir ${design_name} \
# --write_global_placement True \
# --use_cell_inflate False \
# --detail_placement False \
# --design_name ${design_name}" 2>&1


docker exec compassionate_pascal bash -c \
"cd /workspace && python3 main.py \
  --custom_json xplace_work/contest.json \
  --load_from_raw True \
  --target_density 0.85 \
  --output_dir ${design_name} \
  --write_global_placement True \
  --detail_placement False \
  --use_m3_snet_density True \
  --m3_stripe_density 0.4 \
  --m3_stripe_density_min 0.25 \
  --design_name ${design_name} \
  --stop_overflow 0.07 \
  --drv_checkpoint fcn_trainer/results/res256/best_model.pt \
  --drv_pred_resolution 256 \
  --drv_force_weight 0.35 \
  --drv_finetune_iters 100 \
  --drv_phase1_vis_freq 0 \
  --exp true \
  --drv_density_update_overflow 0.25 \
  --drv_density_update_step 0.05 \
  --drv_density_update_freq 5  \
  --drv_density_hot_threshold 0.05 \
  --drv_ap_json fcn_trainer/SL_modified_m2_ap.json,fcn_trainer/L_modified_m2_ap.json" 2>&1

