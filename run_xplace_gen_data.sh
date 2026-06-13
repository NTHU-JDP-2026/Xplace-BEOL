#!/bin/bash

design_name=$1
target_density=$2
m3_reduction=$3
cell_inflate=$4
python3 /home/jiarui/Xplace/tool/generate_contest_json.py "$design_name"



# docker exec compassionate_pascal bash -c \
# "cd /workspace && python3 main.py \
# --custom_json xplace_work/contest.json \
# --load_from_raw True \
# --target_density 1 \
# --output_dir ${design_name} \
# --write_global_placement True \
# --use_cell_inflate True \
# --detail_placement False \
# --draw_placement True \
# --design_name ${design_name}" 2>&1

docker exec compassionate_pascal bash -c \
"cd /workspace && python3 main.py \
--custom_json xplace_work/contest.json \
--load_from_raw True \
--target_density ${target_density} \
--use_m3_snet_density True \
--m3_density_reduction_factor ${m3_reduction} \
--output_dir ${design_name} \
--write_global_placement True \
--use_cell_inflate True \
--detail_placement False \
--draw_placement True \
--design_name ${design_name}" 2>&1
