#!/bin/bash

design_name=$1

python3 tool/generate_contest_json.py "$design_name"



docker exec compassionate_pascal bash -c \
"cd /workspace && python3 main.py \
--custom_json xplace_work/contest.json \
--load_from_raw True \
--target_density 0.6 \
--output_dir ${design_name} \
--write_global_placement True \
--use_cell_inflate True \
--detail_placement False \
--design_name ${design_name}" 2>&1
