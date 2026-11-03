#!/bin/bash
set -e
CAD_USER=${CAD_USER:?set CAD_USER}
CAD_HOST=${CAD_HOST:?set CAD_HOST}
CAD_DIR=${CAD_DIR:?set CAD_DIR}

DESIGN_NAME=$1

scp "./result/${DESIGN_NAME}/placement_${DESIGN_NAME}_gp.def" \
    "${CAD_USER}@${CAD_HOST}:${CAD_DIR}/exp/exp.def"

ssh "${CAD_USER}@${CAD_HOST}" "cd ${CAD_DIR} && ./run_xplace_flow_exp.sh ${DESIGN_NAME}"
scp "${CAD_USER}@${CAD_HOST}:${CAD_DIR}/designs/${DESIGN_NAME}/${DESIGN_NAME}.drc.rpt" "./exp/${DESIGN_NAME}.drc.rpt"
scp "${CAD_USER}@${CAD_HOST}:${CAD_DIR}/designs/${DESIGN_NAME}/${DESIGN_NAME}_summary.rpt" "./exp/${DESIGN_NAME}_summary.rpt"