#!/bin/bash
set -e
# relative paths below are repo-root relative
cd "$(dirname "$0")/.."
CAD_USER=${CAD_USER:?set CAD_USER}
CAD_HOST=${CAD_HOST:?set CAD_HOST}
CAD_DIR=${CAD_DIR:?set CAD_DIR}

DESIGN_NAME=$1
DEF_NAME=$2

scp "./exp/phase2_def/${DEF_NAME}" \
    "${CAD_USER}@${CAD_HOST}:${CAD_DIR}/exp/exp.def"

ssh "${CAD_USER}@${CAD_HOST}" "cd ${CAD_DIR} && ./run_xplace_flow_exp.sh ${DESIGN_NAME}"
scp "${CAD_USER}@${CAD_HOST}:${CAD_DIR}/designs/${DESIGN_NAME}/${DESIGN_NAME}.drc.rpt" "./exp/phase2_rpt/${DEF_NAME}.drc.rpt"
scp "${CAD_USER}@${CAD_HOST}:${CAD_DIR}/designs/${DESIGN_NAME}/${DESIGN_NAME}_summary.rpt" "./exp/phase2_rpt/${DEF_NAME}_summary.rpt"