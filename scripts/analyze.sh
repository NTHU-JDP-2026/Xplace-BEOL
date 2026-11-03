#!/bin/bash

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <design_name>"
    exit 1
fi

DESIGN_NAME="$1"

for i in $(seq 0 10 90); do
    SUFFIX=$(printf "%05d" "$i")
    DEF_FILE="${DESIGN_NAME}_ft${SUFFIX}.def"

    echo "=================================================="
    echo "Running ${HERE}/exp_def.sh ${DEF_FILE}"
    echo "=================================================="

    "${HERE}/exp_def.sh" "${DEF_FILE}"
done

echo "All experiments finished."