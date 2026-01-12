#!/bin/bash

SCRIPT_PATH="main_2a2b.py"
LOG_DIR="experiment_logs"
mkdir -p $LOG_DIR

echo "start.."

SRC_VALUES=(3.0)

for srcweight in "${SRC_VALUES[@]}"; do
    echo "=== srcweight=$srcweight, n_epochs=2000 ==="
    for subid in {1..9}; do
        echo "subid=$subid (srcweight=$srcweight)"
        python $SCRIPT_PATH \
            --datasetname "2a" \
            --datasetdir "/standard_2a_data/" \
            --subid $subid \
            --w_tdf 0.01 \
            --w_adv 0.01 \
            --tdf_t 1.0 \
            --srcweight $srcweight \
            --batch_size 128 \
            --lr 0.0002 \
            --lr_proto 0.002 \
            --w_protonorm 0 \
            --b1 0.5 \
            --b2 0.999 \
            --n_epochs 2000 
        echo "finish subid=$subid"
        sleep 1
    done
done
