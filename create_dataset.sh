#!/bin/bash
CONF=confs/data/lm.yaml
TIMEOUT=23:59:59
N_GPU_NODES=0
N_DEVICES_PER_NODE=1
N_THREADS_PER_WORKER=16

LOG_LEVEL=WARNING python3 src/data_management/create_dataset.py \
    --conf_loc $CONF \
    --n_workers $N_THREADS_PER_WORKER 


