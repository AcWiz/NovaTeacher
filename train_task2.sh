#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
       --nnodes=1 \
       --node_rank=0 \
       --master_addr="127.0.0.1" \
       --nproc_per_node=1 \
       --master_port=25500 \
       train.py \
       configs/ssad_fcos/seod_ged_sparse90_p.py\
       --launcher pytorch \
       --work-dir  work_sparse/our/sparse90 \


