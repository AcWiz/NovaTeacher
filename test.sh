#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python test.py \
      configs/ssad_fcos_lamost/seod_lamost_new_50_p.py\
      work_dir/... \
      --show-dir  work_dir/....

