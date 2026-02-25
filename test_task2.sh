#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
python test.py \
       configs/ssad_fcos_lamost/seod_lamost_new_50_p.py \
       work_dir/newLAMOST_ori/denseT/sparse/saprse_50/iter_51200.pth \
       --show-dir  work_dir/newLAMOST_ori/denseT/sparse/saprse_50/iter_51200/test_img/ \


# python test.py \
#        configs/ssad_fcos_lamost/seod_lamost_new_90_p.py \
#        work_dir/new_lamost_ori/sood/sparse/saprse_90/iter_57600.pth \
#        --show-dir  work_dir/new_lamost_ori/sood/sparse/saprse_90/test_img/iter_57600 \


# python test.py \
#       configs/ssad_fcos_lamost/seod_lamost_new_50_p.py \
#       work_dir/newLAMOST_ori/SotT-com_0420_enhance/sparse/saprse_70_180k/iter_105600.pth \
#       --show-dir  work_dir/newLAMOST_ori/SotT-com_0420_enhance/sparse/saprse_70_180k/test_img/iter_105600/ 


# python test.py \
#       configs/ssad_fcos_lamost/seod_lamost_new_50_p.py\
#       work_dir/newLAMOST_ori/SotT-com_0420_enhance/sparse/saprse_90_180k/iter_172800.pth \
#       --show-dir  work_dir/newLAMOST_ori/SotT-com_0420_enhance/sparse/saprse_90_180k/test_img/iter_172800_0.08/