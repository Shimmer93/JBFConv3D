#!/bin/bash

#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu-share
#SBATCH --cpus-per-task=16
##SBATCH --nodelist=hhnode-ib-140

bash tools/dist_run.sh tools/inference.py 8 \
    --ann_file /home/zpengac/pose/PoseSegmentationMask/demo_jbfs5/demo.pkl \
    --jbf_dir /home/zpengac/pose/PoseSegmentationMask/demo_jbfs5 \
    --out_filename demo_output2 \
    --checkpoint /home/zpengac/pose/pyskl/work_dirs/psm/slowonly_r50_ntu120_xsub/joint_final6/best_top1_acc_epoch_24.pth \
    --label-map /home/zpengac/pose/pyskl/tools/data/label_map/nturgbd_120.txt \
    --config configs/jbf/slowonly_r50_8xb16-u48-240e_ntu120-xsub-keypoint_inference.py