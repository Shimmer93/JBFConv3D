#!/bin/bash

# ./tools/dist_train.sh configs/jbf/slowonly_kinetics400-raw-r50_8xb16-u48-120e_hmdb51-split1-keypoint.py 2 --work-dir work_dirs/hmdb51_raw
./tools/dist_test.sh configs/jbf/slowonly_kinetics400-raw-r50_8xb16-u48-120e_hmdb51-split1-keypoint.py work_dirs/hmdb51_raw/best_acc_top1_epoch_20.pth 2 --work-dir work_dirs/hmdb51_raw