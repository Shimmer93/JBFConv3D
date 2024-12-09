# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import cv2
import mmcv
import numpy as np
import os
import os.path as osp
import shutil
import torch
import torch.distributed as dist
from mmengine.dist.utils import get_dist_info, init_dist
import warnings
from scipy.optimize import linear_sum_assignment
import copy
import pickle
from tqdm import tqdm

import mmengine
from mmengine.dataset import Compose
from mmengine.registry import init_default_scope
from mmaction.apis import inference_recognizer, init_recognizer

def parse_args():
    parser = argparse.ArgumentParser(description='JBF demo')
    parser.add_argument('--ann_file', help='annotation file')
    parser.add_argument('--jbf_dir', help='annotation file')
    parser.add_argument('--out_filename', help='output filename')
    parser.add_argument(
        '--config',
        default='configs/posec3d/slowonly_r50_ntu120_xsub/joint.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--checkpoint',
        default='https://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_ntu120_xsub/joint.pth',
        help='skeleton action recognition checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--label-map',
        default='tools/data/label_map/nturgbd_120.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    parser.add_argument('--local-rank', type=int, default=0)
    # * When non-dist is set, will only use 1 GPU
    parser.add_argument('--non-dist', action='store_true', help='whether to use distributed skeleton extraction')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    args = parser.parse_args()
    return args

def inference_per_frame(model, anno, test_pipeline):
    total_frames = anno['total_frames']
    results_pre_frame = []
    for i in range(total_frames - 47):
        anno_i = copy.deepcopy(anno)
        anno_i['frame_inds'] = np.arange(i, i + 48)
        results_i = inference_recognizer(model, anno_i, test_pipeline)
        max_pred_index = results_i.pred_score.argmax().item()
        max_pred_score = results_i.pred_score.max().item()
        results_pre_frame.append((max_pred_index, max_pred_score))

    return results_pre_frame

def main():
    args = parse_args()

    os.makedirs(args.out_filename, exist_ok=True)

    with open(args.ann_file, 'rb') as f:
        annos = pickle.load(f)
    # annos = sorted(annos, key=lambda x: x['total_frames'])

    print('Initializing distributed environment...')
    if args.non_dist:
        my_part = annos
    else:
        init_dist('pytorch', backend='nccl')
        rank, world_size = get_dist_info()
        dist.barrier()
        my_part = annos[rank::world_size]
        
    config = mmengine.Config.fromfile(args.config)
    init_default_scope(config.get('default_scope', 'mmaction'))
    test_pipeline_cfg = config.test_pipeline
    test_pipeline = Compose(test_pipeline_cfg)
    model = init_recognizer(config, args.checkpoint, 'cuda')

    for anno in tqdm(my_part):
        print(f'Processing {anno["frame_dir"]}')
        anno['start_index'] = 0
        anno['clip_len'] = 48
        anno['frame_interval'] = None
        anno['num_clips'] = 1
        anno['jbf_path'] = osp.join(args.jbf_dir, anno['frame_dir'] + '.npy')
        if anno['total_frames'] < 48:
            print(f'{anno["frame_dir"]} has less than 48 frames, skipped')
            continue
        # try:
        results_per_frame = inference_per_frame(model, anno, test_pipeline)
        with open(args.out_filename + '/' + anno['frame_dir'] + '.pkl', 'wb') as f:
            pickle.dump(results_per_frame, f)
        # except Exception as e:
            # print(f'Error processing {anno["frame_dir"]}: {e}')

if __name__ == '__main__':
    main()