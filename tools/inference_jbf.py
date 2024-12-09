import argparse
import copy as cp
import numpy as np
import os
import os.path as osp
import torch.distributed as dist
from tqdm import tqdm

from mmengine import load, dump
from mmengine.dist.utils import get_dist_info, init_dist
from mmaction.apis import inference_recognizer, init_recognizer

default_har_config = 'configs/jbf/slowonly_r50_8xb16-u48-240e_ntu120-xsub-keypoint_inference.py'
default_har_ckpt = 'checkpoints/jbfconv3d_ntu120.pth'

def parse_args():
    parser = argparse.ArgumentParser(description='JBF demo')
    parser.add_argument('--ann-dir', help='annotation directory')
    parser.add_argument('--jbf-dir', help='JBF directory')
    parser.add_argument('--out-result-dir', help='output result directory')
    parser.add_argument('--config', default=default_har_config, help='skeleton action recognition config file path')
    parser.add_argument('--checkpoint', default=default_har_ckpt, help='skeleton action recognition checkpoint file/url')
    parser.add_argument('--label-map', default='tools/data/label_map/nturgbd_120.txt', help='label map file')
    parser.add_argument('--per-video', action='store_true', help='predict action per video instead of per frame')
    parser.add_argument('--local-rank', type=int, default=0)
    # * When non-dist is set, will only use 1 GPU
    parser.add_argument('--non-dist', action='store_true', help='whether to use distributed skeleton extraction')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def inference_per_frame(model, anno):
    total_frames = anno['total_frames']
    results_pre_frame = []
    for i in range(total_frames - 47):
        anno_i = cp.deepcopy(anno)
        anno_i['frame_inds'] = np.arange(i, i + 48)
        results_i = inference_recognizer(model, anno_i)
        max_pred_index = results_i.pred_score.argmax().item()
        max_pred_score = results_i.pred_score.max().item()
        results_pre_frame.append((max_pred_index, max_pred_score))

    return results_pre_frame

def inference_per_video(model, anno):
    results = inference_recognizer(model, anno)
    max_pred_index = results.pred_score.argmax().item()
    max_pred_score = results.pred_score.max().item()
    return max_pred_index, max_pred_score

def main():
    args = parse_args()

    annos = load(args.ann_dir)
    os.makedirs(args.out_result_dir, exist_ok=True)

    print('Initializing distributed environment...')
    if args.non_dist:
        my_part = annos
    else:
        init_dist('pytorch', backend='nccl')
        rank, world_size = get_dist_info()
        dist.barrier()
        my_part = annos[rank::world_size]
        
    model = init_recognizer(args.config, args.checkpoint, 'cuda')

    for anno in tqdm(my_part):
        frame_dir = anno["frame_dir"]
        anno['start_index'] = 0
        anno['clip_len'] = 48
        anno['frame_interval'] = None
        anno['num_clips'] = 1
        anno['jbf_path'] = osp.join(args.jbf_dir, anno['frame_dir'] + '.npy')

        if anno['total_frames'] < 48:
            print(f'{anno["frame_dir"]} has less than 48 frames, skipped')
            continue

        if args.per_video:
            max_pred_index, max_pred_score = inference_per_video(model, anno)
            results = [(max_pred_index, max_pred_score)]
        else:
            results = inference_per_frame(model, anno)
        out_result_fn = osp.join(args.out_result_dir, f'{frame_dir}.pkl')
        dump(results, out_result_fn)

if __name__ == '__main__':
    main()