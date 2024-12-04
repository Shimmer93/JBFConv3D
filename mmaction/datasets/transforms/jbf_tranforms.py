# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy
import mmcv
from mmcv.transforms import BaseTransform, KeyMapper
from mmengine.dataset import Compose
from packaging import version as pv
from scipy.stats import mode
from torch.nn.modules.utils import _pair
from PIL import Image
from io import BytesIO
import os
from scipy.ndimage import gaussian_filter

from mmaction.registry import TRANSFORMS
from .loading import DecordDecode, DecordInit
from .processing import _combine_quadruple, Flip
from .pose_transforms import PoseDecode, GeneratePoseTarget

if pv.parse(scipy.__version__) < pv.parse('1.11.0'):
    get_mode = mode
else:
    from functools import partial
    get_mode = partial(mode, keepdims=True)

def read_jbf(jbf_dict):
    jbf_bytes = jbf_dict['jbf'].tobytes()
    jbf_img = Image.open(BytesIO(jbf_bytes))
    jbf_img = np.array(jbf_img)

    J = jbf_dict['nmaps']
    HJ, W = jbf_img.shape[:2]
    assert HJ % J == 0
    H = HJ // J
    
    jbf_out = jbf_img.reshape(J, H, W).astype(np.float32)
    return jbf_out

def read_jbf_seq(fn):
    jbf_seq = np.load(fn, allow_pickle=True)
    jbf_seq = [read_jbf(jbf_dict) for jbf_dict in jbf_seq]
    return jbf_seq

@TRANSFORMS.register_module()
class JBFDecode(PoseDecode):
    def transform(self, results: Dict) -> Dict:
        results = super().transform(results)
        jbfs = read_jbf_seq(results['jbf_path'])
        results['total_frames'] = len(jbfs)

        if 'frame_inds' not in results:
            results['frame_inds'] = np.arange(results['total_frames'])
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])
        offset = results.get('offset', 0)
        frame_inds = results['frame_inds'] + offset

        jbfs = np.stack(jbfs, axis=0).astype(np.float32) / 255.0
        jbfs = jbfs.transpose(0, 2, 3, 1)
        jbfs = jbfs[frame_inds, ...]

        results['imgs'] = jbfs
        return results
    
@TRANSFORMS.register_module()
class GenerateJBFTarget(GeneratePoseTarget):
    def __init__(self,
                 sigma: float = 0.6,
                 use_score: bool = True,
                 with_kp: bool = True,
                 with_limb: bool = False,
                 with_jmv: bool = True,
                 with_bm: bool = True,
                 with_fm: bool = True,
                 jmv_weight: float = 0.1,
                 skeletons: Tuple[Tuple[int]] = ((0, 1), (0, 2), (1, 3),
                                                 (2, 4), (0, 5), (5, 7),
                                                 (7, 9), (0, 6), (6, 8),
                                                 (8, 10), (5, 11), (11, 13),
                                                 (13, 15), (6, 12), (12, 14),
                                                 (14, 16), (11, 12)),
                 double: bool = False,
                 left_kp: Tuple[int] = (1, 3, 5, 7, 9, 11, 13, 15),
                 right_kp: Tuple[int] = (2, 4, 6, 8, 10, 12, 14, 16),
                 left_limb: Tuple[int] = (0, 2, 4, 5, 6, 10, 11, 12),
                 right_limb: Tuple[int] = (1, 3, 7, 8, 9, 13, 14, 15),
                 scaling: float = 1.) -> None:

        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.double = double

        # an auxiliary const
        self.eps = 1e-4

        self.left_kp = left_kp
        self.right_kp = right_kp
        self.skeletons = skeletons
        self.left_limb = left_limb
        self.right_limb = right_limb
        self.scaling = scaling

        self.with_jmv = with_jmv
        self.with_bm = with_bm
        self.with_fm = with_fm
        self.jmv_weight = jmv_weight

    def transform(self, results: Dict) -> Dict:
        if 'imgs' not in results:
            return super().transform(results)
        
        jbfs = results['imgs']
        J = jbfs.shape[1] - 2

        if not self.with_jmv:
            jbfs = jbfs[:, J:, ...]
            J = 0
        if self.with_bm and not self.with_fm:
            jbfs = jbfs[:, :-1, ...]
        elif not self.with_bm and self.with_fm:
            jbfs = np.concatenate([jbfs[:, :-2, ...], jbfs[:, -1:, ...]], axis=1)
        elif not self.with_bm and not self.with_fm:
            jbfs = jbfs[:, :-2, ...]

        jbfs = gaussian_filter(jbfs, sigma=[0, 0, self.sigma, self.sigma])
        jmvs = jbfs[:, :J, ...] if self.with_jmv else None
        bfms = jbfs[:, J:, :, :] if self.with_bm or self.with_fm else None

        if self.with_limb and self.with_jmv:
            lmvs = np.zeros_like(jmvs)
            for i, limb in enumerate(self.skeletons):
                lmvs[:, i, ...] = jmvs[:, limb, ...].max(axis=1)
            jmvs = lmvs

        results.pop('imgs')

        if self.with_kp or self.with_limb:
            heatmap = self.gen_an_aug(results)
        else:
            heatmap = np.zeros_like(jmvs) if self.with_jmv else None

        if self.with_jmv:
            heatmap = heatmap + jmvs * self.jmv_weight

        if self.double:
            indices = np.arange(heatmap.shape[1], dtype=np.int64)
            left, right = (self.left_kp, self.right_kp) if self.with_kp else (self.left_limb, self.right_limb)
            for l, r in zip(left, right):  # noqa: E741
                indices[l] = r
                indices[r] = l
            heatmap_flip = heatmap[..., ::-1][:, indices]
            heatmap = np.concatenate([heatmap, heatmap_flip])
        results['imgs'] = heatmap

        if self.with_bm or self.with_fm:
            if self.double:
                bfms = np.concatenate([bfms, bfms[..., ::-1]], axis=0)
            results['imgs'] = np.concatenate([results['imgs'], bfms], axis=1)

        return results
    
@TRANSFORMS.register_module()
class JBFFlip(Flip):
    def __init__(self,
                 flip_ratio=0.5,
                 direction='horizontal',
                 flip_label_map=None,
                 left_kp=None,
                 right_kp=None,
                 lazy=False):
        super().__init__(flip_ratio, direction, flip_label_map, left_kp, right_kp, lazy)

    def _flip_imgs(self, imgs, modality):
        _ = [mmcv.imflip_(img, self.direction) for img in imgs]
        lt = len(imgs)
        if modality == 'Flow':
            # The 1st frame of each 2 frames is flow-x
            for i in range(0, lt, 2):
                imgs[i] = mmcv.iminvert(imgs[i])

        imgs = np.stack(imgs, axis=0)
        indices = np.arange(imgs.shape[3], dtype=np.int64)
        left, right = (self.left_kp, self.right_kp)
        for l, r in zip(left, right):  # noqa: E741
            indices[l] = r
            indices[r] = l
        imgs = imgs[..., ::-1, :][..., indices]
        return imgs