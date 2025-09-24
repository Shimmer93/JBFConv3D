# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple, Union

import numpy as np
import scipy
import mmcv
from mmcv.transforms import BaseTransform
from packaging import version as pv
from scipy.stats import mode
from scipy.ndimage import gaussian_filter

from mmaction.registry import TRANSFORMS
from mmaction.utils import read_jbf_seq
from .processing import Flip
from .pose_transforms import PoseDecode, GeneratePoseTarget, MMCompact

if pv.parse(scipy.__version__) < pv.parse('1.11.0'):
    get_mode = mode
else:
    from functools import partial
    get_mode = partial(mode, keepdims=True)

@TRANSFORMS.register_module()
class ReadJBF(BaseTransform):
    def __init__(self):
        pass

    def transform(self, results: Dict) -> Dict:
        num_maps = results['keypoint'].shape[-2] + 2
        jbfs = read_jbf_seq(results['jbf_path'], num_maps)
        jbfs = np.stack(jbfs, axis=0).astype(np.float32) / 255.0
        jbfs = jbfs.transpose(0, 2, 3, 1)
        jmvs = jbfs[..., :-2]
        bfms = jbfs[..., -2:]

        cur_num_person = results['keypoint'].shape[0]
        keypoint_scale = 1 / (np.sqrt(np.sum(jmvs, axis=(1,2)))[None, ..., None] + 1e-6)
        keypoint_scale = np.repeat(keypoint_scale, repeats=cur_num_person, axis=0)

        results['keypoint_scale'] = keypoint_scale
        results['imgs'] = bfms

        return results

@TRANSFORMS.register_module()
class JBFDecode(PoseDecode):
    def transform(self, results: Dict) -> Dict:
        results = super().transform(results)
        
        # num_maps = results['keypoint'].shape[-2] + 2
        # jbfs = read_jbf_seq(results['jbf_path'], num_maps)
        bmvs = results['imgs']
        results['total_frames'] = len(results['imgs'])

        if 'frame_inds' not in results:
            results['frame_inds'] = np.arange(results['total_frames'])
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])
        offset = results.get('offset', 0)
        frame_inds = results['frame_inds'] + offset

        # jbfs = np.stack(jbfs, axis=0).astype(np.float32) / 255.0
        # jbfs = jbfs.transpose(0, 2, 3, 1)
        bmvs = bmvs[frame_inds, ...]

        results['imgs'] = bmvs
        return results
    
@TRANSFORMS.register_module()
class GenerateJBFTarget(GeneratePoseTarget):
    def __init__(self,
                 sigma: float = 0.6,
                 use_score: bool = True,
                 with_kp: bool = True,
                 with_limb: bool = False,
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

        self.with_bm = with_bm
        self.with_fm = with_fm
        self.jmv_weight = jmv_weight

    def transform(self, results: Dict) -> Dict:
        if 'imgs' not in results:
            return super().transform(results)
        
        bfms = results['imgs']
        bfms = np.stack(bfms, axis=0).transpose(0, 3, 1, 2)

        if self.with_bm and not self.with_fm:
            bfms = bfms[:, :-1, ...]
        elif not self.with_bm and self.with_fm:
            bfms = bfms[:, 1:, ...]

        bfms = gaussian_filter(bfms, sigma=[0, 0, self.sigma, self.sigma])

        results.pop('imgs')

        if self.with_kp or self.with_limb:
            heatmap = self.gen_an_aug(results)
        else:
            heatmap = None

        if self.double and heatmap is not None:
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
            if heatmap is None:
                results['imgs'] = bfms
            else:
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

@TRANSFORMS.register_module()
class JBFCompactResizePad(MMCompact):
    def __init__(self,
                 padding: float = 0.25,
                 threshold: int = 10,
                 hw_ratio: Union[float, Tuple[float]] = 1,
                 allow_imgpad: bool = True,
                 interpolation: str = 'bilinear') -> None:
        super().__init__(padding, threshold, hw_ratio, allow_imgpad)
        self.interpolation = interpolation

    def _resize_jbfs(self, jbfs, jbf_shapes):
        return [
            mmcv.imresize(jbf, (int(shape[0]), int(shape[1])), interpolation=self.interpolation)
            for jbf, shape in zip(jbfs, jbf_shapes)
        ]
    
    def _pad_jbfs(self, jbfs, jbf_boxes, sample_box):
        min_x, min_y, max_x, max_y = sample_box
        pad_jbfs = []
        for jbf, jbf_box in zip(jbfs, jbf_boxes):
            jbf_min_x, jbf_min_y, jbf_max_x, jbf_max_y = jbf_box
            pad_u = int(jbf_min_y - min_y)
            pad_d = int(max_y - jbf_max_y)
            pad_l = int(jbf_min_x - min_x)
            pad_r = int(max_x - jbf_max_x)

            if pad_u < 0:
                jbf = jbf[-pad_u:, :, :]
                pad_u = 0
            if pad_d < 0:
                jbf = jbf[:pad_d, :, :]
                pad_d = 0
            if pad_l < 0:
                jbf = jbf[:, -pad_l:, :]
                pad_l = 0
            if pad_r < 0:
                jbf = jbf[:, :pad_r, :]
                pad_r = 0

            pad_jbf = np.pad(jbf, ((pad_u, pad_d), (pad_l, pad_r), (0, 0)))
            pad_jbfs.append(pad_jbf)
        return pad_jbfs

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`MMCompact`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        img_shape = results['img_shape']
        kp = results['keypoint']
        # Make NaN zero
        kp[np.isnan(kp)] = 0.
        min_x, min_y, max_x, max_y = self._get_box(kp, img_shape)

        kp_x, kp_y = kp[..., 0], kp[..., 1]
        kp_x[kp_x != 0] -= min_x
        kp_y[kp_y != 0] -= min_y

        new_shape = (max_y - min_y, max_x - min_x)
        results['img_shape'] = new_shape
        jbf_boxes = results['jbf_boxes'].astype(np.int32)
        # jbf_boxes = (jbf_boxes + 1) * np.array([[new_shape[1], new_shape[0], new_shape[1], new_shape[0]]]) / 2
        # jbf_boxes = jbf_boxes.astype(np.int32)
        jbf_shapes = jbf_boxes[..., [3, 2]] - jbf_boxes[..., [1, 0]]
        results['imgs'] = self._resize_jbfs(results['imgs'], jbf_shapes)
        results['imgs'] = self._pad_jbfs(results['imgs'], jbf_boxes, (min_x, min_y, max_x, max_y))

        return results