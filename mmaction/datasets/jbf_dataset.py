# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, Dict, List, Optional, Union

from mmaction.registry import DATASETS
from .pose_dataset import PoseDataset


@DATASETS.register_module()
class JBFDataset(PoseDataset):
    def __init__(self,
                 ann_file: str,
                 jbf_dir: str,
                 pipeline: List[Union[Dict, Callable]],
                 split: Optional[str] = None,
                 valid_ratio: Optional[float] = None,
                 box_thr: float = 0.5,
                 **kwargs) -> None:
        super().__init__(ann_file, pipeline, split, valid_ratio, box_thr, **kwargs)
        self.jbf_dir = jbf_dir

    def load_data_list(self) -> List[Dict]:
        data_list = super().load_data_list()
        for data in data_list:
            data['jbf_path'] = osp.join(self.jbf_dir, data['frame_dir'] + '.npy')
        return data_list