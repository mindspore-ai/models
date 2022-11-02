# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Data Utils"""
from pathlib import Path
import logging
import numpy as np

import mindspore as ms
from mindspore.context import set_context, PYNATIVE_MODE
from src.lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from src.lib.config import cfg

set_context(mode=PYNATIVE_MODE)


class batchpad():
    """batchpad"""
    def __init__(self, cols):
        self.cols = cols

    def __call__(self, *c):
        ans = []
        assert len(self.cols) == len(c) - 1
        batch_size = len(c[0])
        # assert batch_size > 1
        for ii, key in enumerate(self.cols):
            if cfg.RPN.ENABLED and key == 'gt_boxes3d' or \
                    (cfg.RCNN.ENABLED and cfg.RCNN.ROI_SAMPLE_JIT and key in ['gt_boxes3d', 'roi_boxes3d']):
                max_gt = 1
                for k in range(batch_size):
                    max_gt = max(max_gt, c[ii][k].__len__())
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, 7),
                                            dtype=np.float32)
                for i in range(batch_size):
                    batch_gt_boxes3d[i, :c[ii][i].__len__(), :] = c[ii][i]
                ans.append(batch_gt_boxes3d)
                continue
            if isinstance(c[ii], np.ndarray):
                if batch_size == 1:
                    ans.append(c[ii][np.newaxis, ...])
                else:
                    ans.append(
                        np.concatenate([
                            c[ii][k][np.newaxis, ...]
                            for k in range(batch_size)
                        ],
                                       axis=0))
            else:
                temp = [c[ii][k] for k in range(batch_size)]
                if isinstance(c[ii][0], int):
                    temp = np.array(temp, dtype=np.int32)
                elif isinstance(c[ii][0], float):
                    temp = np.array(temp, dtype=np.float32)
                ans.append(temp)
        return tuple(ans)


def get_cols(mode="TRAIN"):
    """get columns"""
    if cfg.RPN.ENABLED:
        return [
            "sample_id", "pts_input", "pts_rect", "pts_features",
            "rpn_cls_label", "rpn_reg_label", "gt_boxes3d"
        ]
    if mode == 'TRAIN':
        if cfg.RCNN.ROI_SAMPLE_JIT:
            return [
                'sample_id', 'rpn_xyz', 'rpn_features', 'rpn_intensity',
                'seg_mask', 'roi_boxes3d', 'gt_boxes3d', 'pts_depth'
            ]
        return [
            'sample_id', 'pts_input', 'pts_features', 'cls_label',
            'reg_valid_mask', 'gt_boxes3d_ct', 'roi_boxes3d',
            'roi_size'
        ]
    return [
        'sample_id', 'pts_input', 'pts_features', 'roi_boxes3d',
        'roi_scores', 'roi_size', 'gt_boxes3d', 'gt_iou'
    ]


def create_dataloader(logger, args):
    """create dataloader"""
    DATA_PATH = (Path(__file__).parent.parent.absolute() / 'data').absolute()
    # create dataloader

    train_set = KittiRCNNDataset(
        root_dir=DATA_PATH,
        npoints=cfg.RPN.NUM_POINTS,
        split=cfg.TRAIN.SPLIT,
        mode='TRAIN',
        logger=logger,
        classes=cfg.CLASSES,
        rcnn_training_roi_dir=args.rcnn_eval_roi_dir,
        rcnn_training_feature_dir=args.rcnn_eval_feature_dir,
        gt_database_dir=args.gt_database)

    num_class = train_set.num_class
    cols = train_set.getitem_cols(0)

    train_loader = ms.dataset.GeneratorDataset(train_set,
                                               num_parallel_workers=1,
                                               column_names=cols,
                                               shuffle=True)
    train_batch_loader = train_loader.batch(args.batch_size,
                                            drop_remainder=True,
                                            num_parallel_workers=4,
                                            python_multiprocessing=True)

    if args.train_with_eval:
        test_set = KittiRCNNDataset(
            root_dir=DATA_PATH,
            npoints=cfg.RPN.NUM_POINTS,
            split=cfg.TRAIN.VAL_SPLIT,
            mode='EVAL',
            logger=logger,
            classes=cfg.CLASSES,
            rcnn_eval_roi_dir=args.rcnn_eval_roi_dir,
            rcnn_eval_feature_dir=args.rcnn_eval_feature_dir)
        test_loader = ms.dataset.GeneratorDataset(
            test_set, num_parallel_workers=args.workers, shuffle=True)
        test_loader = test_loader.batch(1,
                                        drop_remainder=True,
                                        num_parallel_workers=4)
    else:
        test_loader = None
    return train_batch_loader, test_loader, num_class


def create_dataloader_debug(logger):
    """create dataloader debug"""
    DATA_PATH = (Path(__file__).absolute().parent.parent.absolute() /
                 'data').absolute()
    # create dataloader

    train_set = KittiRCNNDataset(
        root_dir=DATA_PATH,
        npoints=cfg.RPN.NUM_POINTS,
        split=cfg.TRAIN.SPLIT,
        mode='TRAIN',
        logger=logger,
        classes=cfg.CLASSES,
        rcnn_training_roi_dir=None,
        rcnn_training_feature_dir=None,
        gt_database_dir='src/gt_database/train_gt_database_3level_Car.pkl')
    num_class = train_set.num_class

    cols = train_set.getitem_cols(0)
    train_loader = ms.dataset.GeneratorDataset(train_set,
                                               num_parallel_workers=1,
                                               column_names=cols,
                                               shuffle=True)
    train_batch_loader = train_loader.batch(8,
                                            drop_remainder=True,
                                            num_parallel_workers=4,
                                            per_batch_map=batchpad(cols=cols),
                                            python_multiprocessing=False)

    test_loader = None
    return train_batch_loader, test_loader, num_class


def demo():
    """demo"""
    t, _, _ = create_dataloader_debug(logger=logging.getLogger())
    di = t.create_dict_iterator()
    for task in di:
        for k, v in task.items():
            print(k, v.shape)
        exit()


if __name__ == "__main__":
    demo()
