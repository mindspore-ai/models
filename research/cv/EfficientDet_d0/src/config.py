# Copyright 2021 Huawei Technologies Co., Ltd
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
""" config """
from easydict import EasyDict as ed

config = ed({
    "workers": 24,
    "batch_norm_momentum": 0.99,
    "batch_norm_epsilon": 1e-3,
    "dropout_rate": 0.2,
    "drop_connect_rate": 0.2,
    "width_coefficient": 1.0,
    "depth_coefficient": 1.0,
    "depth_divisor": 8,

    "img_shape": [512, 512],
    "num_efficient_boxes": 128,
    "match_thershold": 0.5,
    "nms_thershold": 0.2,
    "min_score": 0.1,
    "max_boxes": 100,

    # settings
    "global_step": 0,
    "momentum": 0.9,
    "weight_decay": 5e-4,

    # network
    "num_default": [9, 9, 9, 9, 9],
    "extras_out_channels": [256, 256, 256, 256, 256],
    "feature_size": [75, 38, 19, 10, 5],
    "aspect_ratios": [(0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0)],
    "steps": (8, 16, 32, 64, 128),
    "anchor_size": (32, 64, 128, 256, 512),
    "prior_scaling": (0.1, 0.2),
    "gamma": 2.0,
    "alpha": 0.75,

    # mean and std in RGB order, actually this part should remain unchanged as long as your dataset is similar to coco.
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],

    # this is coco anchors, change it if necessary
    "anchors_scales": '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]',
    "anchors_ratios": '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]',

    "mindrecord_dir": "/data/efficientdet_ch/MindRecordImg",
    "coco_root": "/data/coco2017/",
    "train_data_type": "train2017",
    "val_data_type": "val2017",
    "instances_set": "annotations/instances_{}.json",
    "coco_classes": ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
                     'truck', 'boat', 'traffic light', 'fire hydrant', '', 'stop sign',
                     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                     'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack',
                     'umbrella', '', '', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                     'skateboard', 'surfboard', 'tennis racket', 'bottle', '',
                     'wine glass', 'cup', 'fork', 'knife', 'spoon',
                     'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                     'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                     'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                     'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                     'oven', 'toaster', 'sink', 'refrigerator', '', 'book', 'clock', 'vase',
                     'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
    "num_classes": 90,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 5,
    "keep_checkpoint_max": 10,
    "save_checkpoint_path": "./ckpt",
    "lr": 0.012,
    "batch_size": 16,
    "loss_scale": 1,
    "epoch_size": 500
})
