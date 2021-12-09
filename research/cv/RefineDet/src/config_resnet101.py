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
"""Basic config parameters for RefineDet models."""
from easydict import EasyDict as ed

config_320 = ed({
    "model": "refinedet_resnet101",
    "img_shape": [320, 320],
    "num_ssd_boxes": -1,
    "match_threshold": 0.5,
    "nms_threshold": 0.6,
    "min_score": 0.1,
    "max_boxes": 100,

    # learing rate settings
    "lr_init": 0.001,
    "lr_end_rate": 0.001,
    "warmup_epochs": 2,
    "momentum": 0.9,
    "weight_decay": 1.5e-4,

    # network
    # vgg16 config
    "num_default": [3, 3, 3, 3],
    "extra_arm_channels": [512, 1024, 2048, 512],
    "extra_odm_channels": [256, 256, 256, 256],
    "L2normalizations": [10, 8, -1, -1],
    "arm_source": ["b4", "b5", "fc7", "b6_2"], # four source layers, last one is the end of backbone

    # box utils config
    "feature_size": [40, 20, 10, 5],
    "min_scale": 0.2,
    "max_scale": 0.95,
    "aspect_ratios": [(), (2,), (2,), (2,)],
    "steps": (8, 16, 32, 64),
    "prior_scaling": (0.1, 0.2),
    "gamma": 2.0,
    "alpha": 0.75,

    # `mindrecord_dir` and `coco_root` are better to use absolute path.
    "feature_extractor_base_param": "",
    "pretrain_vgg_bn": False,
    "checkpoint_filter_list": ['multi_loc_layers', 'multi_cls_layers'],
    "mindrecord_dir": "./data/MindRecord",
    "coco_root": "./data/COCO2017",
    "train_data_type": "train2017",
    # The annotation.json position of voc validation dataset.
    "voc_json": "annotations/voc_instances_val.json",
    # voc original dataset.
    "voc_root": "",
    "voc_test_root": "./data/voc_test",
    "voc0712_root": "./data/VOC0712",
    "voc0712plus_root": "./data/VOC0712Plus",
    # if coco or voc used, `image_dir` and `anno_path` are useless.
    "image_dir": "",
    "anno_path": "",
    "val_data_type": "val2017",
    "instances_set": "annotations/instances_{}.json",
    "voc_classes": ('background', 'aeroplane', 'bicycle', 'bird',
                    'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
    "voc_num_classes": 21,
    "coco_classes": ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                     'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                     'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                     'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                     'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                     'kite', 'baseball bat', 'baseball glove', 'skateboard',
                     'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                     'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                     'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                     'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                     'refrigerator', 'book', 'clock', 'vase', 'scissors',
                     'teddy bear', 'hair drier', 'toothbrush'),
    "coco_num_classes": 81,
    "classes": (),
    "num_classes": ()
})

config_512 = ed({
    "model": "refinedet_resnet101",
    "img_shape": [512, 512],
    "num_ssd_boxes": -1,
    "match_threshold": 0.5,
    "nms_threshold": 0.6,
    "min_score": 0.1,
    "max_boxes": 100,

    # learing rate settings
    "lr_init": 0.001,
    "lr_end_rate": 0.001,
    "warmup_epochs": 2,
    "momentum": 0.9,
    "weight_decay": 1.5e-4,

    # network
    # vgg16 config
    "num_default": [3, 3, 3, 3],
    "extra_arm_channels": [512, 1024, 2048, 512],
    "extra_odm_channels": [256, 256, 256, 256],
    "L2normalizations": [10, 8, -1, -1],
    "arm_source": ["b4", "b5", "fc7", "b6_2"], # four source layers, last one is the end of backbone

    # box utils config
    "feature_size": [64, 32, 16, 8],
    "min_scale": 0.2,
    "max_scale": 0.95,
    "aspect_ratios": [(), (2,), (2,), (2,)],
    "steps": (8, 16, 32, 64),
    "prior_scaling": (0.1, 0.2),
    "gamma": 2.0,
    "alpha": 0.75,

    # `mindrecord_dir` and `coco_root` are better to use absolute path.
    "feature_extractor_base_param": "",
    "pretrain_vgg_bn": False,
    "checkpoint_filter_list": ['multi_loc_layers', 'multi_cls_layers'],
    "mindrecord_dir": "./data/MindRecord",
    "coco_root": "./data/COCO2017",
    "train_data_type": "train2017",
    # The annotation.json position of voc validation dataset.
    "voc_json": "annotations/voc_instances_val.json",
    # voc original dataset.
    "voc_root": "",
    "voc_test_root": "./data/voc_test",
    "voc0712_root": "./data/VOC0712",
    "voc0712plus_root": "./data/VOC0712Plus",
    # if coco or voc used, `image_dir` and `anno_path` are useless.
    "image_dir": "",
    "anno_path": "",
    "val_data_type": "val2017",
    "instances_set": "annotations/instances_{}.json",
    "voc_classes": ('background', 'aeroplane', 'bicycle', 'bird',
                    'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
    "voc_num_classes": 21,
    "coco_classes": ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                     'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                     'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                     'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                     'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                     'kite', 'baseball bat', 'baseball glove', 'skateboard',
                     'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                     'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                     'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                     'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                     'refrigerator', 'book', 'clock', 'vase', 'scissors',
                     'teddy bear', 'hair drier', 'toothbrush'),
    "coco_num_classes": 81,
    "classes": (),
    "num_classes": ()
})
