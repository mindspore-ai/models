# Copyright 2023 Huawei Technologies Co., Ltd
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
# =======================================================================================
""" server client entrance module """
import numpy as np
from mindspore_serving.client import Client
from yolox.paraser import config

idx2label = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
             'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
             'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
             'skateboard',
             'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
             'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
             'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
             'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
             'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def read_image(img_path):
    with open(img_path, 'rb') as f:
        img = f.read()
    return img


def serving_predict(cfg):
    client = Client("%s:%s" % (cfg.ip, cfg.port), cfg.servable_name, "inference")
    instances = []
    img = read_image(cfg.infer_img)
    instances.append({'image': img, 'input_size': np.array(config.input_size), 'nms_thre': config.nms_thre,
                      'conf_thre': config.conf_thre, 'num_classes': config.num_classes})

    result = client.infer(instances)[0]
    cls = result['cls'][0]
    scores = result['scores'][0]
    bboxes = result['bboxes'][0]

    function = lambda x: idx2label[int(x)]
    cls = list(map(function, cls))
    print('scores: ', scores)
    print(' ')
    print('bboxes: ', bboxes)
    print(' ')
    print('cls: ', cls)


if __name__ == '__main__':
    serving_predict(config)
