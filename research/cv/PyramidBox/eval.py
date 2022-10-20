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

# This file was copied from project [ZhaoWeicheng][Pyramidbox.pytorch]

import os
import argparse
from PIL import Image
from mindspore import Tensor, context
from mindspore import load_checkpoint, load_param_into_net
import numpy as np

from src.config import cfg
from src.pyramidbox import build_net
from src.augmentations import to_chw_bgr
from src.prior_box import PriorBox
from src.detection import Detect
from src.evaluate import evaluation

parser = argparse.ArgumentParser(description='PyramidBox Evaluatuon on Fddb')
parser.add_argument('--model', type=str, default='checkpoints/pyramidbox.pth', help='trained model')
parser.add_argument('--thresh', default=0.1, type=float, help='Final confidence threshold')
args = parser.parse_args()

FDDB_IMG_DIR = cfg.FACE.FDDB_DIR
FDDB_FOLD_DIR = os.path.join(FDDB_IMG_DIR, 'FDDB-folds')
FDDB_OUT_DIR = 'FDDB-out'

if not os.path.exists(FDDB_OUT_DIR):
    os.mkdir(FDDB_OUT_DIR)

def detect_face(net_, img_, thresh):
    x = to_chw_bgr(img_).astype(np.float32)
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = Tensor(x)[None, :, :, :]
    size = x.shape[2:]

    loc, conf, feature_maps = net_(x)

    prior_box = PriorBox(cfg, feature_maps, size, 'test')
    default_priors = prior_box.forward()

    detections = Detect(cfg).detect(loc, conf, default_priors)

    scale = np.array([img_.shape[1], img_.shape[0], img_.shape[1], img_.shape[0]])
    bboxes = []
    for i in range(detections.shape[1]):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            box = []
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).astype(np.int32)

            j += 1
            box += [pt[0], pt[1], pt[2] - pt[0], pt[3] - pt[1], score]
            bboxes += [box]

    return bboxes

if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE)
    net = build_net('test', cfg.NUM_CLASSES)
    params = load_checkpoint(args.model)
    load_param_into_net(net, params)
    net.set_train(False)

    print("Start detecting FDDB images")
    for index in range(1, 11):
        if not os.path.exists(os.path.join(FDDB_OUT_DIR, str(index))):
            os.mkdir(os.path.join(FDDB_OUT_DIR, str(index)))
        print(f"Detecting folder {index}")
        file_path = os.path.join(cfg.FACE.FDDB_DIR, 'FDDB-folds', 'FDDB-fold-%02d.txt' % index)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                image_path = os.path.join(cfg.FACE.FDDB_DIR, line) + '.jpg'
                img = Image.open(image_path)
                if img.mode == 'L':
                    img = img.convert('RGB')
                img = np.array(img)
                line = line.replace('/', '_')
                with open(os.path.join(FDDB_OUT_DIR, str(index), line + '.txt'), 'w') as w:
                    w.write(line)
                    w.write('\n')
                    boxes = detect_face(net, img, args.thresh)
                    if not boxes is None:
                        w.write(str(len(boxes)))
                        w.write('\n')
                        for box_ in boxes:
                            w.write(f'{int(box_[0])} {int(box_[1])} {int(box_[2])} {int(box_[3])} {box_[4]}\n')
    print("Detection Done!")
    print("Start evluation!")

    evaluation(FDDB_OUT_DIR, FDDB_FOLD_DIR)
