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
'''
This file evaluates the model used.
'''
from __future__ import division

import argparse
import os
import numpy as np

from src.config import config
from src.utils.transforms import flip_back
from src.utils.coco import evaluate
from src.utils.inference import get_final_preds

flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
              [9, 10], [11, 12], [13, 14], [15, 16]]


def parse_args():
    '''
    parse_args
    '''
    parser = argparse.ArgumentParser(description='get_acc')
    parser.add_argument('--result_path', required=True,
                        default=None, help='Location of result.')
    parser.add_argument('--data_path', required=True,
                        default=None, help='Location of .npy file.')
    opt_args = parser.parse_args()
    return opt_args


def get_acc(cfg, result_path, npy_path):
    '''
    get_acc
    '''
    centers = np.load(os.path.join(npy_path, "centers.npy"))
    scales = np.load(os.path.join(npy_path, "scales.npy"))
    scores = np.load(os.path.join(npy_path, "scores.npy"))
    ids = np.load(os.path.join(npy_path, "ids.npy"))
    num_samples = len(os.listdir(result_path)) // 2
    all_preds = np.zeros((num_samples, 17, 3),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 2))
    image_id = []
    idx = 0
    print(num_samples)
    out_shape = [1, 17, 64, 48]
    for i in range(num_samples):
        f1 = os.path.join(result_path, str(i) + "_0.bin")
        output = np.fromfile(f1, np.float32).reshape(out_shape)
        if cfg.TEST.FLIP_TEST:
            f2 = os.path.join(result_path, "flipped" + str(i) + "_0.bin")
            output_flipped = np.fromfile(f2, np.float32).reshape(out_shape)
            output_flipped = flip_back(output_flipped, flip_pairs)
            if cfg.TEST.SHIFT_HEATMAP:
                output_flipped[:, :, :, 1:] = \
                    output_flipped.copy()[:, :, :, 0:-1]

            output = (output + output_flipped) * 0.5

        c = centers[i]
        s = scales[i]
        score = scores[i]
        file_id = list(ids[i])

        preds, maxvals = get_final_preds(output.copy(), c, s)
        num_images, _ = preds.shape[:2]
        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        all_boxes[idx:idx + num_images, 0] = np.prod(s * 200, 1)
        all_boxes[idx:idx + num_images, 1] = score
        image_id.extend(file_id)
        idx += num_images

    output_dir = "result/"
    ann_path = config.DATASET.ROOT + config.DATASET.TEST_JSON
    print(all_preds[:idx].shape, all_boxes[:idx].shape, len(image_id))
    _, perf_indicator = evaluate(
        cfg, all_preds[:idx], output_dir, all_boxes[:idx], image_id, ann_path)
    print("AP:", perf_indicator)
    return perf_indicator


if __name__ == '__main__':
    args = parse_args()
    get_acc(config, args.result_path, args.data_path)
