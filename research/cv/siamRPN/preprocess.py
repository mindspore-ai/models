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
"""310eval preprocess"""
import os
import numpy as np
from src.config import config

def generate_anchors(total_stride, base_size, scales, ratios, score_size):
    """ generate anchors """
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4), dtype=np.float32)
    size = base_size * base_size
    count = 0
    for ratio in ratios:
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size // 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor

def generateConfigBin():
    """ get anchors and hanning for eval """
    valid_scope = 2 * config.valid_scope + 1
    anchors = generate_anchors(config.total_stride, config.anchor_base_size, config.anchor_scales,
                               config.anchor_ratios, valid_scope)

    windows = np.tile(np.outer(np.hanning(config.score_size), np.hanning(config.score_size))[None, :],
                      [config.anchor_num, 1, 1]).flatten()
    path1 = os.path.join(os.getcwd(), "ascend_310_infer", "src", "anchors.bin")
    path2 = os.path.join(os.getcwd(), "ascend_310_infer", "src", "windows.bin")
    if os.path.exists(path1):
        os.remove(path1)
    if os.path.exists(path2):
        os.remove(path2)
    anchors.tofile(path1)
    windows.tofile(path2)

if __name__ == '__main__':
    generateConfigBin()
