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

import os

import numpy as np
import cv2
from tqdm import tqdm

from config import check_config
from src.dataset.mpii import MPII
from src.dataset.pose import Batch


def reshape_image(cfg=None, batch=None):
    """
    reshape image
    """
    test_shape = (cfg.image_width, cfg.image_height)
    img = batch[Batch.inputs].transpose([1, 2, 0])
    img_shape = img.shape
    ratio = (test_shape[0] / img_shape[1], test_shape[1] / img_shape[0])
    img = cv2.resize(img, test_shape, interpolation=cv2.INTER_CUBIC)
    return np.expand_dims(img.transpose([2, 0, 1]), axis=0), ratio

def data_to_bin(cfg=None):
    """
    convert images to .bin
    Args:
        cfg: config
    """
    dataset = MPII(cfg)
    dataset.set_mirror(False)

    num_images = len(dataset)
    ratios = []
    path = "./mpii/bin/"
    if not os.path.exists(path):
        os.makedirs(path)
    for i in tqdm(range(num_images)):
        batch = dataset.get_item(i)
        img, ratio = reshape_image(cfg, batch)
        img.tofile(path + str(i) + ".bin")
        ratios.append(ratio)
    np.save("./mpii/ratios.npy", np.array(ratios, dtype=np.float32))


def main(cfg=None):
    cfg = check_config(cfg, 0)
    cfg.model_arts.IS_MODEL_ARTS = False

    cfg.train = False
    data_to_bin(cfg)

if __name__ == '__main__':
    main("./config/mpii_eval_ascend.yaml")
