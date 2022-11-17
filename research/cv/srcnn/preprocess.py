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
"""srcnn test"""
import os

import PIL.Image as pil_image
import numpy as np

from src.utils import convert_rgb_to_ycbcr
from src.model_utils.config import config

def preprocess():
    cfg = config
    imgs = os.listdir(cfg.image_path)
    imgNum = len(imgs)
    for i in range(imgNum):
        s = imgs[i]
        target = s.split('.')[0]
        image = pil_image.open(cfg.image_path + "/" + imgs[i]).convert('RGB')

        image_width = cfg.image_width
        image_height = cfg.image_height
        image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)

        image = np.array(image).astype(np.float32)
        ycbcr = convert_rgb_to_ycbcr(image)

        y = ycbcr[..., 0]
        y /= 255.
        y = np.expand_dims(np.expand_dims(y, 0), 0)
        y.tofile(cfg.output_path + '/' + target + ".bin")

if __name__ == '__main__':
    preprocess()
