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

from src.utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr
from src.model_utils.config import config

def postprocess():
    cfg = config

    imgs = os.listdir(cfg.image_path)
    imgNum = len(imgs)
    average = 0
    for i in range(imgNum):
        s = imgs[i]
        target = s.split('.')[0]
        img_type = s.split('.')[1]
        image = pil_image.open(cfg.image_path + "/" + imgs[i]).convert('RGB')

        image_width = cfg.image_width
        image_height = cfg.image_height
        image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)

        image = np.array(image).astype(np.float32)
        ycbcr = convert_rgb_to_ycbcr(image)

        y = ycbcr[..., 0]
        y /= 255.
        y = np.expand_dims(np.expand_dims(y, 0), 0)
        preds = np.fromfile(cfg.predict_path + "/" + target + ".output", np.float32)
        preds.resize(1, 1, image_width, image_height)
        psnr = calc_psnr(y, preds)
        psnr = psnr.item(0)
        print(target + " PSNR: %.4f" % psnr)
        average = average + psnr
        preds = np.multiply(preds, 255.0)
        preds = preds.squeeze(0).squeeze(0)
        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        output.save(cfg.result_path + "/" + target + "_srcnn." + img_type)
    print("Average PSNR: %.4f" % (average / imgNum))

if __name__ == '__main__':
    postprocess()
