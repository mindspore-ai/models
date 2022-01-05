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
"""
    postprocess
"""
import os
import cv2
import numpy as np
from src.modelarts_utils.config import config

def save_image(img, img_path):
    """Save a numpy image to the disk

    Parameters:
        img (numpy array / Tensor): image to save.
        image_path (str): the path of the image.
    """
    img = decode_image(img)

    img_pil = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path + '.jpg', img_pil * 255.0)

def decode_image(img):
    """Decode a [1, C, H, W] Tensor to image numpy array."""
    mean = 0.5
    std = 0.5

    return (img * std + mean).transpose(1, 2, 0)

if __name__ == '__main__':
    infer_output_img = config.eval_outputdir
    object_imageSize = config.img_size
    bifile_dir = config.bifile_outputdir

    for file in os.listdir(bifile_dir):
        file_name = os.path.join(bifile_dir, file)
        output = np.fromfile(file_name, np.float32).reshape(3, object_imageSize, object_imageSize)
        print(output.shape)
        save_image(output, infer_output_img + "/" + file.split('.')[0])
        print("=======image", file, "saved success=======")
    print("Generate images success!")
