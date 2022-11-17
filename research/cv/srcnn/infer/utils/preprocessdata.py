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
"""srcnn test"""
import os
import PIL.Image as pil_image
import numpy as np

def convert_rgb_to_y(img):
    if isinstance(img, np.ndarray):
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    raise Exception('Unknown Type', type(img))

def preprocess():
    imgs = os.listdir('../data/data/Set5')
    imgNum = len(imgs)
    for i in range(imgNum):
        scale = 2
        s = imgs[i]
        target = s.split('.')[0]
        hr = pil_image.open('../data/data/Set5' + "/" + imgs[i]).convert('RGB')
        hr_width = 512
        hr_height = 512
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // scale, hr_height // scale), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)

        y = convert_rgb_to_y(lr)
        y /= 255.
        y = np.expand_dims(np.expand_dims(y, 0), 0)
        y.tofile('../data/data/Set' + '/' + target + ".bin")

if __name__ == '__main__':
    preprocess()
    