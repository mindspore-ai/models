# coding=utf-8

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
import PIL.Image as pil_image

def convert_rgb_to_y(img):
    if isinstance(img, np.ndarray):
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    raise Exception('Unknown Type', type(img))

def calc_psnr(img1, img2):
    return 10. * np.log10(1. / np.mean((img1 - img2) ** 2))

scale = 2
def run():
    image_path = '../data/data/Set5'
    result_path = '../mxbase/result'
    if os.path.isdir(image_path):
        img_infos = os.listdir(image_path)
    for i in range(len(img_infos)):
        img_infos[i] = os.path.splitext(img_infos[i])[0]
    psnr1 = []
    for i in range(len(img_infos)):
        hr = pil_image.open(image_path + "/" + img_infos[i] + '.png').convert('RGB')
        result = open(result_path + "/" + img_infos[i] + '.bin')
        hr_width = 512
        hr_height = 512
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        y0 = convert_rgb_to_y(hr)
        y0 /= 255.
        y0 = np.expand_dims(np.expand_dims(y0, 0), 0)
        y_predict = np.fromfile(result, dtype=np.float32)
        y_predict = y_predict.reshape(1, 512, 512)
        y_predict = np.expand_dims(y_predict, 0)
        psnr = calc_psnr(y0, y_predict)
        psnr = psnr.item(0)
        psnr1.append(psnr)
    print(psnr1)
    psnr = np.sum(psnr1)/len(img_infos)
    print("=======================================")
    print("PSNR: %.4f" % psnr)
    print("=======================================")
if __name__ == '__main__':
    run()
    