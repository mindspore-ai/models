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
"""
    preprocess
"""
import os
import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser('preprocess')
parser.add_argument('--image_dir', type=str, default='', help='eval data dir')
parser.add_argument('--result_dir', type=str, default='./pre_results/',
                    help='during validating, the file path of Generated image.')
args = parser.parse_args()

def get(filepath):
    """Get values according to the filepath """
    filepath = str(filepath)
    with open(filepath, 'rb') as f:
        value_buf = f.read()
    return value_buf

def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes."""
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    image = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        image = image.astype(np.float32) / 255.
    return image

def img2tensor(image, bgr2rgb=True):
    """Numpy array to tensor."""
    if image.shape[2] == 3 and bgr2rgb:
        if image.dtype == 'float64':
            image = image.astype('float32')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))
    return image

def padding(_img, target_shape):
    """padding image for 310."""
    h, w = target_shape[0], target_shape[1]
    img_h, img_w, _ = _img.shape
    dh, dw = h - img_h, w - img_w
    if dh < 0 or dw < 0:
        raise RuntimeError(f"target_shape is bigger than img.shape, {target_shape} > {_img.shape}")
    if dh != 0 or dw != 0:
        _img = np.pad(_img, ((0, dh), (0, dw), (0, 0)), "constant")
    return _img


if __name__ == '__main__':
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    images = os.listdir(args.image_dir)
    for img in images:
        image_path = os.path.join(args.image_dir, img)
        file_path = os.path.join(args.result_dir, img[ : -4] + ".bin")
        img_bytes = get(image_path)
        img = imfrombytes(img_bytes, float32=True)
        img = img2tensor(img, bgr2rgb=True)
        img = img.transpose(1, 2, 0)
        org_img = padding(img, [200, 200])
        org_img = org_img.transpose(2, 0, 1)
        img = org_img.copy()
        img.tofile(file_path)
        print("output file is ", file_path)

    print("=" * 20, "export bin files finished", "=" * 20)
