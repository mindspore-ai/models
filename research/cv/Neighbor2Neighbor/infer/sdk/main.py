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
"""main"""
import argparse
import os
import time
import glob
import numpy as np
import cv2
from api.infer import SdkApi
from config import config as cfg
import PIL.Image as Image

def parser_args():
    """parser_args"""
    parser = argparse.ArgumentParser(description="Neighbor2Neighbor inference")

    parser.add_argument("--img_path",
                        type=str,
                        required=False,
                        default="../data/input/",
                        help="image directory.")
    parser.add_argument(
        "--pipeline_path",
        type=str,
        required=False,
        default="./config/neighbor2neighbor.pipeline",
        help="image file path. The default is '/infer/sdk/config/neighbor2neighbor.pipeline'. ")
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        default="dvpp",
        help=
        "rgb: high-precision, dvpp: high performance. The default is 'dvpp'.")
    parser.add_argument(
        "--infer_mode",
        type=str,
        required=False,
        default="infer",
        help=
        "infer:only infer, eval: accuracy evaluation. The default is 'infer'.")
    parser.add_argument(
        "--infer_result_dir",
        type=str,
        required=False,
        default="./infer_result",
        help=
        "cache dir of inference result. The default is './infer_result'."
    )
    arg = parser.parse_args()
    return arg

class AugmentNoise():
    '''AugmentNoise'''
    def __init__(self, style):
        if style.startswith('gauss'):
            self.params = [
                float(p) / 255.0 for p in style.replace('gauss', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):
            self.params = [
                float(p) for p in style.replace('poisson', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

    def add_noise(self, x):
        '''add_noise'''
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        if self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        if self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        assert self.style == "poisson_range"
        min_lam, max_lam = self.params
        lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
        return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)

def test(img_path):
    '''test'''
    noisetype = "gauss25"
    noise_generator = AugmentNoise(noisetype)
    out_dir = "./preresult"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    file_list = glob.glob(os.path.join(img_path, '*'))

    for file in file_list:
        # read image
        img_clean = np.array(Image.open(file), dtype='float32') / 255.0

        img_test = noise_generator.add_noise(img_clean)
        H = img_test.shape[0]
        W = img_test.shape[1]
        val_size = (max(H, W) + 31) // 32 * 32
        img_test = np.pad(img_test, [[0, val_size - H], [0, val_size - W], [0, 0]], 'reflect')

        img_clean = np.array(img_clean).astype(np.float32)
        img_test = np.array(img_test).astype(np.float32)

        # predict
        img_clean = np.expand_dims(np.transpose(img_clean, (2, 0, 1)), 0)
        img_test = np.expand_dims(np.transpose(img_test, (2, 0, 1)), 0)

        # save images
        file_name = file.split('/')[-1].split('.')[0] + ".bin"
        img_test.tofile(os.path.join(out_dir, file_name))

def process_img(img_file):
    img = np.fromfile(img_file, dtype=np.float32).reshape(768, 768, 1, 3)
    print(img.shape)
    img = img.reshape(768, 768, 3, 1)
    print(img.shape)
    return img


def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):
    """resize a rectangular image to a padded rectangular"""
    shape = img.shape[:2]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (width - new_shape[0]) / 2
    dh = (height - new_shape[1]) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, dw, dh


def image_inference(pipeline_path, stream_name, img_dir, result_dir):
    """image_inference"""
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)
    print(stream_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    img_data_plugin_id = 0

    print("\nBegin to inference for {}.\n".format(img_dir))
    file_list = os.listdir(img_dir)
    total_len = len(file_list)
    for img_id, file_name in enumerate(file_list):
        file_path = os.path.join(img_dir, file_name)
        save_path = os.path.join(result_dir, "{}_0.bin".format(os.path.splitext(file_name)[0]))
        print(save_path)
        img_np = process_img(file_path)
        img_shape = img_np.shape
        sdk_api.send_img_input(stream_name,
                               img_data_plugin_id, "appsrc0",
                               img_np.tobytes(), img_shape)
        start_time = time.time()
        result = sdk_api.get_result(stream_name)
        end_time = time.time() - start_time
        with open(save_path, "wb") as fp:
            fp.write(result)
            print(
                f"End-2end inference, file_name: {file_path},"
                f"{img_id + 1}/{total_len}, elapsed_time: {end_time}.\n"
            )


if __name__ == "__main__":
    args = parser_args()
    test(args.img_path)
    bin_path = "./preresult"
    image_inference(args.pipeline_path, cfg.STREAM_NAME.encode("utf-8"), bin_path, args.infer_result_dir)
