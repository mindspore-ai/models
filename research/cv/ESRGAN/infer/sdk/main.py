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
'''sdk main'''

import argparse
import os
import time

import cv2
import numpy as np

from api.infer import SdkApi
from config import config as cfg
from src.util.util import get, imfrombytes, img2tensor


object_imageSize = 800

def parser_args():
    '''set parameter'''
    parser = argparse.ArgumentParser(description="esrgan inference")
    parser.add_argument("--img_path",
                        type=str,
                        required=True,
                        help="image directory.")
    parser.add_argument(
        "--pipeline_path",
        type=str,
        required=False,
        default="config/esrgan.pipeline",
        help="pipeline path. The default is 'config/esrgan.pipeline'. ")
    parser.add_argument(
        "--infer_result_dir",
        type=str,
        required=False,
        default="./infer_result/",
        help="cache dir of inference result. The default is './infer_result/'.")

    return parser.parse_args()

def img_convert(img_np, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert image numpy arrays for 310."""
    result = []
    img_np = np.clip(img_np, *min_max)
    img_np = (img_np - min_max[0]) / (min_max[1] - min_max[0])
    img_np = img_np.transpose(1, 2, 0)
    if img_np.shape[2] == 1:  # gray image
        img_np = np.squeeze(img_np, axis=2)
    else:
        if rgb2bgr:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    if out_type == np.uint8:
        # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
        img_np = (img_np * 255.0).round()
    img_np = img_np.astype(out_type)
    result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def unpadding(img, target_shape):
    """unpadding image for 310."""
    a, b = target_shape[0], target_shape[1]
    img_h, img_w, _ = img.shape
    if img_h > a:
        img = img[:a, :, :]
    if img_w > b:
        img = img[:, :b, :]
    return img

def read_img(img_path):
    img_bytes = get(img_path)
    img_lq = imfrombytes(img_bytes, float32=True)
    img = img2tensor(img_lq, pad=True)
    img = img.transpose(2, 0, 1).reshape((200, 200, 3))
    return img

def image_inference(pipeline_path, stream_name, img_dir, result_dir,
                    replace_last):
    '''sdk process'''
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)

    print(result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    img_data_plugin_id = 0
    images = os.listdir(img_dir)
    total_len = len(images)
    for ind, file_name in enumerate(images):
        file_path = os.path.join(img_dir, file_name)
        print("file_path is ", file_path)
        img_np = read_img(file_path)
        img_shape = img_np.shape

        sdk_api.send_img_input(stream_name,
                               img_data_plugin_id, "appsrc0",
                               img_np.tobytes(), img_shape)
        start_time = time.time()
        result = sdk_api.get_result(stream_name)
        end_time = time.time() - start_time

        print(result_dir)
        save_path = os.path.join(result_dir, file_name)
        save_path = save_path.replace(
            '.' + save_path.split('.')[-1], '_0.bin')
        print('save_path is ', save_path)
        with open(save_path, "wb") as fp:
            fp.write(result)
            print(
                f"End-2end inference, file_name: {file_path}, {ind + 1}/{total_len}, elapsed_time: {end_time}.\n"
            )

if __name__ == "__main__":
    args = parser_args()

    stream_name1 = cfg.STREAM_NAME.encode("utf-8")
    image_inference(args.pipeline_path, stream_name1, args.img_path,
                    args.infer_result_dir, True)
