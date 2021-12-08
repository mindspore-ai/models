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
"""start to infer"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from PIL import Image

from api.infer import SdkApi
from config import config as cfg

def parser_args():
    """parse the input"""
    parser = argparse.ArgumentParser(description="textfusnet inference")

    parser.add_argument("--img_path",
                        type=str,
                        required=True,
                        help="image directory.")
    parser.add_argument(
        "--pipeline_path",
        type=str,
        required=False,
        default="config/textfusenet.pipeline",
        help="image file path. The default is 'config/textfusenet.pipeline'. ")
    parser.add_argument(
        "--infer_result_dir",
        type=str,
        required=False,
        default="../data/infer_result/",
        help=
        "cache dir of inference result. The default is '../data/infer_result'."
    )

    args_ = parser.parse_args()
    return args_


def get_img_metas(file_name):
    """get image metas"""
    img = Image.open(file_name)
    img_size = img.size

    org_width, org_height = img_size
    resize_ratio = min(cfg.MODEL_WIDTH * 1.0 / org_width, cfg.MODEL_HEIGHT * 1.0 / org_height)

    img_metas = np.array([img_size[1], img_size[0]] +
                         [resize_ratio, resize_ratio])
    return img_metas


def convert_result(result, img_metas):
    """convert the result"""
    if result is None:
        return result
    bboxes = result.get("MxpiObject")
    for bbox in bboxes:
        bbox['x0'] = max(min(bbox['x0']/img_metas[3], img_metas[1]-1), 0)
        bbox['x1'] = max(min(bbox['x1']/img_metas[3], img_metas[1]-1), 0)
        bbox['y0'] = max(min(bbox['y0']/img_metas[2], img_metas[0]-1), 0)
        bbox['y1'] = max(min(bbox['y1']/img_metas[2], img_metas[0]-1), 0)
        mask_height = bbox['imageMask']['shape'][1]
        mask_width = bbox['imageMask']['shape'][0]
        print(f'Image_metas:{img_metas}, shape:({mask_height}, {mask_width})')
    return result


def process_img(img_file):
    """process image"""
    img = cv2.imread(img_file)
    img_h, img_w, _ = img.shape
    scale_w = cfg.MODEL_WIDTH * 1.0 / img_w
    scale_h = cfg.MODEL_HEIGHT * 1.0 / img_h
    ratio = scale_w
    if ratio > scale_h:
        ratio = scale_h
    new_w = int(img_w * ratio + 0.5)
    new_h = int(img_h * ratio + 0.5)
    model_img = cv2.resize(img, (new_w, new_h))
    pad_img = np.zeros(
        (cfg.MODEL_HEIGHT, cfg.MODEL_WIDTH, 3)).astype(model_img.dtype)
    right = cfg.MODEL_WIDTH - new_w
    bottom = cfg.MODEL_HEIGHT - new_h
    cv2.copyMakeBorder(model_img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, pad_img, 0)
    pad_img.astype(np.float16)
    return pad_img


def image_inference(pipeline_path, stream_name, img_dir, result_dir, replace_last):
    """image inference"""
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    img_data_plugin_id = 0
    img_metas_plugin_id = 1
    print(f"\nBegin to inference for {img_dir}.\n\n")

    file_list = os.listdir(img_dir)
    total_len = len(file_list)
    for img_id, file_name in enumerate(file_list):
        if not file_name.lower().endswith((".jpg", "jpeg")):
            continue
        file_path = os.path.join(img_dir, file_name)
        save_path = os.path.join(result_dir,
                                 f"{os.path.splitext(file_name)[0]}.json")
        if not replace_last and os.path.exists(save_path):
            print(
                f"The infer result json({save_path}) has existed, will be skip."
            )
            continue

        img_np = process_img(file_path)
        sdk_api.send_img_input(stream_name,
                               img_data_plugin_id, "appsrc0",
                               img_np.tobytes(), img_np.shape)

        # set image data
        img_metas = get_img_metas(file_path).astype(np.float16)
        sdk_api.send_tensor_input(stream_name, img_metas_plugin_id,
                                  "appsrc1", img_metas.tobytes(), [1, 4],
                                  cfg.TENSOR_DTYPE_FLOAT16)

        start_time = time.time()
        result = sdk_api.get_result(stream_name)
        end_time = time.time() - start_time

        result = convert_result(result, img_metas)

        with open(save_path, "w") as fp:
            fp.write(json.dumps(result))
        print(
            f"End-2end inference, file_name: {file_path}, {img_id + 1}/{total_len}, elapsed_time: {end_time}.\n"
        )


if __name__ == "__main__":
    args = parser_args()

    stream_name_textfusenet = cfg.STREAM_NAME.encode("utf-8")
    image_inference(args.pipeline_path, stream_name_textfusenet, args.img_path,
                    args.infer_result_dir, True)
