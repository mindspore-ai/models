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

import cv2
from api.infer import SdkApi
from config import config as cfg
from StreamManagerApi import StreamManagerApi




def parser_args():
    """parser_args"""
    parser = argparse.ArgumentParser(description="metric_learn inference")

    parser.add_argument("--img_path",
                        type=str,
                        required=False,
                        default="../../data/Stanford_Online_Products",
                        help="image directory.")
    parser.add_argument(
        "--pipeline_path",
        type=str,
        required=False,
        default="./config/metric_learn.pipeline",
        help="image file path. The default is '/metric_learn/infer/sdk/config/metric_learn.pipeline'. ")
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
        default="../../data/infer_result",
        help=
        "cache dir of inference result. The default is '../data/infer_result'.")
    arg = parser.parse_args()
    return arg

def process_img(img_file):
    img0 = cv2.imread(img_file)
    img = resize_i(img0, height=cfg.MODEL_HEIGHT, width=cfg.MODEL_WIDTH)
    return img


def resize_i(img, height=224, width=224):
    """resize img"""
    percent = float(height) / min(img.shape[0], img.shape[1])
    resized_width = int(round(img.shape[1] * percent))
    resized_height = int(round(img.shape[0] * percent))
    img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4)
    shape = (224, 224)
    resized = cv2.resize(img, shape, interpolation=cv2.INTER_LINEAR)
    return resized

def image_inference(pipeline_path, stream_name, data_dir, result_dir):
    stream_manager_api = StreamManagerApi()
    start_time = time.time()
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)
    print(stream_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    img_data_plugin_id = 0

    print("\nBegin to inference for {}.\n".format(data_dir))
    TRAIN_LIST = "../data/Stanford_Online_Products/test_half.txt"
    TRAIN_LISTS = open(TRAIN_LIST, "r").readlines()
    max_len = 30003

    # cal_acc
    for _, item in enumerate(TRAIN_LISTS):
        if _ >= max_len:
            break
        items = item.strip().split()
        path = items[0]
        father = path.split("/")[0]
        father_path = os.path.join(result_dir, father)
        if not os.path.exists(father_path):
            os.makedirs(father_path)
        file_path = os.path.join(data_dir, path)
        save_bin_path = os.path.join(result_dir, "{}.bin".format(path.split(".")[0]))
        img_np = process_img(file_path)
        img_shape = img_np.shape
        # SDK
        sdk_api.send_img_input(stream_name,
                               img_data_plugin_id, "appsrc0",
                               img_np.tobytes(), img_shape)

        result = sdk_api.get_result(stream_name)
        with open(save_bin_path, "wb") as fp:
            fp.write(result)
            print(
                "End-2end inference, file_name:", file_path,
                "\n"
            )
    end_time = time.time()
    print("cost: ", end_time-start_time, "s")
    print("fps: ", 30003.0/(end_time-start_time), "imgs/sec")
    stream_manager_api.DestroyAllStreams()

if __name__ == "__main__":
    args = parser_args()
    image_inference(args.pipeline_path, cfg.STREAM_NAME.encode("utf-8"), args.img_path,
                    args.infer_result_dir)
