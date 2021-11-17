# !/usr/bin/env python

# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
"""
Sdk internece
"""
import argparse
import json
import os
import time

import copy
import cv2
import numpy as np

from api.infer import SdkApi
from api.visual import visual_image
from api.postprocess import data_process
from api.image import get_affine_transform
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StringVector
from config import config as cfg
from eval.eval_by_sdk import cal_acc


def parser_args():
    """
    configuration parameter, input from outside
    """
    parser = argparse.ArgumentParser(description="centernet inference")

    parser.add_argument("--img_path",
                        type=str,
                        required=True,
                        help="image file path.")
    parser.add_argument(
        "--pipeline_path",
        type=str,
        required=False,
        default="config/centernet.pipeline",
        help="pipeline file path. The default is 'config/centernet.pipeline'. ")
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
        default="../data/infer_result",
        help=
        "cache dir of inference result. The default is '../data/infer_result'."
    )

    parser.add_argument("--ann_file",
                        type=str,
                        required=False,
                        help="eval ann_file.")

    arg = parser.parse_args()
    return arg

def process_img(img_file):
    """
    Preprocessing the images
    """
    mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
    std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
    input_size = [512, 512]
    img = cv2.imread(img_file)
    size = img.shape
    inp_width = size[1]
    inp_height = size[0]
    down_ratio = 4
    c = np.array([inp_width / 2., inp_height / 2.], dtype=np.float32)
    s = max(inp_height, inp_width) * 1.0
    img_metas = {'c': c, 's': s,
                 'out_height': input_size[0] // down_ratio,
                 'out_width': input_size[1] // down_ratio}
    trans_input = get_affine_transform(c, s, 0, [input_size[0], input_size[1]])
    inp_img = cv2.warpAffine(img, trans_input, (cfg.MODEL_WIDTH, cfg.MODEL_HEIGHT), flags=cv2.INTER_LINEAR)
    inp_img = (inp_img.astype(np.float32) / 255. - mean) / std
    eval_image = inp_img.reshape((1,) + inp_img.shape)
    model_img = eval_image.transpose(0, 3, 1, 2)

    return model_img, img_metas

def image_inference(pipeline_path, stream_name, img_dir, result_dir):
    """
    image inference: get inference for images
    """
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    img_data_plugin_id = 0
    print(f"\nBegin to inference for {img_dir}.\n")

    file_list = os.listdir(img_dir)
    total_len = len(file_list)
    for img_id, file_name in enumerate(file_list):
        if not file_name.lower().endswith((".jpg", "jpeg")):
            continue
        image_name, _ = os.path.splitext(file_name)
        file_path = os.path.join(img_dir, file_name)

        img_np, meta = process_img(file_path)
        sdk_api.send_tensor_input(stream_name,
                                  img_data_plugin_id, "appsrc0",
                                  img_np.tobytes(), img_np.shape, cfg.TENSOR_DTYPE_FLOAT32)

        keys = [b"mxpi_tensorinfer0"]
        keyVec = StringVector()
        for key in keys:
            keyVec.push_back(key)
        start_time = time.time()
        infer_result = sdk_api. get_protobuf(stream_name, 0, keyVec)
        end_time = time.time() - start_time
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result[0].messageBuf)
        result = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr,
                               dtype='float32').reshape((1, 100, 6))
        img_id += 1
        output = data_process(result, meta, image_name, cfg.NUM_CLASSES)
        print(
            f"End-2end inference, file_name: {file_path}, {img_id}/{total_len}, elapsed_time: {end_time}.\n"
        )

        save_pred_image_path = os.path.join(result_dir, "pred_image")
        if not os.path.exists(save_pred_image_path):
            os.makedirs(save_pred_image_path)
        gt_image = cv2.imread(file_path)
        anno = copy.deepcopy(output["annotations"])
        visual_image(gt_image, anno, save_pred_image_path, score_threshold=cfg.SCORE_THRESH)
        pred_res_file = os.path.join(result_dir, 'infer_{}_result.json').format(image_name)
        with open(pred_res_file, 'w+') as f:
            json.dump(output["annotations"], f, indent=1)

if __name__ == "__main__":
    args = parser_args()
    stream_name0 = cfg.STREAM_NAME.encode("utf-8")
    print("stream_name0:")
    print(stream_name0)
    image_inference(args.pipeline_path, stream_name0, args.img_path,
                    args.infer_result_dir)
    if args.infer_mode == "eval":
        print("Infer end.")
        print("Begin to eval...")
        cal_acc(args.ann_file, args.infer_result_dir)
