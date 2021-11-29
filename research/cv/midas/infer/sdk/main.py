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
'''sdk main'''

import argparse
import os
import time
import cv2
import numpy as np
from api.infer import SdkApi
from config import config as cfg
from eval.eval_by_sdk import run_eval


def parser_args():
    '''set parameter'''
    parser = argparse.ArgumentParser(description="midas inference")
    parser.add_argument("--img_path",
                        type=str,
                        required=True,
                        help="image directory.")
    parser.add_argument(
        "--pipeline_path",
        type=str,
        required=False,
        default="config/midas_ms_test.pipeline",
        help="image file path. The default is 'config/midas_ms_test.pipeline'. ")

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
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        default="TUM",
        help=
        "dataset name."
    )
    parser.add_argument(
        "--visualization",
        type=bool,
        default=True,
        help=
        "visualization."
    )

    args1 = parser.parse_args()
    return args1


def write_depth(path, depth, bits=1):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
        :param path:
        :param depth:
        :param bits:
    """
    write_pfm(path + ".pfm", depth.astype(np.float32))

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2 ** (8 * bits)) - 1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.type)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))

    return out


def process_img(img_file):
    '''cv preprocess'''
    img = cv2.imread(img_file)
    pad_img = cv2.resize(img, (cfg.MODEL_WIDTH, cfg.MODEL_HEIGHT), interpolation=cv2.INTER_AREA)
    print(type(pad_img))
    pad_img.astype(np.float32)
    return pad_img


def image_inference(pipeline_path, stream_name, img_dir, result_dir, dataset_name):
    '''sdk process'''
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    img_data_plugin_id = 0
    dirs = []
    if dataset_name == 'TUM':
        data_set = img_dir + "/TUM/rgbd_dataset_freiburg2_desk_with_person"
        dirs.append("TUM")
    elif dataset_name == "Kitti":
        data_set = img_dir + "/Kitti_raw_data"
        dirs = os.listdir(data_set)
    elif dataset_name == "Sintel":
        data_set = img_dir + "/Sintel/final_left"
        dirs = os.listdir(data_set)
    else:
        data_set = img_dir
        dirs = os.listdir(data_set)

    for dir1 in dirs:
        if dataset_name == "TUM":
            images = os.listdir(os.path.join(data_set, 'rgb'))
        elif dataset_name == "Kitti":
            images = os.listdir(os.path.join(data_set, dir1, "image"))
        elif dataset_name == "Sintel":
            images = os.listdir(os.path.join(data_set, dir1))
        else:
            images = os.listdir(os.path.join(data_set, dir1))
        total_len = len(images)
        for ind, file_name in enumerate(images):
            if dataset_name == "TUM":
                file_path = os.path.join(data_set, 'rgb', file_name)
            elif dataset_name == "Kitti":
                file_path = os.path.join(data_set, dir1, "image", file_name)
            elif dataset_name == "Sintel":
                file_path = os.path.join(data_set, dir1, file_name)
            img_np = process_img(file_path)
            img_shape = img_np.shape
            sdk_api.send_img_input(stream_name,
                                   img_data_plugin_id, "appsrc0",
                                   img_np.tobytes(), img_shape)
            start_time = time.time()
            result = sdk_api.get_result(stream_name)
            end_time = time.time() - start_time

            save_path = os.path.join(result_dir, dataset_name, dir1)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, file_name)
            save_path = save_path.replace('.' + save_path.split('.')[-1], '_0.bin')
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
                    args.infer_result_dir, args.dataset_name)
    if args.infer_mode == "eval":
        args.dataset_path = args.img_path
        args.result_path = args.infer_result_dir
        run_eval(args)
