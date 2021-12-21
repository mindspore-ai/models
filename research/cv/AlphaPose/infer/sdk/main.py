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


def parser_args():
    '''set parameter'''
    parser = argparse.ArgumentParser(description="maskrcnn inference")
    parser.add_argument("--img_path",
                        type=str,
                        required=True,
                        help="image directory.")
    parser.add_argument(
        "--pipeline_path",
        type=str,
        required=False,
        default="config/maskrcnn_ms.pipeline",
        help="image file path. The default is 'config/maskrcnn_ms.pipeline'. ")
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        default="dvpp",
        help="rgb: high-precision, dvpp: high performance. The default is 'dvpp'.")
    parser.add_argument(
        "--infer_mode",
        type=str,
        required=False,
        default="infer",
        help="infer:only infer, eval: accuracy evaluation. The default is 'infer'.")
    parser.add_argument(
        "--infer_result_dir",
        type=str,
        required=False,
        default="../data/infer_result",
        help="cache dir of inference result. The default is '../data/infer_result'."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        default="TUM",
        help="dataset name."
    )

    parser.add_argument("--ann_file",
                        type=str,
                        required=False,
                        help="eval ann_file.")

    return parser.parse_args()


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


def write_pfm(path, image, scale=1):
    """Write pfm file.

    Args:
        path (str): path file
        image (array): data
        scale (int, optional): Scale. Defaults to 1.
    """

    with open(path, "wb") as file:
        color = None

        if image.dtype.name != "float32":
            raise Exception("Image dtype must be float32.")

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif (
                len(image.shape) == 2 or len(
                    image.shape) == 3 and image.shape[2] == 1
        ):  # greyscale
            color = False
        else:
            raise Exception(
                "Image must have H x W x 3, H x W x 1 or H x W dimensions.")

        file.write("PF\n" if color else "Pf\n".encode())
        file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale

        file.write("%f\n".encode() % scale)

        image.tofile(file)


def process_img(img_file):
    '''bin preprocess'''
    f = open(img_file, mode='rb')
    img = np.fromfile(f, dtype=np.float32).reshape((256, 192, 3))
    return img


def image_inference(pipeline_path, stream_name, img_dir, result_dir,
                    replace_last, dataset_name, model_type):
    '''sdk process'''
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    img_data_plugin_id = 0
    dirs = []
    if dataset_name == 'images':
        data_set = img_dir + "images"
        dirs.append("images")
    elif dataset_name == "Kitti":
        data_set = img_dir + "/Kitti_raw_data"
        dirs = os.listdir(data_set)
    elif dataset_name == "Sintel":
        data_set = img_dir + "/Sintel/final_left"
        dirs = os.listdir(data_set)
    else:
        data_set = img_dir
        dirs = os.listdir(data_set)

    for d in dirs:
        if dataset_name == "images":
            images = os.listdir(os.path.join(data_set))
        elif dataset_name == "Kitti":
            images = os.listdir(os.path.join(data_set, d, "images"))
        elif dataset_name == "Sintel":
            images = os.listdir(os.path.join(data_set, d))
        else:
            images = os.listdir(os.path.join(data_set, d))
        total_len = len(images)
        for ind, file_name in enumerate(images):
            if dataset_name == "images":
                file_path = os.path.join(data_set, file_name)
            elif dataset_name == "Kitti":
                file_path = os.path.join(data_set, d, "images", file_name)
            elif dataset_name == "Sintel":
                file_path = os.path.join(data_set, d, file_name)
            print(img_dir, "  ", d, "  ", file_name)
            print("file_path is ", file_path)
            img_np = process_img(file_path)
            img_shape = img_np.shape
            print("111", img_shape)
            sdk_api.send_img_input(stream_name,
                                   img_data_plugin_id, "appsrc0",
                                   img_np.tobytes(), img_shape)
            start_time = time.time()
            result = sdk_api.get_result(stream_name)
            end_time = time.time() - start_time

            save_path = os.path.join(result_dir, dataset_name, d)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, file_name)
            print('.' + save_path.split('.')[-1])
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
                    args.infer_result_dir, True, args.dataset_name, args.model_type)
