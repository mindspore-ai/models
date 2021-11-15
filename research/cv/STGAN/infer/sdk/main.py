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
""" Model Main """
import argparse
import time
import os
import numpy as np
import cv2
from api.infer import SdkApi
from config import config as cfg

def parser_args():
    """ Args Setting """
    parser = argparse.ArgumentParser(description="stgan inference")

    parser.add_argument("--img_path",
                        type=str,
                        required=True,
                        help="image directory.")
    parser.add_argument(
        "--pipeline_path",
        type=str,
        required=False,
        default="config/stgan.pipeline",
        help="image file path. The default is 'config/stgan.pipeline'. ")
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        default="dvpp",
        help=
        "rgb: high-precision, dvpp: high performance. The default is 'dvpp'.")
    parser.add_argument(
        "--infer_result_dir",
        type=str,
        required=False,
        default="../data/sdk_result",
        help=
        "cache dir of inference result. The default is '../data/sdk_result'."
    )

    parser.add_argument("--ann_file",
                        type=str,
                        required=False,
                        help="eval ann_file.")

    args_ = parser.parse_args()
    return args_

def get_labels(img_dir):
    """ Get Labels Setting """
    # labels preprocess
    selected_attrs = cfg.SELECTED_ATTRS
    lines = [
        line.rstrip() for line in open(
            os.path.join(img_dir, 'anno', 'list_attr_celeba.txt'), 'r')
    ]

    all_attr_names = lines[1].split()
    attr2idx = {}
    idx2attr = {}
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name
    lines = lines[2:]
    items = {}
    for i, line in enumerate(lines):
        split = line.split()
        filename = split[0]
        values = split[1:]
        label = []
        for attr_name in selected_attrs:
            idx = attr2idx[attr_name]
            label.append(int(values[idx]) * -0.5)
        items[filename] = np.array(label).astype(np.float32)
    return items

def process_img(img_file):
    """ Preprocess Image """
    img = cv2.imread(img_file)
    model_img = cv2.resize(img, (cfg.MODEL_WIDTH, cfg.MODEL_HEIGHT))
    img_ = model_img[:, :, ::-1].transpose((2, 0, 1))
    img_ = np.expand_dims(img_, axis=0)
    img_ = np.array((img_-127.5)/127.5).astype(np.float32)
    return img_

def decode_image(img):
    """ Decode Image """
    mean = 0.5 * 255
    std = 0.5 * 255
    return (img * std + mean).astype(np.uint8).transpose(
        (1, 2, 0))

def image_inference(pipeline_path, stream_name, img_dir, result_dir,
                    replace_last, model_type):
    """ Image Inference """
    # init stream manager
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    img_data_plugin_id = 0
    img_label_plugin_id = 1

    label_items = get_labels(img_dir)

    print(f"\nBegin to inference for {img_dir}.\n\n")

    file_list = os.listdir(os.path.join(img_dir, 'image'))
    for _, file_name in enumerate(file_list):
        if not file_name.lower().endswith((".jpg", "jpeg")):
            continue
        file_path = os.path.join(img_dir, 'image', file_name)
        save_path = os.path.join(result_dir,
                                 f"{os.path.splitext(file_name)[0]}.jpg")
        if not replace_last and os.path.exists(save_path):
            print(f"The infer result image({save_path}) has existed, will be skip.")
            continue

        img_np = process_img(file_path)

        start_time = time.time()
        sdk_api.send_tensor_input(stream_name, img_data_plugin_id, "appsrc0",
                                  img_np.tobytes(), img_np.shape, cfg.TENSOR_DTYPE_FLOAT32)

        # set label data
        label_dim = np.expand_dims(label_items[file_name], axis=0)
        sdk_api.send_tensor_input(stream_name, img_label_plugin_id, "appsrc1",
                                  label_dim.tobytes(), [1, 4], cfg.TENSOR_DTYPE_FLOAT32)

        result = sdk_api.get_result(stream_name)
        end_time = time.time() - start_time
        print(f"The image({save_path}) inference time is {end_time}")
        data = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype=np.float32)
        data = data.reshape(3, 128, 128)
        img = decode_image(data)
        img = img[:, :, ::-1]
        cv2.imwrite(save_path, img)

if __name__ == "__main__":
    args = parser_args()
    args.replace_last = True
    args.stream_name = cfg.STREAM_NAME.encode("utf-8")
    image_inference(args.pipeline_path, args.stream_name, args.img_path,
                    args.infer_result_dir, args.replace_last, args.model_type)
