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

"""post process for 310 inference"""
import os
import argparse

import yaml
import cv2
import numpy as np


def get_args():
    """
    Cmd example:

    python postprocess.py
        --config_path=./config.yaml
        --result_path=./scripts/result_Files
        --pre_path=./scripts/preprocess_Result
        --save_path=./scripts/postprocess_Result
    """
    parser = argparse.ArgumentParser(description="Semantic human matting")
    parser.add_argument("--config_path", type=str, default="./config.yaml", help="config path")
    parser.add_argument("--result_path", type=str, default="./scripts/result_Files", help="infer path")
    parser.add_argument("--pre_path", type=str, default="./scripts/preprocess_Result", help="pre path")
    parser.add_argument("--save_path", type=str, default="./scripts/postprocess_Result", help="save path")
    args = parser.parse_args()
    print(args)
    return args


def get_config_from_yaml(args):
    yaml_file = open(args.config_path, "r", encoding="utf-8")
    file_data = yaml_file.read()
    yaml_file.close()

    y = yaml.load(file_data, Loader=yaml.FullLoader)
    cfg = y["infer"]
    cfg["result_path"] = args.result_path
    cfg["pre_path"] = args.pre_path
    cfg["save_path"] = args.save_path

    return cfg


def safe_makedirs(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)


def cal_sad(cfg):
    """Calculate Sad metric"""
    print("postprocess and calculate metric ...")
    safe_makedirs(cfg["save_path"])
    files = os.listdir(cfg["result_path"])
    files = list(filter(lambda i: "_1.bin" in i, files))
    files.sort()
    list_sad = list()
    for _, file in enumerate(files):
        file_name = file.replace("_1.bin", "")

        file_infer = os.path.join(cfg["result_path"], file)
        alpha = np.fromfile(file_infer, dtype=np.float32).reshape((1, 1, cfg["size"], cfg["size"]))

        image_path = os.path.join(cfg["pre_path"], "clip_data", "{}.jpg".format(file_name))
        image = cv2.imread(image_path)

        label_path = os.path.join(cfg["pre_path"], "label", "{}.png".format(file_name))
        label = cv2.imread(label_path)

        # generate foreground image
        alpha_np = alpha[0, 0, :, :]
        origin_h, origin_w, _ = image.shape
        alpha_fg = cv2.resize(alpha_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)
        fg = np.multiply(alpha_fg[..., np.newaxis], image)
        fg_path = os.path.join(cfg["save_path"], "{}.jpg".format(file_name))
        cv2.imwrite(fg_path, fg)

        # generate metric Sad (original image size)
        image_gt = label[:, :, 0]
        image_gt = image_gt.astype(np.float64) / 255
        sad = np.abs(alpha_fg - image_gt).sum() / 1000
        list_sad.append(sad)

        print("{}\tsad\t{}".format(image_path, sad))
    print("Total images: {}, total sad: {}, ave sad: {}".format(len(list_sad), np.sum(list_sad), np.mean(list_sad)))


if __name__ == "__main__":
    cal_sad(get_config_from_yaml(get_args()))
