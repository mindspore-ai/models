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

"""preprocess for 310 inference"""
import os
import shutil
import argparse

import yaml
import cv2
import numpy as np


def get_args():
    """
    Cmd example:

    python preprocess.py
    --config_path=./config.yaml
    --output_path=./scripts/preprocess_Result
    """
    parser = argparse.ArgumentParser(description="Semantic human matting")
    parser.add_argument("--config_path", type=str, default=None, help="config path")
    parser.add_argument("--output_path", type=str, default=None, help="output path")
    args = parser.parse_args()
    print(args)
    return args


def get_config_from_yaml(args):
    yaml_file = open(args.config_path, "r", encoding="utf-8")
    file_data = yaml_file.read()
    yaml_file.close()

    y = yaml.load(file_data, Loader=yaml.FullLoader)
    cfg = y["infer"]
    cfg["output_path_pre"] = args.output_path
    return cfg


def safe_makedirs(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)


def safe_modify_file_name(file_name):
    if not os.path.exists(file_name):
        if "jpg" in file_name:
            return file_name.replace("jpg", "png")
        return file_name.replace("png", "jpg")
    return file_name


def preprocess(cfg):
    img_path = os.path.join(cfg["output_path_pre"], "img_data")
    clip_path = os.path.join(cfg["output_path_pre"], "clip_data")
    label_path = os.path.join(cfg["output_path_pre"], "label")
    safe_makedirs(img_path)
    safe_makedirs(clip_path)
    safe_makedirs(label_path)

    test_pic_path = os.path.split(cfg["file_test_list"])[0]
    print("preprocess ...")
    with open(cfg["file_test_list"]) as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            line = line.strip()
            img_clip = os.path.join(test_pic_path, "clip_img", line.replace("matting", "clip"))
            img_alpha = os.path.join(test_pic_path, "alpha", line)
            img_clip = safe_modify_file_name(img_clip)
            img_alpha = safe_modify_file_name(img_alpha)

            file_name = os.path.split(line)[1].split(".")[0]
            file_name_save = "matting_{}_{}.bin".format(str(idx).zfill(4), file_name)
            img_file_path = os.path.join(img_path, file_name_save)
            clip_file_path = os.path.join(clip_path, file_name_save.replace(".bin", ".jpg"))
            label_file_path = os.path.join(label_path, file_name_save.replace(".bin", ".png"))

            img_src = cv2.imread(img_clip)
            image_resize = cv2.resize(img_src, (cfg["size"], cfg["size"]), interpolation=cv2.INTER_CUBIC)
            image_resize = (
                image_resize
                - (
                    104.0,
                    112.0,
                    121.0,
                )
            ) / 255.0
            x = np.expand_dims(image_resize, axis=3)
            inputs = np.transpose(x, (3, 2, 0, 1))
            inputs = inputs.astype(np.float32)
            inputs.tofile(img_file_path)

            shutil.copyfile(img_alpha, label_file_path)
            shutil.copyfile(img_clip, clip_file_path)

        print("Total images is [{}], preprocess is finished!".format(len(lines)))


if __name__ == "__main__":
    preprocess(get_config_from_yaml(get_args()))
