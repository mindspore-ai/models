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

import os
import shutil
import cv2
import numpy as np
from model_utils.config import config as cfg
from src.utils import get_patch, read_img, get_file_list


def preprocess(dir_path, preprocess_result, postprocess_result):
    file_list = get_file_list(dir_path, cfg.img_suffix)
    for i, img_path in enumerate(file_list):
        test_img = read_img(img_path, cfg.grayscale)
        if test_img.shape[:2] != (cfg.im_resize, cfg.im_resize):
            test_img = cv2.resize(test_img, (cfg.im_resize, cfg.im_resize))
        if cfg.im_resize != cfg.mask_size:
            tmp = (cfg.im_resize - cfg.mask_size) // 2
            test_img = test_img[tmp : tmp + cfg.mask_size, tmp : tmp + cfg.mask_size]
        test_img_ = test_img / 255.0
        if test_img.shape[:2] == (cfg.crop_size, cfg.crop_size):
            if cfg.grayscale:
                patches = np.expand_dims(test_img_, axis=(0, 1))
            else:
                patches = np.expand_dims(test_img_, axis=0)
                patches = np.transpose(patches, (0, 3, 1, 2))
        else:
            patches = get_patch(test_img_, cfg.crop_size, cfg.stride)
            if cfg.grayscale:
                patches = np.expand_dims(patches, 1)
            else:
                patches = np.transpose(patches, (0, 3, 1, 2))
        patches = patches.astype(np.float32)
        patches.tofile(os.path.join(preprocess_result, f"{i}.bin"))
    test_path_s = len(cfg.test_dir)
    if cfg.test_dir[-1] != "/":
        test_path_s += 1
    with open("310_file_pair.txt", "w") as f:
        for i, img_path in enumerate(file_list):
            file_name, _ = os.path.splitext(img_path)
            img_name = file_name[test_path_s:]
            f.write(
                "{} {} {}\n".format(
                    img_name,
                    os.path.join(preprocess_result, f"{i}.bin"),
                    os.path.join(postprocess_result, f"{i}_0.bin"),
                )
            )
    print("[INFO] Completed! Total {} data.".format(len(file_list)))


if __name__ == "__main__":
    if os.path.exists("./preprocess_result"):
        shutil.rmtree("./preprocess_result")
    os.makedirs("./preprocess_result")
    preprocess(cfg.test_dir, "./preprocess_result", "./postprocess_result")
