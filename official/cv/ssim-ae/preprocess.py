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
# coding=gbk
import os
import argparse
import random
import shutil
import numpy as np
from model_utils.options import Options_310
from src.dataset import get_patch
import cv2
random.seed(1)
np.random.seed(1)


def read_img(img_path, grayscale):
    if grayscale:
        im = cv2.imread(img_path, 0)
    else:
        im = cv2.imread(img_path)
    return im


def copy_to(path):
    sub_fold = os.listdir(path)
    target_path = path + "/all_test/"
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    else:
        shutil.rmtree(target_path)
        os.mkdir(target_path)
    sub_fold = sorted(sub_fold)
    for i, sub_folder_name in enumerate(sub_fold):
        if sub_folder_name not in ('all_test', 'good'):
            imgName_list = os.listdir(path + sub_folder_name)
            imgName_list = sorted(imgName_list)
            for x in imgName_list:
                image_path = path + sub_folder_name + "/" + x
                shutil.copy(image_path, target_path + str(i) + "_" + x)


def preprocess(result_path, dir_path):
    dirs = os.listdir(dir_path)
    file_list = []
    for file in dirs:
        file_list.append(file)
    file_list = sorted(file_list)
    total = 0

    for i, file_dir in enumerate(file_list):
        total = total + 1
        print("image_path: ", dir_path+file_dir)
        test_img = read_img(dir_path+file_dir, cfg["grayscale"])

        if test_img.shape[:2] != (im_resize, im_resize):
            test_img = cv2.resize(test_img, (im_resize, im_resize)) / 255.0
        if im_resize != cfg["mask_size"]:
            tmp = (im_resize - cfg["mask_size"]) // 2
            test_img_ = test_img[tmp:tmp + cfg["mask_size"], tmp:tmp + cfg["mask_size"]]

        else:
            test_img_ = test_img

        if test_img_.shape[:2] == (crop_size, crop_size):
            if cfg["grayscale"]:
                test_img_ = np.expand_dims(test_img_, axis=(0, 1))
            else:
                test_img_ = np.expand_dims(test_img_, axis=0)
                test_img_ = np.transpose(test_img_, (0, 3, 1, 2))
            patch_num = 1
            for j in range(0, patch_num):
                file_name = "AE_SSIM_" + str(i) + "_patch_" + str(j) + ".bin"
                file_path = os.path.join(result_path, file_name)
                out = np.array(test_img_).astype(np.float32)
                out.tofile(file_path)

        else:
            patches = get_patch(test_img_, crop_size, cfg["stride"])
            if cfg["grayscale"]:
                patches = np.expand_dims(patches, 1)
            else:
                patches = np.transpose(patches, (0, 3, 1, 2))
            patch_num = patches.shape[0]

            for j in range(0, patch_num):
                file_name = "AE_SSIM_" + str(i) + "_patch_" + str(j) + ".bin"
                file_path = os.path.join(result_path, file_name)
                out = patches[[j]]
                out = np.array(out).astype(np.float32)
                out.tofile(file_path)

    print("[INFO] Completed! Total {} data.".format(total))


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./datasets/texture2/', help='eval data dir')
parser.add_argument('--result_path', type=str, default='./data/preprocess_Result/', help='result path')
args = parser.parse_args()
cfg = Options_310().opt
im_resize = cfg["data_augment"]["im_resize"]
crop_size = cfg["data_augment"]["crop_size"]
if __name__ == '__main__':
    data_path = args.data_path
    test_path = data_path+"/test/"
    sub_folder = os.listdir(test_path)
    if os.path.isdir(test_path + "/" + sub_folder[0]):
        copy_to(data_path + "/test/")
        preprocess(args.result_path, args.data_path+"/test/all_test/")
    else:
        preprocess(args.result_path, args.data_path + "/test/")
