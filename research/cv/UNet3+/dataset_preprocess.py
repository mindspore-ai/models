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
'''Preprocessing of the dataset: turn nii into png'''
import os
import math
import random
import shutil
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import cv2
from src.config import config as cfg

if __name__ == "__main__":

    if not os.path.exists(os.path.join(cfg.buffer_path, "ct")):
        os.makedirs(os.path.join(cfg.buffer_path, "ct"))
    if not os.path.exists(os.path.join(cfg.buffer_path, "seg")):
        os.makedirs(os.path.join(cfg.buffer_path, "seg"))

    ct_path = os.path.join(cfg.source_path, "CT")

    for index, file in enumerate(tqdm(os.listdir(ct_path))):

        ct_src = sitk.ReadImage(os.path.join(ct_path, file), sitk.sitkInt16)
        mask = sitk.ReadImage(os.path.join(cfg.source_path, "seg", \
                    file.replace('volume', 'segmentation')), sitk.sitkUInt8)
        ct_array = sitk.GetArrayFromImage(ct_src)
        mask_array = sitk.GetArrayFromImage(mask)

        ct_array[ct_array > 200] = 200
        ct_array[ct_array < -200] = -200

        z = np.any(mask_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]

        ct_crop = ct_array[start_slice - 1:end_slice + 1, :, :]
        mask_crop = mask_array[start_slice:end_slice + 1, :, :]

        for n_slice in range(mask_crop.shape[0]):
            maskImg = mask_crop[n_slice, :, :] * 255
            cv2.imwrite(os.path.join(cfg.buffer_path, "seg", str(index) + "_" + str(n_slice) + ".png"), maskImg)

            ctImageArray = np.zeros((ct_crop.shape[1], ct_crop.shape[2], 3), np.float)
            ctImageArray[:, :, 0] = ct_crop[n_slice - 1, :, :]
            ctImageArray[:, :, 1] = ct_crop[n_slice, :, :]
            ctImageArray[:, :, 2] = ct_crop[n_slice + 1, :, :]
            ctImg = ct_crop[n_slice, :, :]
            ctImg = ctImg.astype(np.float)

            cv2.imwrite(os.path.join(cfg.buffer_path, "ct", str(index) + "_" + str(n_slice) + ".png"), ctImageArray)
    print("Data transform Done！")

    if not os.path.exists(os.path.join(cfg.dest_path, "train", "ct")):
        os.makedirs(os.path.join(cfg.dest_path, "train", "ct"))
    if not os.path.exists(os.path.join(cfg.dest_path, "train", "seg")):
        os.makedirs(os.path.join(cfg.dest_path, "train", "seg"))
    if not os.path.exists(os.path.join(cfg.dest_path, "test", "ct")):
        os.makedirs(os.path.join(cfg.dest_path, "test", "ct"))
    if not os.path.exists(os.path.join(cfg.dest_path, "test", "seg")):
        os.makedirs(os.path.join(cfg.dest_path, "test", "seg"))

    seg = os.listdir(os.path.join(cfg.buffer_path, "seg"))
    random.seed(1000)
    random.shuffle(seg)

    print("Start to split train data！")
    for index, i in enumerate(seg[:math.floor(len(seg)*0.8)]):
        if (index+1)%1000 == 0:
            print(index+1, "/", math.floor(len(seg)*0.8))
        shutil.move(os.path.join(cfg.buffer_path, "ct", i), os.path.join(cfg.dest_path, "train", "ct"))
        shutil.move(os.path.join(cfg.buffer_path, "seg", i), os.path.join(cfg.dest_path, "train", "seg"))

    print("Start to split val data！")
    for index, i in enumerate(seg[math.floor(len(seg)*0.8):]):
        if (index+1)%1000 == 0:
            print(index, "/", len(seg)-math.floor(len(seg)*0.8))
        shutil.move(os.path.join(cfg.buffer_path, "ct", i), os.path.join(cfg.dest_path, "test", "ct"))
        shutil.move(os.path.join(cfg.buffer_path, "seg", i), os.path.join(cfg.dest_path, "test", "seg"))

    shutil.rmtree(cfg.buffer_path)
    print("Data processing and splitting finished!")
