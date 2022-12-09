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
"""write mindrecord for LLNet"""
import os
import time
from math import ceil
import numpy as np
import cv2
from mindspore.mindrecord import FileWriter

if __name__ == '__main__':
    start_time = time.time()
    np.random.seed(10)

    dataset_directory = './dataset/'

    train_patches_per_image = 1250
    val_patches_per_image = 1250
    patches_per_image = train_patches_per_image + val_patches_per_image

    train_mindrecords = 'train/train_'+str(train_patches_per_image)+'patches_per_image.mindrecords'
    train_mindrecords = os.path.join(dataset_directory, train_mindrecords)
    val_mindrecords = 'val/val_'+str(val_patches_per_image)+'patches_per_image.mindrecords'
    val_mindrecords = os.path.join(dataset_directory, val_mindrecords)

    if not os.path.exists(os.path.join(dataset_directory, 'train')):
        os.mkdir(os.path.join(dataset_directory, 'train'))

    if not os.path.exists(os.path.join(dataset_directory, 'val')):
        os.mkdir(os.path.join(dataset_directory, 'val'))

    data_schema = {"label": {"type": "int32"},
                   "origin": {"type": "float32", "shape": [-1]},
                   "noise_darkened": {"type": "float32", "shape": [-1]}}

    if os.path.exists(train_mindrecords):
        os.remove(train_mindrecords)
    if os.path.exists(train_mindrecords + ".db"):
        os.remove(train_mindrecords + ".db")
    if os.path.exists(val_mindrecords):
        os.remove(val_mindrecords)
    if os.path.exists(val_mindrecords + ".db"):
        os.remove(val_mindrecords + ".db")

    train_writer = FileWriter(file_name=train_mindrecords, shard_num=1)
    train_writer.add_schema(data_schema, "llnet train dataset")
    val_writer = FileWriter(file_name=val_mindrecords, shard_num=1)
    val_writer.add_schema(data_schema, "llnet val dataset")

    images_directory = os.path.join(dataset_directory, 'train_val_images')
    image_file_names = os.listdir(images_directory)

    print("Generating ", train_mindrecords)
    print("Generating ", val_mindrecords)
    j = 0
    for i, image_file_name in enumerate(image_file_names):
        print('{}-th image ...'.format(i+1))
        image_gray = cv2.imread(os.path.join(images_directory, image_file_name), cv2.IMREAD_UNCHANGED)
        image_gray = image_gray / 255.

        image_height, image_width = image_gray.shape[:2]

        # crop patches
        rand_xx = np.random.randint(low=0, high=image_width-17, size=patches_per_image)
        rand_yy = np.random.randint(low=0, high=image_height-17, size=patches_per_image)

        for xx, yy in zip(rand_xx, rand_yy):
            origin_patch = image_gray[yy:yy+17, xx:xx+17]

            gamma = np.random.uniform(2, 5)
            image_darkened = origin_patch ** gamma
            sigma = (np.random.uniform())**0.5 * 0.1
            gauss_noise = np.random.normal(0, sigma, (17, 17))
            noise_darkened_patch = np.clip(image_darkened  + gauss_noise, 0, 1)

            origin_patch = origin_patch.reshape(-1)

            sample = {"label": i,
                      "origin": np.array(origin_patch, dtype=np.float32),
                      "noise_darkened": np.array(noise_darkened_patch, dtype=np.float32)}
            data = []
            data.append(sample)

            if j % 2 == 0:
                train_writer.write_raw_data(data)
            else:
                val_writer.write_raw_data(data)
            j += 1

    train_writer.commit()
    val_writer.commit()
    print("total ", j, " patches")
    print("time  ", ceil((time.time() - start_time)), " seconds")
