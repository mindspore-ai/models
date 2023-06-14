# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
import numpy as np
from PIL import Image

import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV
import mindspore.dataset as ds
from mindspore import dtype as mstype


class TrainDatasetGenerator:
    def __init__(self, train_data_dir):
        self.rgb_name_lists, self.depth_name_lists = self.get_rgb_and_depth_name_lists(train_data_dir)

    def __getitem__(self, index):
        rgb_name = self.rgb_name_lists[index]
        depth_name = self.depth_name_lists[index]

        rgb = Image.open(rgb_name)
        depth = Image.open(depth_name)
        rgb = np.array(rgb) / 255.0
        depth = np.array(depth) / 255.0 * 10.0
        depth = depth.reshape((depth.shape[0], depth.shape[1], 1))
        return rgb, depth

    def __len__(self):
        return len(self.rgb_name_lists)

    def get_rgb_and_depth_name_lists(self, train_data_dir):
        rgb_name_lists = []
        depth_name_lists = []
        scene_lists = os.listdir(train_data_dir)
        for scene_folder in scene_lists:
            file_lists = os.listdir(train_data_dir + "/" + scene_folder)
            for file_name in file_lists:
                if file_name[-4:] == ".jpg":
                    rgb_name = train_data_dir + "/" + scene_folder + "/" + file_name
                    rgb_name_lists.append(rgb_name)
                    depth_name = rgb_name.replace(".jpg", ".png")
                    depth_name_lists.append(depth_name)
        return rgb_name_lists, depth_name_lists


class TestDatasetGenerator:
    def __init__(self, test_data_dir):
        self.rgb_name_lists, self.depth_name_lists = self.get_rgb_and_depth_name_lists(test_data_dir)

    def __getitem__(self, index):
        rgb_name = self.rgb_name_lists[index]
        depth_name = self.depth_name_lists[index]
        rgb = Image.open(rgb_name)
        depth = Image.open(depth_name)
        rgb = np.array(rgb) / 255.0
        depth = np.array(depth) / 1000.0
        depth = depth.reshape((depth.shape[0], depth.shape[1], 1))
        return rgb, depth

    def __len__(self):
        return len(self.rgb_name_lists)

    def get_rgb_and_depth_name_lists(self, test_data_dir):
        rgb_name_lists = []
        depth_name_list = []
        file_lists = os.listdir(test_data_dir)
        file_lists.sort()
        for file_name in file_lists:
            if file_name[6:-4] == "colors":
                file_name = test_data_dir + "/" + file_name
                rgb_name_lists.append(file_name)
            if file_name[6:-4] == "depth":
                file_name = test_data_dir + "/" + file_name
                depth_name_list.append(file_name)

        return rgb_name_lists, depth_name_list


def create_test_dataset(test_data_dir, batch_size):
    dataset_generator = TestDatasetGenerator(test_data_dir)
    test_ds = ds.GeneratorDataset(dataset_generator, ["rgb", "ground_truth"], shuffle=False)

    type_cast_op_image = C.TypeCast(mstype.float32)
    type_cast_op_label = C.TypeCast(mstype.float32)
    crop = CV.Crop((12, 16), (456, 608))
    rgb_resize = CV.Resize((228, 304))
    depth_resize = CV.Resize((55, 74))
    hwc2chw = CV.HWC2CHW()

    test_ds = test_ds.map(operations=[type_cast_op_image, crop, rgb_resize, hwc2chw], input_columns="rgb")
    test_ds = test_ds.map(operations=[type_cast_op_label, crop, depth_resize, hwc2chw], input_columns="ground_truth")
    test_ds = test_ds.batch(batch_size=batch_size)

    return test_ds


if __name__ == "__main__":
    ds_generator = TestDatasetGenerator(test_data_dir="./Data/Test")

    test_dataset = ds.GeneratorDataset(ds_generator, ["rgb", "ground_truth"], shuffle=False)
    test_dataset = test_dataset.shuffle(buffer_size=1000)
    test_dataset = test_dataset.batch(batch_size=8)

    for data in test_dataset.create_dict_iterator():
        print("rgb: {}".format(data["rgb"].shape), "GT: {}".format(data["ground_truth"].shape))

    train_data_set_dir = "./Data/Train"
    train_dataset_generator = TrainDatasetGenerator(train_data_set_dir)
    train_dataset = ds.GeneratorDataset(train_dataset_generator, ["rgb", "ground_truth"], shuffle=True)
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(batch_size=32)
    for data in train_dataset.create_dict_iterator():
        print("rgb: {}".format(data["rgb"].shape), "GT: {}".format(data["ground_truth"].shape))
