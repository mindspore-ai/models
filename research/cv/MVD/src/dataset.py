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
""" dataset.py """

import os
import random
import numpy as np
from PIL import Image


class SYSUDatasetGenerator():
    """
    Data Generator for custom DataLoader in mindspore, for SYSU dataset
    """
    def __init__(self, data_dir="./data/SYSU/", colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        self.train_color_image =\
            np.load(os.path.join(data_dir, 'train_rgb_resized_img.npy'))
        self.train_color_label =\
            np.load(os.path.join(data_dir, 'train_rgb_resized_label.npy'))

        self.train_thermal_image =\
            np.load(os.path.join(data_dir, 'train_ir_resized_img.npy'))
        self.train_thermal_label =\
            np.load(os.path.join(data_dir, 'train_ir_resized_label.npy'))

        print(f"Color Image Size:{len(self.train_color_image)}")
        print(f"Color Label Size:{len(self.train_color_label)}")

        self.cindex = colorIndex
        self.tindex = thermalIndex

    def __next__(self):
        pass

    def __getitem__(self, index):

        img1, target1 =\
            self.train_color_image[self.cindex[index]], self.train_color_label[self.cindex[index]]
        img2 = self.train_thermal_image[self.tindex[index]]
        target2 = self.train_thermal_label[self.tindex[index]]

        return (img1, img2, target1, target2)

    def __len__(self):
        return len(self.cindex)


def load_data(input_data_path):
    """
    loading data
    """
    with open(input_data_path, encoding="utf-8"):
        with open(input_data_path, 'rt', encoding="utf-8") as path_str:
            data_file_list = path_str.read().splitlines()
            # Get full list of image and labels
            file_image = [s.split(' ')[0] for s in data_file_list]
            file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label


class RegDBDatasetGenerator():
    """
    Data Generator for custom DataLoader in mindspore, for RegDB dataset
    """
    def __init__(self, data_dir, trial, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        # data_dir = '../Datasets/RegDB/'
        train_color_list = os.path.join(data_dir, f'idx/train_visible_{trial}' + '.txt')
        train_thermal_list = os.path.join(data_dir, f'idx/train_thermal_{trial}' + '.txt')

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for _, file_ in enumerate(color_img_file):
            img = Image.open(os.path.join(data_dir, file_))
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for _, file_ in enumerate(thermal_img_file):
            img = Image.open(os.path.join(data_dir, file_))
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        self.cindex = colorIndex
        self.tindex = thermalIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cindex[index]],\
            self.train_color_label[self.cindex[index]]
        img2, target2 = self.train_thermal_image[self.tindex[index]],\
            self.train_thermal_label[self.tindex[index]]

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class TestData():
    """
    Test Data Generator for mindspore custom dataloader
    """
    def __init__(self, test_img_file, test_label, img_size=(144, 288)):

        test_image = []
        for _, file_ in enumerate(test_img_file):
            img = Image.open(file_)
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]

        return (img1, target1)

    def __len__(self):
        return len(self.test_image)


def process_query_sysu(data_path, mode='all'):
    """
    preprocess sysu query
    """
    if mode == 'all':
        ir_cameras = ['cam3', 'cam6']
    elif mode == 'indoor':
        ir_cameras = ['cam3', 'cam6']

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_ir = []

    with open(file_path, 'r', encoding="utf-8") as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = [str(x).zfill(4) for x in ids]


    for id_ in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id_)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir) if i[0] != '.'])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)


def process_gallery_sysu(data_path, mode='all', random_seed=0):
    """
    preprocess sysu gallery
    """
    random.seed(random_seed)

    if mode == 'all':
        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    elif mode == 'indoor':
        rgb_cameras = ['cam1', 'cam2']

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r', encoding="utf-8") as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = [f"{x:04d}" for x in ids]

    for id_ in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id_)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir) if i[0] != '.'])
                files_rgb.append(random.choice(new_files))
    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)


def process_test_regdb(img_dir, trial=1, modal='visible'):
    """
    preprocess regdb test
    """
    if modal == 'visible':
        input_data_path = os.path.join(img_dir, f'idx/test_visible_{trial}' + '.txt')
    elif modal == 'thermal':
        input_data_path = os.path.join(img_dir, f'idx/test_thermal_{trial}' + '.txt')

    with open(input_data_path, encoding="utf-8"):
        with open(input_data_path, 'rt', encoding="utf-8") as path_str:
            data_file_list = path_str.read().splitlines()
            # Get full list of image and labels
            file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
            file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, np.array(file_label)
