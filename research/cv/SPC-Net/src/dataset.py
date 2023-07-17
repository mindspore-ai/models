# Copyright 2023 Huawei Technologies Co., Ltd
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
import cv2
import imageio
import numpy as np
from PIL import Image
import scipy.misc as misc

import mindspore as ms
import mindspore.dataset as de
from mindspore.dataset import vision, transforms

from src import cityscapes_labels

id_to_trainid = cityscapes_labels.label2trainid
trainid_to_trainid = cityscapes_labels.trainId2trainId
ID_TO_IGNORE_OR_GROUP = {}
IGNORE_LABEL = 255
color_to_trainid = cityscapes_labels.color2trainId


trainid_to_trainid_synthia = {
        0: IGNORE_LABEL,  # void
        1: 10,            # sky
        2: 2,             # building
        3: 0,             # road
        4: 1,             # sidewalk
        5: 4,             # fence
        6: 8,             # vegetation
        7: 5,             # pole
        8: 13,            # car
        9: 7,             # traffic sign
        10: 11,           # pedestrian - person
        11: 18,           # bicycle
        12: 17,           # motorcycle
        13: IGNORE_LABEL, # parking-slot
        14: IGNORE_LABEL, # road-work
        15: 6,            # traffic light
        16: 9,            # terrain
        17: 12,           # rider
        18: 14,           # truck
        19: 15,           # bus
        20: 16,           # train
        21: 3,            # wall
        22: IGNORE_LABEL  # Lanemarking
    }


def gen_id_to_ignore():
    global ID_TO_IGNORE_OR_GROUP
    for i in range(66):
        ID_TO_IGNORE_OR_GROUP[i] = IGNORE_LABEL
    ### Convert each class to cityscapes one
    ### Road
    # Road
    ID_TO_IGNORE_OR_GROUP[13] = 0
    # Lane Marking - General
    ID_TO_IGNORE_OR_GROUP[24] = 0
    # Manhole
    ID_TO_IGNORE_OR_GROUP[41] = 0
    ### Sidewalk
    # Curb
    ID_TO_IGNORE_OR_GROUP[2] = 1
    # Sidewalk
    ID_TO_IGNORE_OR_GROUP[15] = 1
    ### Building
    # Building
    ID_TO_IGNORE_OR_GROUP[17] = 2
    ### Wall
    # Wall
    ID_TO_IGNORE_OR_GROUP[6] = 3
    ### Fence
    # Fence
    ID_TO_IGNORE_OR_GROUP[3] = 4
    ### Pole
    # Pole
    ID_TO_IGNORE_OR_GROUP[45] = 5
    # Utility Pole
    ID_TO_IGNORE_OR_GROUP[47] = 5
    ### Traffic Light
    # Traffic Light
    ID_TO_IGNORE_OR_GROUP[48] = 6
    ### Traffic Sign
    # Traffic Sign
    ID_TO_IGNORE_OR_GROUP[50] = 7
    ### Vegetation
    # Vegitation
    ID_TO_IGNORE_OR_GROUP[30] = 8
    ### Terrain
    # Terrain
    ID_TO_IGNORE_OR_GROUP[29] = 9
    ### Sky
    # Sky
    ID_TO_IGNORE_OR_GROUP[27] = 10
    ### Person
    # Person
    ID_TO_IGNORE_OR_GROUP[19] = 11
    ### Rider
    # Bicyclist
    ID_TO_IGNORE_OR_GROUP[20] = 12
    # Motorcyclist
    ID_TO_IGNORE_OR_GROUP[21] = 12
    # Other Rider
    ID_TO_IGNORE_OR_GROUP[22] = 12
    ### Car
    # Car
    ID_TO_IGNORE_OR_GROUP[55] = 13
    ### Truck
    # Truck
    ID_TO_IGNORE_OR_GROUP[61] = 14
    ### Bus
    # Bus
    ID_TO_IGNORE_OR_GROUP[54] = 15
    ### Train
    # On Rails
    ID_TO_IGNORE_OR_GROUP[58] = 16
    ### Motorcycle
    # Motorcycle
    ID_TO_IGNORE_OR_GROUP[57] = 17
    ### Bicycle
    # Bicycle
    ID_TO_IGNORE_OR_GROUP[52] = 18


class CityscapesData:
    def __init__(self, root, do_train=False):
        self.root = root
        self.do_train = do_train
        if self.do_train:
            self.img_path = os.path.join(self.root, 'leftImg8bit_trainvaltest', 'leftImg8bit', 'train')
            self.label_path = os.path.join(self.root, 'gtFine_trainvaltest', 'gtFine', 'train')
        else:
            self.img_path = os.path.join(self.root, 'leftImg8bit_trainvaltest', 'leftImg8bit', 'val')
            self.label_path = os.path.join(self.root, 'gtFine_trainvaltest', 'gtFine', 'val')
        self.img_list = []
        self.label_list = []
        img_postfix = '_leftImg8bit.png'
        label_postfix = '_gtFine_labelIds.png'
        for folder in os.listdir(self.img_path):
            temp_folder = os.path.join(self.img_path, folder)
            for img in os.listdir(temp_folder):
                self.img_list.append(os.path.join(temp_folder, img))
                label = img.split(img_postfix)[0] + label_postfix
                self.label_list.append(os.path.join(self.label_path, folder, label))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label_path = self.label_list[index]
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = np.array(img)
        label = np.array(label)
        label_copy = label.copy()
        for k, v in id_to_trainid.items():
            label_copy[label == k] = v
        label = label_copy.astype(np.uint8)
        return img, label, img_name


class BDD100KData:
    def __init__(self, root, do_train=False):
        self.root = root
        self.do_train = do_train
        if self.do_train:
            self.img_path = os.path.join(self.root, 'images', 'train')
            self.label_path = os.path.join(self.root, 'labels', 'train')
        else:
            self.img_path = os.path.join(self.root, 'images', 'val')
            self.label_path = os.path.join(self.root, 'labels', 'val')
        self.img_list = []
        self.label_list = []
        img_postfix = '.jpg'
        label_postfix = '_train_id.png'
        for img in os.listdir(self.img_path):
            self.img_list.append(os.path.join(self.img_path, img))
            label = img.split(img_postfix)[0] + label_postfix
            self.label_list.append(os.path.join(self.label_path, label))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label_path = self.label_list[index]
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = np.array(img)
        label = np.array(label)
        label_copy = label.copy()
        for k, v in trainid_to_trainid.items():
            label_copy[label == k] = v
        label = label_copy.astype(np.uint8)
        return img, label, img_name


class MapillaryData:
    def __init__(self, root, do_train=False):
        gen_id_to_ignore()
        self.root = root
        self.do_train = do_train
        if self.do_train:
            self.img_path = os.path.join(self.root, 'training', 'images')
            self.label_path = os.path.join(self.root, 'training', 'labels')
        else:
            self.img_path = os.path.join(self.root, 'validation', 'images')
            self.label_path = os.path.join(self.root, 'validation', 'labels')
        self.img_list = []
        self.label_list = []
        img_postfix = '.jpg'
        label_postfix = '.png'
        for img in os.listdir(self.img_path):
            self.img_list.append(os.path.join(self.img_path, img))
            label = img.split(img_postfix)[0] + label_postfix
            self.label_list.append(os.path.join(self.label_path, label))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label_path = self.label_list[index]
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = np.array(img)
        label = np.array(label)
        label_copy = label.copy()
        for k, v in ID_TO_IGNORE_OR_GROUP.items():
            label_copy[label == k] = v
        label = label_copy.astype(np.uint8)
        # resize
        size = 1536
        height, width = img.shape[0], img.shape[1]
        if height != width:
            min_value = min(height, width)
            if min_value != height:
                top = (height - min_value) // 2
                bottom = height - min_value - top
                img = img[top:-bottom, :, :]
                label = label[top:-bottom, :]
            else:
                left = (width - min_value) // 2
                right = width - min_value - left
                img = img[:, left:-right, :]
                label = label[:, left:-right]
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (size, size), interpolation=cv2.INTER_NEAREST)
        return img, label, img_name


class GTAVData:
    def __init__(self, root, do_train=False):
        self.root = root
        self.do_train = do_train
        if self.do_train:
            self.img_path = os.path.join(self.root, 'images', 'train')
            self.label_path = os.path.join(self.root, 'labels', 'train')
        else:
            self.img_path = os.path.join(self.root, 'images', 'valid')
            self.label_path = os.path.join(self.root, 'labels', 'valid')
        self.img_list = []
        self.label_list = []
        img_postfix = '.png'
        label_postfix = '.png'
        for img in os.listdir(self.img_path):
            self.img_list.append(os.path.join(self.img_path, img))
            label = img.split(img_postfix)[0] + label_postfix
            self.label_list.append(os.path.join(self.label_path, label))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label_path = self.label_list[index]
        img = Image.open(img_path).convert('RGB')
        label = misc.imread(label_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        while (img.size[1], img.size[0]) != label[:, :, 0].shape:
            if index + 1 == len(self.img_list):
                index = 0
            else:
                index += 1
            img_path = self.img_list[index]
            label_path = self.label_list[index]
            img = Image.open(img_path).convert('RGB')
            label = misc.imread(label_path)
            img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = np.array(img)
        image_size = label[:, :, 0].shape
        label_copy = np.full(image_size, IGNORE_LABEL, dtype=np.uint8)
        for k, v in color_to_trainid.items():
            if v not in [255, -1]:
                label_copy[
                    (label == np.array(k))[:, :, 0] &
                    (label == np.array(k))[:, :, 1] &
                    (label == np.array(k))[:, :, 2]
                ] = v
        label = label_copy.astype(np.uint8)
        return img, label, img_name


class SynthiaData:
    def __init__(self, root, do_train=False):
        self.root = root
        self.do_train = do_train
        if self.do_train:
            self.img_path = os.path.join(self.root, 'RGB', 'train')
            self.label_path = os.path.join(self.root, 'GT', 'LABELS', 'train')
        else:
            self.img_path = os.path.join(self.root, 'RGB', 'val')
            self.label_path = os.path.join(self.root, 'GT', 'LABELS', 'val')
        self.img_list = []
        self.label_list = []
        img_postfix = '.png'
        label_postfix = '.png'
        for img in os.listdir(self.img_path):
            self.img_list.append(os.path.join(self.img_path, img))
            label = img.split(img_postfix)[0] + label_postfix
            self.label_list.append(os.path.join(self.label_path, label))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label_path = self.label_list[index]
        img = Image.open(img_path).convert('RGB')
        label = imageio.imread(label_path, format='PNG-FI')
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = np.array(img)
        label = np.array(label, dtype=np.uint8)[:, :, 0]
        label_copy = label.copy()
        for k, v in trainid_to_trainid_synthia.items():
            label_copy[label == k] = v
        label = label_copy.astype(np.uint8)
        return img, label, img_name


class IDDData:
    def __init__(self, root, do_train=False):
        self.root = root
        self.do_train = do_train
        if self.do_train:
            self.img_path = os.path.join(self.root, 'leftImg8bit', 'train')
            self.label_path = os.path.join(self.root, 'gtFine', 'train')
        else:
            self.img_path = os.path.join(self.root, 'leftImg8bit', 'val')
            self.label_path = os.path.join(self.root, 'gtFine', 'val')
        self.img_list = []
        self.label_list = []
        img_postfix = '_leftImg8bit.png'
        label_postfix = '_gtFine_labelcsTrainIds.png'
        for folder in os.listdir(self.img_path):
            temp_folder = os.path.join(self.img_path, folder)
            for img in os.listdir(temp_folder):
                self.img_list.append(os.path.join(temp_folder, img))
                label = img.split(img_postfix)[0] + label_postfix
                self.label_list.append(os.path.join(self.label_path, folder, label))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label_path = self.label_list[index]
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = np.array(img)
        label = np.array(label)
        label_copy = label.copy()
        for k, v in trainid_to_trainid.items():
            label_copy[label == k] = v
        label = label_copy.astype(np.uint8)
        return img, label, img_name


def create_dataset(root, data='cityscapes', do_train=False, batch_size=1, num_parallel_workers=4):
    if data in ['cityscapes', 'c']:
        dataset = CityscapesData(root=os.path.join(root, 'Cityscapes'), do_train=do_train)
    elif data in ['bdd', 'bdd100k', 'b']:
        dataset = BDD100KData(root=os.path.join(root, 'BDD100k'), do_train=do_train)
    elif data in ['mapillary', 'm']:
        dataset = MapillaryData(root=os.path.join(root, 'Mapillary'), do_train=do_train)
    elif data in ['gtav', 'g']:
        dataset = GTAVData(root=os.path.join(root, 'GTA5'), do_train=do_train)
    elif data in ['synthia', 's']:
        dataset = SynthiaData(root=os.path.join(root, 'Synthia'), do_train=do_train)
    elif data in ['idd', 'i']:
        dataset = IDDData(root=os.path.join(root, 'IDD_Segmentation'), do_train=do_train)
    else:
        raise AttributeError('No this data!')
    dataset_column_names = ['image', 'label', 'img_name']
    ds = de.GeneratorDataset(
        dataset,
        olumn_names=dataset_column_names,
        num_parallel_workers=num_parallel_workers,
        shuffle=False,
        python_multiprocessing=False
    )
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        vision.HWC2CHW(),
    ]
    label_transform = transforms.TypeCast(ms.uint8)
    ds = ds.map(input_columns="image", operations=image_transforms, num_parallel_workers=num_parallel_workers)
    ds = ds.map(input_columns="label", operations=label_transform, num_parallel_workers=num_parallel_workers)
    ds = ds.batch(batch_size, drop_remainder=False)
    return ds
