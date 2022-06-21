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
'''dataset'''
import os
import random
import cv2
import numpy as np
import mindspore
import mindspore.dataset as ds
import mindspore.dataset.vision as C2
import mindspore.dataset.transforms as C
mindspore.set_seed(1)

def prepare_image_cv2(im):
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im

def prepare_label_cv2(im):
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im

def get_imageId_from_fileName(filename, id_iter):
    """Get imageID from fileName if fileName is int, else return id_iter."""
    filename = os.path.splitext(filename)[0]
    if filename.isdigit():
        return int(filename)
    return id_iter

def rand_crop_pad(data, label):
    scale1 = 1.0
    scale2 = 1.5
    img_w = 481
    img_h = 321
    sc = np.random.uniform(scale1, scale2)
    new_h, new_w = int(sc * data.shape[0]), int(sc * data.shape[1])
    data = cv2.resize(data, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    mask = np.ones((1, new_h, new_w), np.float32)
    if data.shape[1] < img_w:
        w = img_w - data.shape[1]
        data = np.pad(data, ((0, 0), (0, w), (0, 0)), 'constant', constant_values=(0, 0))
        label = np.pad(label, ((0, 0), (0, w), (0, 0)), 'constant', constant_values=(1, 1))
        mask = mask = np.pad(mask, ((0, 0), (0, 0), (0, w)), 'constant', constant_values=(0, 0))
        width1 = 0
        width2 = img_w
    else:
        width1 = random.randint(0, data.shape[1] - img_w)
        width2 = width1 + img_w
    if data.shape[0] < img_h:
        h = img_h - data.shape[0]
        data = np.pad(data, ((0, h), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))
        label = np.pad(label, ((0, h), (0, 0), (0, 0)), 'constant', constant_values=(1, 1))
        mask = mask = np.pad(mask, ((0, 0), (0, h), (0, 0)), 'constant', constant_values=(0, 0))
        height1 = 0
        height2 = img_h
    else:
        height1 = random.randint(0, data.shape[0] - img_h)
        height2 = height1 + img_h

    data = data[height1:height2, width1:width2, :]
    label = label[height1:height2, width1:width2, :]
    mask = mask[:, height1:height2, width1:width2]
    return data, label, mask

def create_dataset(data_path, is_training=True, is_shuffle=True, batch_size=1, repeat_size=1,
                   device_num=1, rank=0, num_parallel_workers=24):
    """create dataset for train or test"""
    dataset = HED_Dataset_t(data_path, is_training)
    print("total patch numbers per epoch", len(dataset))
    if is_training:
        trans1 = [
            C2.Normalize(mean=[104.00698793, 116.66876762, 122.67891434], std=[1, 1, 1]),
            C2.HWC2CHW(),
            C.TypeCast(mindspore.float32)
        ]
        dataloader = ds.GeneratorDataset(dataset, ['image', 'label', 'mask'], num_parallel_workers=num_parallel_workers,
                                         shuffle=is_shuffle, num_shards=device_num, shard_id=rank)
        # apply map operations on images
        dataloader = dataloader.map(input_columns='image', operations=trans1,
                                    num_parallel_workers=num_parallel_workers)
        dataloader = dataloader.map(input_columns='label', operations=C.TypeCast(mindspore.float32),
                                    num_parallel_workers=num_parallel_workers)
        dataloader = dataloader.map(input_columns='mask', operations=C.TypeCast(mindspore.float32),
                                    num_parallel_workers=num_parallel_workers)
        dataloader = dataloader.batch(batch_size, drop_remainder=True)
    else:
        dataset = HED_Dataset_e(data_path, is_training)
        dataloader = ds.GeneratorDataset(dataset, ['test', 'label'], num_parallel_workers=8, shuffle=is_shuffle)
        dataloader = dataloader.map(input_columns='test', operations=C.TypeCast(mindspore.float32))
        dataloader = dataloader.batch(batch_size, drop_remainder=False)

    # apply DatasetOps
    dataloader = dataloader.repeat(repeat_size)

    return dataloader

class HED_Dataset_t():
    '''hed_dataset'''
    def __init__(self, dataset_path, is_training=True):
        if not os.path.exists(dataset_path):
            raise RuntimeError("the input image dir {} is invalid!".format(dataset_path))
        self.dataset_path = dataset_path
        self.is_training = is_training
        with open(self.dataset_path, 'r') as data_f:
            self.filelist = data_f.readlines()

    def __getitem__(self, index):
        if self.is_training:
            img_file, lb_file = self.filelist[index].split()
            lb = np.array(cv2.imread(lb_file), dtype=np.float32)
            lb[lb < 127.5] = 0.0
            lb[lb >= 127.5] = 1.0
            img_file = np.array(cv2.imread(img_file), dtype=np.float32)
            if img_file.shape[0] > img_file.shape[1]:
                img_file = np.rot90(img_file, 1).copy()
            if lb.shape[0] > lb.shape[1]:
                lb = np.rot90(lb, 1).copy()
            img_file, lb, mask = rand_crop_pad(img_file, lb)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            lb = lb[np.newaxis, :, :]
        return img_file, lb, mask

    def __len__(self):
        return len(self.filelist)

class HED_Dataset_e():
    '''hed_dataset'''
    def __init__(self, dataset_path, is_training=True):
        if not os.path.exists(dataset_path):
            raise RuntimeError("the input image dir {} is invalid!".format(dataset_path))
        self.dataset_path = dataset_path
        self.is_training = is_training
        with open(self.dataset_path, 'r') as data_f:
            self.filelist = data_f.readlines()

    def __getitem__(self, index):
        if not self.is_training:
            img_file, lb = self.filelist[index].split()
            img_file = np.array(cv2.imread(img_file), dtype=np.float32)
            if img_file.shape[0] > img_file.shape[1]:
                img_file = np.rot90(img_file, 1).copy()
            img_file -= np.array((104.00698793, 116.66876762, 122.67891434))
            img_file = np.transpose(img_file, (2, 0, 1))
        return img_file, lb

    def __len__(self):
        return len(self.filelist)

if __name__ == '__main__':
    from eval import get_files
    from src.model_utils.config import config

    test_img = get_files(os.path.join(config.data_path, 'BSDS500/data/images/test'),
                         extension_filter='.jpg')
    test_label = get_files(os.path.join(config.data_path, 'BSDS500/data/labels/test'),
                           extension_filter='.jpg')
    test_path = os.path.join(config.data_path, 'output/test.lst')

    f = open(test_path, "w")
    for img_f, label_f in zip(test_img, test_label):
        f.write(str(img_f) + " " + str(label_f))
        f.write('\n')

    test_loader = create_dataset(test_path, is_training=False, is_shuffle=False)
    dataset_size = test_loader.get_dataset_size()

    data_shape = (0, 0)
    i = 1
    for data_fi in test_loader.create_dict_iterator(output_numpy=True, num_epochs=1):
        if data_shape == data_fi['test'].shape:
            print("i: {}/{}, data_shape equal.".format(i, dataset_size))
        else:
            data_shape = data_fi['test'].shape
            print("i: {}/{}, data['test'].shape: ".format(i, dataset_size), data_fi['test'].shape)
        i += 1
