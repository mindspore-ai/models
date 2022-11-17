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

"""
A function that returns a dataset for classification.
"""
import random
import os

from PIL import Image, ImageFile
from mindspore import dtype as mstype
import mindspore.dataset as de
import mindspore.dataset.vision as vision_C
import mindspore.dataset.transforms as normal_C
from src.datasets.sampler import DistributedSampler
from src.model_utils.matlab_cp2tform import get_similarity_transform_for_cv2
import cv2
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

def alignment(src_img, src_pts):
    of = 2
    ref_pts = [[30.2946+of, 51.6963+of], [65.5318+of, 51.5014+of],
               [48.0252+of, 71.7366+of], [33.5493+of, 92.3655+of], [62.7299+of, 92.2041+of]]
    crop_size = (96+of*2, 112+of*2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

class TxtDataset():
    """
    read dataset from txt
    """
    def __init__(self, root, txt_name):
        super(TxtDataset, self).__init__()
        self.imgs = []
        self.labels = []
        self.ptsrem = []
        self.num = 0
        fin = open(txt_name, "r")
        num = 0
        for line in fin:
            src_pts = []
            num = num+1
            lets = line.split('\t')
            for i in range(5):
                src_pts.append([int(lets[2*i+2]), int(lets[2*i+3])])
            self.ptsrem.append(src_pts)
            img_name = lets[0]
            img_name = img_name
            label = lets[1]
            self.imgs.append(os.path.join(root, img_name))
            self.labels.append(int(label))
        fin.close()

    def __getitem__(self, index):
        img = np.array(Image.open(self.imgs[index]).convert('RGB'), np.float32)
        img = img[:, :, ::-1]
        img = alignment(img, self.ptsrem[index])
        if random.random() > 0.5:
            rx = random.randint(0, 2*2)
            ry = random.randint(0, 2*2)
            img = img[ry:ry+112, rx:rx+96, :]
        else:
            img = img[2:2+112, 2:2+96, :]
        img = (img-127.5)/128
        return img, self.labels[index]
    def __len__(self):
        return len(self.imgs)


def classification_dataset_imagenet(data_dir, image_size, per_batch_size, max_epoch, rank, group_size, mode='train',
                                    input_mode='folder', root='', num_parallel_workers=None, shuffle=None,
                                    sampler=None, class_indexing=None, drop_remainder=True, transform=None,
                                    target_transform=None):

    if transform is None:
        if mode == 'train':
            transform_img = [
                vision_C.RandomColorAdjust(brightness=0.4, saturation=0.4),
                #vision_C.GaussianBlur((3,3),0.05),
                vision_C.RandomHorizontalFlip(prob=0.5),
                vision_C.HWC2CHW()
            ]
        else:
            transform_img = [
                vision_C.Resize((112, 96)),
                vision_C.HWC2CHW()
            ]
    else:
        transform_img = transform

    if target_transform is None:
        transform_label = [
            normal_C.TypeCast(mstype.int32)
        ]
    else:
        transform_label = target_transform


    dataset = TxtDataset(root, data_dir)
    sampler = DistributedSampler(dataset, rank, group_size, shuffle=shuffle)
    de_dataset = de.GeneratorDataset(dataset, ["image", "label"], sampler=sampler)

    de_dataset = de_dataset.map(input_columns="image", num_parallel_workers=8, operations=transform_img)
    de_dataset = de_dataset.map(input_columns="label", num_parallel_workers=8, operations=transform_label)

    columns_to_project = ["image", "label"]
    de_dataset = de_dataset.project(columns=columns_to_project)

    de_dataset = de_dataset.batch(per_batch_size, drop_remainder=drop_remainder)
    de_dataset = de_dataset.repeat(1)
    return de_dataset
