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

"""
Use this file to generate train and val dataset
"""

import os
import os.path
import sys
from PIL import Image
import numpy as np
from numpy.random import randint
import mindspore.dataset as ds
from src.transforms import GroupNormalize, Stack, ToMindSporeFormatTensor, GroupScale, \
    GroupCenterCrop, GroupMultiScaleCrop, GroupRandomHorizontalFlip


class VideoRecord:
    """
    the util to generate data set.
    """

    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet():
    """
    to generate data set.
    """

    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):
        self.root_path = root_path
        dirname, _ = os.path.split(os.path.abspath(sys.argv[0]))
        self.list_file = os.path.join(dirname, list_file)
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.transform = transform

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def __getitem__(self, index):
        record = self.video_list[index]
        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        pilImgs, label = self.get(record, segment_indices)
        return pilImgs, label

    def __len__(self):
        return len(self.video_list)

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                              size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + 1

    def get(self, record, indices):
        """
        get record
        """

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                assert i < self.new_length
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1
        if self.transform:
            for t in self.transform:
                if isinstance(t, list):
                    for sub_t in t:
                        images = sub_t(images)
                else:
                    images = t(images)
        return images, record.label

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        if self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')
            return [x_img, y_img]
        raise ValueError("Unknown  {}".format(directory))


def create_dataset_train(args, rgb_read_format, input_size=224, data_length=1):
    """
    create train dataloader
    """
    train_augmentation = [GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                          GroupRandomHorizontalFlip(is_flow=False)]
    input_mean = [104, 117, 128]
    input_std = [1]
    normalize = GroupNormalize(input_mean, input_std)
    train_transforms = [train_augmentation,
                        Stack(roll=True),
                        ToMindSporeFormatTensor(div=False),
                        normalize]
    if args.modality in ["RGB", "RGBDiff"]:
        image_tmpl = args.rgb_prefix + rgb_read_format
    else:
        image_tmpl = args.flow_prefix + rgb_read_format

    train_dataset_generator = TSNDataSet("", args.train_list, num_segments=args.num_segments,
                                         new_length=data_length,
                                         modality=args.modality, transform=train_transforms,
                                         image_tmpl=image_tmpl)

    print("Train dataset generator length: ", len(train_dataset_generator))
    return train_dataset_generator


def create_dataset_val(args, rgb_read_format, input_size=224, data_length=1):
    """
    create val dataloader
    """
    input_mean = [104, 117, 128]
    input_std = [1]
    normalize = GroupNormalize(input_mean, input_std)
    crop_size = input_size
    scale_size = input_size * 256 // 224

    val_transforms = [GroupScale(int(scale_size)),
                      GroupCenterCrop(crop_size),
                      Stack(roll=True),
                      ToMindSporeFormatTensor(div=False),
                      normalize
                      ]
    if args.modality in ["RGB", "RGBDiff"]:
        image_tmpl = args.rgb_prefix + rgb_read_format
    else:
        image_tmpl = args.flow_prefix + rgb_read_format

    val_dataset_generator = TSNDataSet("", args.val_list, num_segments=args.num_segments,
                                       new_length=data_length,
                                       modality=args.modality,
                                       image_tmpl=image_tmpl,
                                       random_shift=False,
                                       transform=val_transforms)
    val_dataset = ds.GeneratorDataset(val_dataset_generator, ["image", "label"], shuffle=False)
    val_dataset = val_dataset.batch(args.batch_size, drop_remainder=True)
    return val_dataset
