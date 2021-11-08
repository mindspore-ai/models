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
create dataset
"""
import os
import os.path
import numpy as np
from PIL import Image

import mindspore.dataset as ds

class VideoRecord:
    """Video files"""
    def __init__(self, root_path, row):
        self.root_path = root_path
        self._data = row

    @property
    def path(self):
        return os.path.join(self.root_path, self._data[0])

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet:
    """TSN dataset"""
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        image = []
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            image = [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            image = [x_img, y_img]
        return image

    def _parse_list(self):
        self.video_list = [VideoRecord(self.root_path, x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:

            offsets = np.multiply(list(range(self.num_segments)), average_duration) \
                 + np.random.randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(np.random.randint(record.num_frames - self.new_length + 1, size=self.num_segments))
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

    def __getitem__(self, index):
        record = self.video_list[index]
        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        """get data"""
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for _ in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        for process in self.transform:
            images = process(images)
        return images, record.label

    def __len__(self):
        return len(self.video_list)


class Get_Diff:
    """Multi scale transform."""
    def __init__(self, modality, new_length, num_segments, keep_rgb=False):
        self.modality = modality
        self.new_length = new_length
        self.keep_rgb = keep_rgb
        self.reverse = list(range(self.new_length, 0, -1))

        self.num_segments = num_segments
        self.input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2

    def __call__(self, image, label, batch_info):

        image = np.array(image)
        input_view = image.reshape((-1, self.num_segments, self.new_length + 1, self.input_c,) + image.shape[2:])
        if self.keep_rgb:
            new_data = input_view.copy()
        else:
            new_data = input_view[:, :, 1:, :, :, :].copy()

        for x in self.reverse:
            if self.keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
        return np.array(new_data), np.array(label)


def create_dataset(root_path, list_file, batch_size, num_segments=3, new_length=1, modality='RGB',\
     image_tmpl='img_{:05d}.jpg', random_shift=True, test_mode=0, transform=None, run_distribute=False,\
          worker=8, num_shards=1, shard_id=0):
    """
    :param test_mode: 0 train; 1 eval; 2 test;
    :return: list
    """
    data = TSNDataSet(root_path, list_file, num_segments=num_segments, new_length=new_length,\
         modality=modality, image_tmpl=image_tmpl, random_shift=random_shift, test_mode=test_mode, transform=transform)
    if test_mode:
        shuffle = False
    else:
        shuffle = True
    if run_distribute:
        dataset = ds.GeneratorDataset(data, column_names=["input", "label"], num_parallel_workers=worker,\
             shuffle=shuffle, num_shards=num_shards, shard_id=shard_id)
    else:
        dataset = ds.GeneratorDataset(data, column_names=["input", "label"],\
             num_parallel_workers=worker, shuffle=shuffle)

    if modality == "RGBDiff" and test_mode == 0:
        getdiff = Get_Diff(modality, new_length, num_segments)
        dataset = dataset.batch(batch_size=batch_size, per_batch_map=getdiff, input_columns=["input", "label"],\
             num_parallel_workers=worker)
    else:
        dataset = dataset.batch(batch_size)

    return dataset
