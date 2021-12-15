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
""""Dataset"""
import os
from PIL import Image
import numpy as np
from numpy.random import randint
from mindspore.dataset import GeneratorDataset, DistributedSampler
from src.utils.transforms import GroupNormalize, GroupScale, GroupCenterCrop, Stack


class VideoRecord():
    """"Dataset"""
    def __init__(self, row):
        self.video_data = row

    @property
    def path(self):
        return self.video_data[0]

    @property
    def num_frames(self):
        return int(self.video_data[1])

    @property
    def label(self):
        return int(self.video_data[2])


class TSMDataSet:
    """"TSMDataset"""
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        """"Dataset"""
        if os.path.exists(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))):
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]

        print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
        return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]

    def _parse_list(self):
        """"_parse_list"""
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]
        self.video_list = [VideoRecord(item) for item in tmp]

        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            for v in self.video_list:
                v.video_data[1] = int(v.video_data[1]) / 2
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets)
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_val_indices(self, record):
        """_get_val_indices"""
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets)

        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record):
        """_get_test_indices"""
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets)
        if self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets
        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets

    def _trans(self, images, transform):
        if isinstance(transform, list):
            for t in transform:
                images = self._trans(images, t)
            return images
        if transform is not None:
            return transform(images)
        return images

    def __getitem__(self, index):
        # print(index)
        record = self.video_list[index]
        # check this is a legit video folder
        if self.image_tmpl == 'flow_{}_{:05d}.jpg':
            file_name = self.image_tmpl.format('x', 1)
            full_path = os.path.join(self.root_path, record.path, file_name)
        elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            file_name = self.image_tmpl.format(int(record.path), 'x', 1)
            full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
        else:
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(self.root_path, record.path, file_name)

        while not os.path.exists(full_path):
            print('################## Not Found:', os.path.join(self.root_path, record.path, file_name))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':
                file_name = self.image_tmpl.format('x', 1)
                full_path = os.path.join(self.root_path, record.path, file_name)
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                file_name = self.image_tmpl.format(int(record.path), 'x', 1)
                full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
            else:
                file_name = self.image_tmpl.format(1)
                full_path = os.path.join(self.root_path, record.path, file_name)

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        """get"""
        images = list()
        for seg_ind in indices:
            # print(seg_ind)
            p = int(seg_ind)
            for _ in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)

                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1
        images = self._trans(images, self.transform)
        return np.array(images).astype(np.float32), np.array(record.label).astype(np.int32)

    def __len__(self):
        return len(self.video_list)


def get_datasets(args, rank, input_mean, train_augmentation, input_std, scale_size, crop_size, prefix):
    """get_datasets"""
    train_transform = [train_augmentation, Stack(roll=False),
                       GroupNormalize(input_mean, input_std)]
    train_dataset = TSMDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                               new_length=1, modality=args.modality, image_tmpl=prefix,
                               transform=train_transform, dense_sample=args.dense_sample)

    train_sampler = DistributedSampler(args.gpus, rank, True)
    args.train_batches = len(train_dataset) // args.batch_size

    train_loader = GeneratorDataset(
        source=train_dataset, sampler=train_sampler, python_multiprocessing=True,
        num_parallel_workers=args.workers, column_names=['frames', 'label'])
    train_loader = train_loader.batch(args.batch_size, drop_remainder=True)

    test_transform = [GroupScale(size=(int(scale_size), int(scale_size))), GroupCenterCrop(crop_size),
                      Stack(roll=False), GroupNormalize(input_mean, input_std)]

    val_dataset = TSMDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                             new_length=1, modality=args.modality, image_tmpl=prefix, random_shift=False,
                             transform=test_transform, dense_sample=args.dense_sample)
    val_sampler = DistributedSampler(args.gpus, rank, False)
    args.val_batches = len(val_dataset) // args.batch_size
    val_loader = GeneratorDataset(
        source=val_dataset, sampler=val_sampler,
        num_parallel_workers=args.workers, column_names=['frames', 'label'])
    val_loader = val_loader.batch(args.batch_size, drop_remainder=True)
    return train_loader, val_loader
