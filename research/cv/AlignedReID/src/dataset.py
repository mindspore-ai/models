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
""" ReID dataset processing """

import os
import pickle
from collections import defaultdict

import mindspore.dataset as ds
import mindspore.dataset.vision as C
import numpy as np
from PIL import Image


def parse_im_name(im_name, parse_type='id'):
    """Get the person id or cam from an image name."""
    assert parse_type in ('id', 'cam')
    if parse_type == 'id':
        parsed = int(im_name[:8])
    else:
        parsed = int(im_name[9:13])
    return parsed


class TrainReIDSequenceDataset:
    """ Train dataset

    Args:
        im_dir: path to image folder
        partition_file: Path to partition pkl file
        ims_per_id: number of imager per id
        part: part of data for evaluation
        rank: process id
        group_size: device number
        seed: random seed
    """
    def __init__(
            self,
            im_dir=None,
            partition_file=None,
            ims_per_id=None,
            part='trainval',
            rank=0,
            group_size=1,
            seed=16562,
    ):
        # The im dir of all images
        self.im_dir = im_dir
        self.ims_per_id = ims_per_id
        self.group_size = group_size
        self.rank = rank

        self.im_names, self.ids2labels = self._get_dataset_values(partition_file, part)

        im_ids = [parse_im_name(name, 'id') for name in self.im_names]
        self.ids_to_im_inds = defaultdict(list)
        for ind, id_ in enumerate(im_ids):
            self.ids_to_im_inds[id_].append(ind)

        self.ids_orig = list(self.ids_to_im_inds.keys())
        self.ids = self.ids_orig.copy()

        self.image_queue, self.label_queue = None, None
        self.rng = np.random.RandomState(seed)
        self.reinit_queue(shuffle=False)

        self.index_shift = len(self) * self.rank

    @property
    def num_classes(self):
        """ Number of classes in dataset (0 for evaluation) """
        return len(self.ids2labels)

    @staticmethod
    def _get_dataset_values(partition_file, part='trainval'):
        """ Get images info from partition_file

        Args:
            partition_file: Path to partition pkl file
            part: part of data for evaluation

        Returns:
            image names
            map of ids to labels
        """
        with open(partition_file, 'rb') as f:
            partitions = pickle.load(f)
        im_names = partitions['{}_im_names'.format(part)]
        ids2labels = partitions['{}_ids2labels'.format(part)]
        return im_names, ids2labels

    def _get_sample(self, id_):
        """ Get images names sequence by id """
        inds = self.ids_to_im_inds[id_]
        replace = len(inds) < self.ims_per_id
        inds = self.rng.choice(inds, self.ims_per_id, replace=replace)
        im_names = [self.im_names[ind] for ind in inds]
        return im_names

    def reinit_queue(self, shuffle=True):
        """ Update image sequence and shuffle ids if shuffle=True """
        self.ids = self.ids_orig.copy()
        if shuffle:
            self.rng.shuffle(self.ids)
        self.image_queue = [self._get_sample(id_) for id_ in self.ids]
        self.label_queue = [[self.ids2labels[id_]] * self.ims_per_id for id_ in self.ids]

    def __getitem__(self, index):
        """ Get image and info by index

        Returns:
          image in numpy format
          image label
        """
        # Shuffle data on epoch start (dont work for distributed sampler)
        if index == 0:
            self.reinit_queue(shuffle=True)
        # Update index by rank
        index = self.index_shift + index
        # Get id index and image queue local index
        ind_id = index // self.ims_per_id
        ind_el = index % self.ims_per_id
        im_name = os.path.join(self.im_dir, self.image_queue[ind_id][ind_el])
        label = self.label_queue[ind_id][ind_el]

        img = Image.open(im_name).convert('RGB')
        img = np.asarray(img)

        return img, label

    def __len__(self):
        """ Dataset length """
        return len(self.image_queue) // self.group_size * self.ims_per_id


class ReIDSequenceDataset:
    """ Test dataset

    Args:
        im_dir: path to image folder
        partition_file: Path to partition pkl file
        part: part of data for evaluation
    """
    def __init__(
            self,
            im_dir=None,
            partition_file=None,
            part='test',
    ):
        # The im dir of all images
        self.im_dir = im_dir
        self.im_names, self.marks = self._get_dataset_values(partition_file, part)

    @property
    def num_classes(self):
        """ Number of classes in dataset (0 for evaluation) """
        return 0

    @staticmethod
    def _get_dataset_values(partition_file, part='test'):
        """ Get images info from partition_file

        Args:
            partition_file: Path to partition pkl file
            part: part of data for evaluation

        Returns:
            image names
            image marks
        """
        with open(partition_file, 'rb') as f:
            partitions = pickle.load(f)
        im_names = partitions['{}_im_names'.format(part)]
        marks = partitions['{}_marks'.format(part)]
        return im_names, marks

    def __getitem__(self, index):
        """ Get image and info by index

        Returns:
          image in numpy format
          image info:
            person id
            camera id
            image mark
        """
        name = self.im_names[index]
        im_name = os.path.join(self.im_dir, name)
        label = [
            parse_im_name(name, 'id'),
            parse_im_name(name, 'cam'),
            self.marks[index]
        ]
        img = Image.open(im_name).convert('RGB')
        img = np.asarray(img)

        return img, label

    def __len__(self):
        """ Dataset length """
        return len(self.im_names)


def statistic_normalize_img(img, statistic_norm, mean, std):
    """Statistic normalize images."""
    # img: RGB
    if isinstance(img, Image.Image):
        img = np.array(img)
    img = img / 255.
    if statistic_norm:
        img = (img - mean) / std
    return img


def _reshape_data(image, image_size, mean, std):
    """Reshape image."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    ori_w, ori_h = image.size
    ori_image_shape = np.array([ori_w, ori_h], np.int32)
    h, w = image_size

    image = image.resize((w, h), Image.BILINEAR)
    image_data = statistic_normalize_img(image, statistic_norm=True, mean=mean, std=std)
    if len(image_data.shape) == 2:
        image_data = np.expand_dims(image_data, axis=-1)
        image_data = np.concatenate([image_data, image_data, image_data], axis=-1)
    image_data = image_data.astype(np.float32)

    return image_data, ori_image_shape


def reshape_fn(image, label, image_size, mean, std):
    """ Function to process image and label """
    image, _ = _reshape_data(image, image_size=image_size, mean=mean, std=std)
    return image, label.astype(np.int32)


def create_dataset(
        image_folder,
        partition_file,
        ims_per_id=4,
        ids_per_batch=32,
        batch_size=None,
        rank=0,
        group_size=1,
        resize_h_w=(256, 128),
        num_parallel_workers=8,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        istrain=True,
        return_len=False,
):
    """ Create dataloader for ReID

    Args:
        image_folder: path to image folder
        partition_file: Path to partition pkl file
        ims_per_id: number of ids in batch
        ids_per_batch: number of imager per id
        batch_size: batch size (if None then batch_size=ims_per_id*ids_per_batch)
        rank: process id
        group_size: device number
        resize_h_w: height and width of image
        num_parallel_workers: number of parallel workers
        mean: image mean value for normalization
        std: image std value for normalization
        istrain: is train data
        return_len: return dataset len

    Returns:

    """
    part = 'trainval' if istrain else 'test'
    flip_prob = 0.5 if istrain else 0

    mean = np.array(mean)
    std = np.array(std)

    if batch_size is None:
        batch_size = ids_per_batch * ims_per_id

    if istrain:
        reid_dataset = TrainReIDSequenceDataset(
            image_folder,
            partition_file,
            ims_per_id=ims_per_id,
            part=part,
            rank=rank,
            group_size=group_size,
        )
    else:
        reid_dataset = ReIDSequenceDataset(
            image_folder,
            partition_file,
            part=part,
        )

    dataset = ds.GeneratorDataset(
        source=reid_dataset,
        column_names=['image', 'label'],
        shuffle=False,
    )

    compose_map_func = (lambda image, img_id: reshape_fn(image, img_id, resize_h_w, mean=mean, std=std))

    change_swap_op = C.HWC2CHW()

    if flip_prob > 0:
        rand_flip = C.RandomHorizontalFlip(flip_prob)
        trans = [rand_flip, change_swap_op]
    else:
        trans = [change_swap_op]

    dataset = dataset.map(
        operations=compose_map_func,
        input_columns=["image", "label"],
        output_columns=["image", "label"],
        num_parallel_workers=num_parallel_workers,
    )

    dataset = dataset.map(operations=trans, input_columns=["image"], num_parallel_workers=num_parallel_workers)

    dataset = dataset.batch(batch_size, drop_remainder=False)

    if return_len:
        return dataset, reid_dataset.num_classes, len(reid_dataset)

    return dataset, reid_dataset.num_classes
