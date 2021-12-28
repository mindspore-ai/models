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
"""dataset"""
import os
import sys
import json
import logging
import random
import numpy as np
import mindspore.dataset as de
from mindspore.communication import get_rank, get_group_size
from PIL import Image


try:
    import cPickle as pickle
    from cStringIO import StringIO
except ImportError:
    import pickle
    from io import BytesIO
logger = logging.getLogger(__name__)

python_ver = sys.version_info


def read_json(filename):
    with open(filename, 'r') as file:
        info = json.load(file)
    return info


class VideoDataSet():
    """VideoDataSet"""
    def __init__(self, video_info_list, config, data_dir, mode='train'):
        self.video_info_list = video_info_list
        self.mode = mode
        self.short_size = config['short_size']
        self.target_size = config['target_size']
        self.img_mean = np.array(config['image_mean']).reshape(
            [3, 1, 1]).astype(np.float32)
        self.img_std = np.array(config['image_std']).reshape(
            [3, 1, 1]).astype(np.float32)
        self.T = config['T']
        self.N = config['N']
        self.data_dir = data_dir

    def __getitem__(self, index):
        video_info = self.video_info_list[index]
        video_name = video_info['video_name']
        video_file = video_name[0:-4]

        video_path = os.path.join(self.data_dir, self.mode, video_file) + ".pkl"
        if python_ver < (3, 0):
            data_loader = pickle.load(open(video_path, 'rb'))
        else:
            data_loader = pickle.load(open(video_path, 'rb'), encoding='bytes')
        _, label_pkl, frames = data_loader
        imgs = video_loader(frames, self.T, self.N, self.mode)
        ret_label = np.array(label_pkl).astype(np.int32)
        return imgs_transform(imgs, self.mode, self.T, self.N, self.short_size, self.target_size, self.img_mean,
                              self.img_std), ret_label

    def __len__(self):
        return len(self.video_info_list)


def imgs_transform(imgs, mode, segnum, seglen, short_size, target_size, img_mean, img_std):
    """transform"""
    images = imgs
    imgs = group_scale(images, short_size)
    if mode == 'train':
        imgs = group_random_crop(imgs, target_size)
        imgs = group_random_flip(imgs)
    else:
        imgs = group_center_crop(imgs, target_size)

    np_imgs = np.array(imgs[0]).astype('float32')
    np_imgs = np_imgs.transpose((2, 0, 1))
    np_imgs = np.reshape(np_imgs, (1, 3, target_size, target_size))
    np_imgs = np_imgs / 255
    for i in range(len(imgs) - 1):
        img = np.array(imgs[i + 1]).astype('float32')
        img = img.transpose((2, 0, 1))
        img = np.reshape(img, (1, 3, target_size, target_size))
        img = img / 255
        np_imgs = np.concatenate((np_imgs, img))
    imgs = np_imgs
    imgs -= img_mean
    imgs /= img_std
    imgs = np.reshape(imgs, (segnum, seglen * 3, target_size, target_size))

    return imgs


def create_dataset(data_dir, config, shuffle=True, num_worker=1, do_trains='train', list_path=None):
    """create_dataset"""
    rank_id = None
    device_num = 1
    if config['target'] == "Ascend":
        device_num, rank_id = _get_rank_info()
    batch_size = config['batch_size']

    if do_trains == 'train':
        if config['run_online']:
            data_path = list_path
            data_dir = config['local_data_url']
        else:
            data_path = data_dir + 'train.json'
        video_info_list = read_json(data_path)
        dataset = VideoDataSet(video_info_list=video_info_list, data_dir=data_dir, config=config)
    else:
        if config['run_online']:
            data_path = list_path
            data_dir = config['local_data_url']
        else:
            data_path = data_dir + 'val.json'
        video_info_list = read_json(data_path)
        dataset = VideoDataSet(video_info_list=video_info_list, config=config, data_dir=data_dir, mode='val')

    if device_num > 1:
        de_dataset = de.GeneratorDataset(source=dataset, column_names=["data", "label"], shuffle=shuffle,
                                         num_parallel_workers=num_worker, num_shards=device_num, shard_id=rank_id)
        de_dataset = de_dataset.batch(batch_size, num_parallel_workers=num_worker, drop_remainder=shuffle)
    else:
        de_dataset = de.GeneratorDataset(source=dataset, column_names=["data", "label"], shuffle=shuffle,
                                         num_parallel_workers=num_worker)
        de_dataset = de_dataset.batch(batch_size, drop_remainder=shuffle)

    return de_dataset


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id


def group_random_crop(img_group, target_size):
    """group_random_crop"""
    w, h = img_group[0].size
    th, tw = target_size, target_size

    out_images = []
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    for img in img_group:
        if w == tw and h == th:
            out_images.append(img)
        else:
            out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return out_images


def group_random_flip(img_group):
    v = random.random()
    if v < 0.5:
        ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
        return ret
    return img_group


def group_center_crop(img_group, target_size):
    """group_center_crop"""
    img_crop = []
    for img in img_group:
        w, h = img.size
        th, tw = target_size, target_size

        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img_crop.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return img_crop


def group_scale(imgs, target_size):
    """group_scale"""
    resized_imgs = []
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.size
        if (w <= h and w == target_size) or (h <= w and h == target_size):
            resized_imgs.append(img)
            continue

        if w < h:
            ow = target_size
            oh = int(target_size * 4.0 / 3.0)
            resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))
        else:
            oh = target_size
            ow = int(target_size * 4.0 / 3.0)
            resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))

    return resized_imgs


def imageloader(buf):
    if isinstance(buf, str):
        img = Image.open(StringIO(buf))
    else:
        img = Image.open(BytesIO(buf))

    return img.convert('RGB')


def video_loader(frames, nsample, seglen, mode):
    """video_loader"""
    videolen = len(frames)
    average_dur = int(videolen / nsample)
    imgs = []
    for i in range(nsample):
        idx = 0
        if mode == 'train':
            if average_dur >= seglen:
                idx = random.randint(0, average_dur - seglen)
                idx += i*average_dur
            elif average_dur >= 1:
                idx += i*average_dur
            else:
                idx = i
        else:
            if average_dur >= seglen:
                idx = (average_dur - seglen)//2
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i
        for jj in range(idx, idx + seglen):
            imgbuf = frames[int(jj % videolen)]
            img = imageloader(imgbuf)
            imgs.append(img)

    return imgs
