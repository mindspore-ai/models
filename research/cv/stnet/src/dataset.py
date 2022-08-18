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
"""dataset"""
import os
import sys
import json
import logging
import random
import numbers
import warnings

import cv2
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


class UcfVideoDataset:
    """Load your own video classification dataset."""

    def __init__(self, dataset_root_path, mode='train', clip_len=5,
                 frame_sample_rate=2, crop_size=112, short_side_size=128,
                 new_height=128, new_width=171, keep_aspect_ratio=False,
                 num_segment=7, num_crop=1, test_num_segment=7, test_num_crop=3):
        self.dataset_root_path = dataset_root_path
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.dataset_samples, self.label_array = self.get_data_and_labels()
        if mode == 'test':
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))

    def random_horizontal_flip(self, clip):
        '''
        horizontal flip the image randomly, rate: 0.5
        '''
        if random.random() < 0.5:
            if isinstance(clip[0], np.ndarray):
                return [np.fliplr(img) for img in clip]
            if isinstance(clip[0], Image.Image):
                return [
                    img.transpose(Image.FLIP_LEFT_RIGHT) for img in clip
                ]
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            ' but got list of {0}'.format(type(clip[0])))
        return clip

    def center_crop(self, clip, size):
        """
        center_crop
        """
        if isinstance(size, numbers.Number):
            size = (size, size)
        h, w = size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, _ = clip[0].shape
        elif isinstance(clip[0], Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = int(round((im_w - w) / 2.))
        y1 = int(round((im_h - h) / 2.))
        if isinstance(clip[0], np.ndarray):
            cropped = [img[y1:y1 + h, x1:x1 + w, :] for img in clip]

        elif isinstance(clip[0], Image.Image):
            cropped = [
                img.crop((x1, y1, x1 + w, y1 + h)) for img in clip
            ]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        return cropped

    def random_crop(self, clip, size):
        """
        random_crop
        """
        if isinstance(size, numbers.Number):
            size = (size, size)
        h, w = size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, _ = clip[0].shape
        elif isinstance(clip[0], Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = random.randint(0, im_w - w)
        y1 = random.randint(0, im_h - h)

        if isinstance(clip[0], np.ndarray):
            cropped = [img[y1:y1 + h, x1:x1 + w, :] for img in clip]

        elif isinstance(clip[0], Image.Image):
            cropped = [
                img.crop((x1, y1, x1 + w, y1 + h)) for img in clip
            ]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        return cropped

    def random_resize_clip(self, clip, ratio=(3. / 4., 4. / 3.), interpolation='nearest'):
        '''
        random_resize_clip
        '''
        scaling_factor = random.uniform(ratio[0], ratio[1])

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, _ = clip[0].shape
        elif isinstance(clip[0], Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)

        return self.resize_clip(clip, new_size, interpolation)

    def resize_clip(self, clip, size, interpolation='bilinear'):
        '''
        resize the clip
        '''
        if isinstance(clip[0], np.ndarray):
            if isinstance(size, numbers.Number):
                im_h, im_w, _ = clip[0].shape
                # Min spatial dim already matches minimal size
                if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                       and im_h == size):
                    return clip
                new_h, new_w = get_resize_sizes(im_h, im_w, size)
                size = (new_w, new_h)
            else:
                size = size[0], size[1]
            if interpolation == 'bilinear':
                np_inter = cv2.INTER_LINEAR
            else:
                np_inter = cv2.INTER_NEAREST
            scaled = [
                cv2.resize(img, size, interpolation=np_inter) for img in clip
            ]
        elif isinstance(clip[0], Image.Image):
            if isinstance(size, numbers.Number):
                im_w, im_h = clip[0].size
                # Min spatial dim already matches minimal size
                if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                       and im_h == size):
                    return clip
                new_h, new_w = get_resize_sizes(im_h, im_w, size)
                size = (new_w, new_h)
            else:
                size = size[1], size[0]
            if interpolation == 'bilinear':
                pil_inter = Image.BILINEAR
            else:
                pil_inter = Image.NEAREST
            scaled = [img.resize(size, pil_inter) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        return scaled

    def __getitem__(self, index):
        if self.mode == 'train':
            scale_t = 1

            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)
            if not buffer.any():
                while not buffer.any():
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)
            buffer = self.resize_clip(buffer, self.short_side_size, "bilinear")
            buffer = self.random_resize_clip(buffer, ratio=(1, 1.25), interpolation='bilinear')
            buffer = self.random_crop(buffer, (int(self.crop_size), int(self.crop_size)))
            buffer = self.random_horizontal_flip(buffer)
            buffer = self.clipToTensor(np.array(buffer))
            buffer = self.normalize(buffer, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            return buffer.astype(np.float32), self.label_array[index]

        if self.mode == 'val':
            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample)
            if not buffer.any():
                while not buffer.any():
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample)
            buffer = self.resize_clip(buffer, self.short_side_size, "bilinear")
            buffer = self.center_crop(buffer, size=(self.crop_size, self.crop_size))
            buffer = self.clipToTensor(np.array(buffer))
            buffer = self.normalize(buffer, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            return buffer.astype(np.float32), self.label_array[index]

        if self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.loadvideo_decord(sample)
            #print()
            while not buffer.any():
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(\
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.loadvideo_decord(sample)

            buffer = self.resize_clip(buffer, self.short_side_size, "bilinear")
            buffer = self.center_crop(buffer, size=(self.crop_size, self.crop_size))
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                 / (self.test_num_crop - 1)
            temporal_step = max(1.0 * (buffer.shape[0] - self.clip_len) \
                                / (self.test_num_segment - 1), 0)
            temporal_start = int(chunk_nb * temporal_step)
            spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                       spatial_start:spatial_start + self.short_side_size, :, :]
            else:
                buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                       :, spatial_start:spatial_start + self.short_side_size, :]

            buffer = self.clipToTensor(buffer)
            buffer = self.normalize(buffer, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            return buffer.astype(np.float32), self.test_label_array[index]
        raise NameError('mode {} unkown'.format(self.mode))

    def normalize(self, buffer, mean, std):
        for i in range(3):
            buffer[i] = (buffer[i] - mean[i]) / std[i]
        return buffer

    def clipToTensor(self, buffer):
        #m (H x W x C) --> #(C x m x H x W)
        return buffer.transpose((3, 0, 1, 2)) / 255.0

    def loadvideo_decord(self, sample, sample_rate_scale=1):
        """
        load images by cv2
        if mode == 'test', we return a whole list, it will be selected by chunk_nb, split_nb in __getitem__;
        otherwise, we select 'num_segment' list segment, other images in list will be discarded
        """
        frames = sorted([os.path.join(sample, img) for img in os.listdir(sample)])
        frame_count = len(frames)
        frame_list = np.empty((frame_count, self.new_height, self.new_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            frame_list[i] = frame

        if self.mode == 'test':
            all_index = [x for x in range(0, frame_count, self.frame_sample_rate)]
            while len(all_index) < self.clip_len:
                all_index.append(all_index[-1])
            buffer = frame_list[all_index]
            return buffer

        # handle temporal segments
        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = frame_count // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len:
                index = np.linspace(0, seg_len, num=seg_len // self.frame_sample_rate)
                index = np.concatenate((index, np.ones(self.clip_len - seg_len // self.frame_sample_rate) * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i*seg_len
            all_index.extend(list(index))

        all_index = all_index[::int(sample_rate_scale)]
        buffer = frame_list[all_index]
        return buffer

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        return len(self.test_dataset)

    def get_data_and_labels(self):
        myclass = self.classes
        dataset_samples, label_array = [], []
        split_class = "train" if self.mode == "train" else "val"
        for index, category in enumerate(sorted(os.listdir(os.path.join(self.dataset_root_path, split_class)))):
            assert category in myclass, str(category)+ " not belong to UCF101 dataset"
            for video in sorted(os.listdir(os.path.join(self.dataset_root_path, split_class, category))):
                dataset_samples.append(os.path.join(self.dataset_root_path, split_class, category, video))
                label_array.append(index)
        return dataset_samples, label_array

    @property
    def classes(self):
        """Category names."""
        return ucf101_label_names


class KineticsVideoDataset():
    """VideoDataSet"""
    def __init__(self, video_info_list, config, data_dir, mode='train'):
        self.video_info_list = video_info_list
        self.mode = mode
        self.short_size = config.short_size
        self.target_size = config.target_size
        self.img_mean = np.array(config.image_mean).reshape(
            [3, 1, 1]).astype(np.float32)
        self.img_std = np.array(config.image_std).reshape(
            [3, 1, 1]).astype(np.float32)
        self.T = config.T
        self.N = config.N
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
        # print(f"DEBUG: [dataset.py] len(frames), type(frames) = {len(frames), type(frames)}")
        imgs = video_loader(frames, self.T, self.N, self.mode)  # np.ndarray of shape (T*N, W, H, 3)
        ret_label = np.array(label_pkl).astype(np.int32)
        return imgs_transform(imgs, self.mode, self.T, self.N, self.short_size, self.target_size, self.img_mean,
                              self.img_std), ret_label
        # return imgs, ret_label

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
    imgs = np.transpose(imgs, (1, 0, 2, 3))  # shape (3, T*N, W, H)

    return imgs


def create_dataset(data_dir, config, shuffle=True, num_worker=4, do_trains=True, list_path=None):
    """create_dataset"""
    dataset_type = config.dataset_type
    assert dataset_type in ["ucf101", "kinetics400"]

    rank_id = config.rank_id
    device_num = config.device_num
    if config.target == "Ascend":
        device_num, rank_id = _get_rank_info()
    batch_size = config.batch_size

    if dataset_type == 'ucf101':
        if do_trains:
            dataset = UcfVideoDataset(dataset_root_path=data_dir,
                                      mode='train')
        else:
            dataset = UcfVideoDataset(dataset_root_path=data_dir,
                                      mode='val')
    else:
        if do_trains:
            if config.run_online:
                data_path = list_path
                data_dir = config.local_data_url
            else:
                data_path = os.path.join(data_dir, 'train.json')
            video_info_list = read_json(data_path)
            dataset = KineticsVideoDataset(video_info_list=video_info_list, data_dir=data_dir, config=config)
        else:
            if config.run_online:
                data_path = list_path
                data_dir = config.local_data_url
            else:
                data_path = os.path.join(data_dir, 'val.json')
            video_info_list = read_json(data_path)
            dataset = KineticsVideoDataset(video_info_list=video_info_list, config=config,
                                           data_dir=data_dir, mode='val')

    if device_num > 1:
        de_dataset = de.GeneratorDataset(source=dataset, column_names=["data", "label"], shuffle=shuffle,
                                         num_parallel_workers=num_worker, num_shards=device_num,
                                         shard_id=rank_id, max_rowsize=21)
        de_dataset = de_dataset.batch(batch_size, drop_remainder=True)
    else:
        de_dataset = de.GeneratorDataset(source=dataset, column_names=["data", "label"], shuffle=shuffle,
                                         num_parallel_workers=num_worker, max_rowsize=21)
        de_dataset = de_dataset.batch(batch_size, drop_remainder=True)

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
        img = Image.open(StringIO(buf)).convert('RGB')
    else:
        img = Image.open(BytesIO(buf)).convert('RGB')
    # img_np = np.array(img).transpose(2, 0, 1)
    # img_np = img_np[:, :, ::-1].copy()  # convert RGB to BGR
    return img


def video_loader(frames, nsample, seglen, mode):
    """video_loader"""
    videolen = len(frames)
    if videolen == 0:
        return []
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
    # imgs_np = np.array(imgs)
    return imgs


ucf101_label_names = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', \
                      'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', \
                      'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', \
                      'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', \
                      'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen', \
                      'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', \
                      'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'HammerThrow', \
                      'Hammering', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump', \
                      'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', \
                      'JugglingBalls', 'JumpRope', 'JumpingJack', 'Kayaking', 'Knitting', \
                      'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', \
                      'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf', \
                      'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', \
                      'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', \
                      'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', \
                      'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding', \
                      'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', \
                      'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', \
                      'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing', \
                      'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', \
                      'YoYo']
