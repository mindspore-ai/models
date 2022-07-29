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
"""
Read and process the dataset
"""

import copy
import json
import math
import os

import cv2
import mindspore
import numpy as np

from src.utils import load_value_file


def rgb_video_loader(video_dir_path, frame_indices):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(cv2.imread(image_path))
        else:
            return video
    return video


def flow_video_loader(video_dir_path, frame_indices):
    videox = []
    videoy = []
    for i in frame_indices:
        image_path_x = os.path.join(video_dir_path, 'image_{:05d}x.jpg'.format(i))
        image_path_y = os.path.join(video_dir_path, 'image_{:05d}y.jpg'.format(i))
        if os.path.exists(image_path_x) and os.path.exists(image_path_y):
            videox.append(cv2.imread(image_path_x, cv2.IMREAD_GRAYSCALE))
            videoy.append(cv2.imread(image_path_y, cv2.IMREAD_GRAYSCALE))
        else:
            return videox, videoy
    return videox, videoy


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)

    if not annotations:
        raise ValueError('Unable to load annotations...')

    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('Loading UCF-101 videos [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            continue

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        if not os.path.exists(n_frames_file_path):
            raise FileNotFoundError('n_frames_file_path does not exist: {}'.format(n_frames_file_path))

        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1]
        }
        if annotations:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class


class get_loader:

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 mode,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=64):

        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.mode = mode

    @staticmethod
    def pixel_transform(img):
        return (img / 255.) * 2 - 1

    @staticmethod
    def joint_flow_imgs(clipx, clipy):
        clip = []
        if len(clipx) != len(clipy):
            raise Exception("List length exception!", len(clipx), len(clipy))
        for i in range(len(clipx)):
            img = np.asarray([clipx[i], clipy[i]]).transpose([1, 2, 0])
            clip.append(img)

        return clip

    def __getitem__(self, index):

        path = self.data[index]['video']
        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        if self.mode == 'rgb':
            clip = rgb_video_loader(path, frame_indices)
            clip = [self.pixel_transform(img) for img in clip]
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
            target = self.data[index]
            if self.target_transform is not None:
                target = self.target_transform(target)

        if self.mode == 'flow':
            clipx, clipy = flow_video_loader(path, frame_indices)
            clipx = [self.pixel_transform(img) for img in clipx]
            clipy = [self.pixel_transform(img) for img in clipy]
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clipx = [self.spatial_transform(img) for img in clipx]
                clipy = [self.spatial_transform(img) for img in clipy]
            clip = self.joint_flow_imgs(clipx, clipy)
            target = self.data[index]
            if self.target_transform is not None:
                target = self.target_transform(target)

        clip = [img.astype(np.float32) for img in clip]
        clip = np.stack(clip, axis=0).transpose(3, 0, 1, 2)
        target = mindspore.Tensor(target, mindspore.int32)
        target = target.asnumpy()

        return (clip, target)

    def __len__(self):
        return len(self.data)
