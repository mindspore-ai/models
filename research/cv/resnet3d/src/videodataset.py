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
customized dataset loader.
"""

import json
from pathlib import Path
from .loader import VideoLoader


def get_class_labels(dataset):
    class_labels_map = {}
    index = 0
    for class_label in dataset['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_database(dataset, subset, root_path, video_path_formatter):
    """
    Get video id, video path and annotations.
    """
    video_ids = []
    video_paths = []
    annotations = []

    for key, value in dataset['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            video_ids.append(key)
            annotations.append(value['annotations'])
            if 'video_path' in value:
                video_paths.append(Path(value['video_path']))
            else:
                label = value['annotations']['label']
                video_paths.append(video_path_formatter(root_path, label, key))

    return video_ids, video_paths, annotations


def make_dataset(root_path, annotation_path, subset, video_path_formatter):
    """
    Get video path, segments, frame_indices, video id and label.
    """
    with annotation_path.open('r') as f:
        data = json.load(f)
    video_ids, video_paths, annotations = get_database(
        data, subset, root_path, video_path_formatter)

    class_to_idx = get_class_labels(data)
    idx_to_class = {}

    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    n_videos = len(video_ids)
    dataset = []
    for i in range(n_videos):
        if i % (n_videos // 5) == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_ids)))

        if 'label' in annotations[i]:
            label = annotations[i]['label']
            label_id = class_to_idx[label]
        else:
            label = 'test'
            label_id = -1

        video_path = video_paths[i]
        if not video_path.exists():
            continue

        segment = annotations[i]['segment']
        if segment[1] == 1:
            continue

        frame_indices = list(range(segment[0], segment[1]))
        sample = {
            'video': video_path,
            'segment': segment,
            'frame_indices': frame_indices,
            'video_id': video_ids[i],
            'label': label_id
        }
        dataset.append(sample)

    return dataset, idx_to_class


class DatasetGenerator:
    """
    Custom dataset generator.
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 video_loader=None,
                 video_path_formatter=(
                     lambda root_path, label, video_id: root_path / label / video_id),
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 temporal_transform=None,
                 target_type='label'
                 ):
        super(DatasetGenerator, self).__init__()
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, video_path_formatter)

        self.image_name_formatter = image_name_formatter

        self.target_type = target_type

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader
        self.temporal_transform = temporal_transform

    def __getitem__(self, index):
        path = self.data[index]['video']
        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clip = self.loader(path, frame_indices)
        return clip, target

    def __len__(self):
        return len(self.data)
