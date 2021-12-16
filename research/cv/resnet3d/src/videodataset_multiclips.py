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

import copy
import json
from pathlib import Path
from .videodataset import make_dataset
from .loader import VideoLoader


def get_target_path(annotation_path):
    annotation_path = str(annotation_path)
    dst_path = annotation_path[:annotation_path.rfind('/')]
    dst_path += '/targets.json'
    dst_path = Path(dst_path)
    return dst_path


class DatasetGeneratorMultiClips:
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
                 target_type=None
                 ):
        super(DatasetGeneratorMultiClips, self).__init__()
        if target_type is None:
            target_type = ['video_id', 'segment']
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, video_path_formatter)

        dst_path = get_target_path(annotation_path)
        self.target_type = target_type
        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader
        self.temporal_transform = temporal_transform

        self.data_to_dump = {}
        self.data_to_dump['class_names'] = self.class_names
        self.data_to_dump['targets'] = {}
        for idx, data in enumerate(self.data):
            video_frame_indices = data['frame_indices']
            if self.temporal_transform is not None:
                video_frame_indices = self.temporal_transform(
                    video_frame_indices)
            segments = []
            for clip_frame_indices in video_frame_indices:
                segments.append(
                    [min(clip_frame_indices), max(clip_frame_indices) + 1])

            if isinstance(self.target_type, list):
                target = [data[t] for t in self.target_type]
            else:
                target = data[self.target_type]

            if 'segment' in self.target_type:
                if isinstance(self.target_type, list):
                    segment_index = self.target_type.index('segment')
                    targets = []
                    for s in segments:
                        targets.append(copy.deepcopy(target))
                        targets[-1][segment_index] = s
                else:
                    targets = segments
            else:
                targets = [target for _ in range(len(segments))]
            self.data_to_dump['targets'][idx] = targets
        with dst_path.open('w') as f:
            json.dump(self.data_to_dump, f)
        self.image_name_formatter = image_name_formatter

    def __getitem__(self, index):
        path = self.data[index]['video']
        video_frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            video_frame_indices = self.temporal_transform(video_frame_indices)
        clips = []
        for clip_frame_indices in video_frame_indices:
            clip = self.loader(path, clip_frame_indices)
            clips.append(clip)

        return clips, index

    def __len__(self):
        return len(self.data)
