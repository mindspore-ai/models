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

import os
import json
import cv2
import numpy as np
from PIL import ImageFile
import mindspore.dataset as de

from src.model_utils.config import config
from src.transform import PreprocessTrainC3D, PreprocessEvalC3D

ImageFile.LOAD_TRUNCATED_IMAGES = True


class video_Dataset():
    def __init__(self):
        self.load_type = config.load_type
        self.crop_shape = config.crop_shape
        self.final_shape = config.final_shape
        self.preprocess = config.preprocess
        self.json_path = config.json_path
        self.img_path = config.img_path

        # Clips processing arguments
        self.clip_length = config.clip_length
        self.clip_offset = config.clip_offset
        self.clip_stride = config.clip_stride
        self.num_clips = config.num_clips
        self.random_offset = config.random_offset

        # Frame-wise processing arguments
        self.resize_shape = config.resize_shape
        self.crop_type = config.crop_type

        # Experiment arguments
        self.batch_size = config.batch_size
        self.class_number = config.num_classes

        self._index = 0

        if self.load_type == 'train':
            self.transforms = PreprocessTrainC3D()
        else:
            self.transforms = PreprocessEvalC3D()

        self._getClips()

    def __getitem__(self, idx):
        vid_info = self.samples[idx]
        base_path = os.path.join(self.img_path, vid_info['base_path'])

        vid_length = len(vid_info['frames'])
        labels = np.zeros((vid_length)) - 1
        input_data = []

        for frame_ind in range(len(vid_info['frames'])):
            frame_path = os.path.join(base_path, vid_info['frames'][frame_ind]['img_path'])

            for frame_labels in vid_info['frames'][frame_ind]['actions']:
                labels[frame_ind] = frame_labels['action_class']

            # Load frame image data and preprocess image accordingly
            input_data.append(cv2.imread(frame_path)[..., ::-1] / 1.)

        # Preprocess data
        vid_data = self.transforms(input_data)
        vid_data = np.transpose(vid_data, (3, 0, 1, 2))

        if self.load_type == 'train':
            one_hot = np.zeros((self.class_number))
            one_hot[labels.astype('int32')[-1]] = 1.
            labels = one_hot
        return (vid_data.astype(np.float32), labels.astype(np.int32))

    def __len__(self):
        return len(self.samples)

    def _getClips(self):
        self.samples = []

        if self.load_type == 'train':
            full_json_path = os.path.join(self.json_path, 'train.json')

        elif self.load_type == 'val':
            full_json_path = os.path.join(self.json_path, 'val.json')

            if not os.path.exists(full_json_path):
                full_json_path = os.path.join(self.json_path, 'test.json')

        else:
            full_json_path = os.path.join(self.json_path, 'test.json')

        json_file = open(full_json_path, 'r')
        json_data = json.load(json_file)
        json_file.close()

        # Load the information for each video and process it into clips
        for video_info in json_data:
            clips = self._extractClips(video_info['frames'])

            # Each clip is a list of dictionaries per frame containing information
            # Example info: object bbox annotations, object classes, frame img path
            for clip in clips:
                self.samples.append(dict(frames=clip, base_path=video_info['base_path']))

    def _extractClips(self, video):
        """
        Processes a single video into uniform sized clips that will be loaded by __getitem__
        Args:
            video:       List containing a dictionary of annontations per frame

        Additional Parameters:
            self.clip_length: Number of frames extracted from each clip
            self.num_clips:   Number of clips to extract from each video
            (-1 uses the entire video, 0 paritions the entire video in clip_length clips)
            self.clip_offset: Number of frames from beginning of video to start extracting clips
            self.clip_stride: Number of frames between clips when extracting them from videos
            self.random_offset: Randomly select a clip_length sized clip from a video
        """
        if self.clip_offset > 0:
            if len(video) - self.clip_offset >= self.clip_length:
                video = video[self.clip_offset:]

        if self.num_clips < 0:
            if len(video) >= self.clip_length:
                # Uniformly sample one clip from the video
                final_video = [video[_idx] for _idx in np.linspace(0, len(video) - 1, self.clip_length, dtype='int32')]
                final_video = [final_video]

            else:
                indices = np.ceil(self.clip_length / float(len(video)))
                indices = indices.astype('int32')
                indices = np.tile(np.arange(0, len(video), 1, dtype='int32'), indices)
                indices = indices[np.linspace(0, len(indices) - 1, self.clip_length, dtype='int32')]

                final_video = [video[_idx] for _idx in indices]
                final_video = [final_video]

        elif self.num_clips == 0:
            if len(video) >= self.clip_length:
                indices = np.arange(start=0, stop=len(video) - self.clip_length + 1, step=self.clip_stride)
                final_video = []

                for _idx in indices:
                    if _idx + self.clip_length <= len(video):
                        final_video.append([video[true_idx] for true_idx in range(_idx, _idx + self.clip_length)])
            else:
                indices = np.ceil(self.clip_length / float(len(video)))
                indices = indices.astype('int32')
                indices = np.tile(np.arange(0, len(video), 1, dtype='int32'), indices)
                indices = indices[:self.clip_length]

                final_video = [video[_idx] for _idx in indices]
                final_video = [final_video]

        else:
            if self.clip_length == -1:
                # This is a special case where we will return the entire video
                # Batch size must equal one or dataloader items may have varying lengths
                # and can't be stacked i.e. throws an error
                assert self.batch_size == 1
                return [video]

            required_length = (self.num_clips - 1) * (self.clip_stride) + self.clip_length

            if self.random_offset:
                if len(video) >= required_length:
                    vid_start = np.random.choice(np.arange(len(video) - required_length + 1), 1)
                    video = video[int(vid_start):]

            if len(video) >= required_length:
                # Get indices of sequential clips overlapped by a clip_stride number of frames
                indices = np.arange(0, len(video), self.clip_stride)

                # Select only the first num clips
                indices = indices.astype('int32')[:self.num_clips]

                video = np.array(video)
                final_video = [video[np.arange(_idx, _idx + self.clip_length).astype('int32')].tolist() for _idx in
                               indices]
            else:
                indices = np.ceil(required_length / float(len(video)))
                indices = indices.astype('int32')
                indices = np.tile(np.arange(0, len(video), 1, dtype='int32'), indices)

                # Starting index of each clip
                clip_starts = np.arange(0, len(indices), self.clip_stride).astype('int32')[:self.num_clips]

                video = np.array(video)
                final_video = [video[indices[_idx:_idx + self.clip_length]].tolist() for _idx in clip_starts]
        return final_video


def classification_dataset(per_batch_size, group_size, shuffle=True, repeat_num=1, drop_remainder=True):
    dataset = video_Dataset()
    dataset_len = len(dataset)

    num_parallel_workers = config.num_workers
    if group_size > 1:
        de_dataset = de.GeneratorDataset(dataset, ["video", "label"], num_parallel_workers=num_parallel_workers,
                                         shuffle=shuffle, num_shards=group_size, shard_id=config.rank)
    elif group_size == 1:
        de_dataset = de.GeneratorDataset(dataset, ["video", "label"], num_parallel_workers=num_parallel_workers,
                                         shuffle=shuffle)
    else:
        raise ValueError("Invalid group size: %d" % group_size)

    columns_to_project = ["video", "label"]
    de_dataset = de_dataset.project(columns=columns_to_project)

    de_dataset = de_dataset.batch(per_batch_size, drop_remainder=drop_remainder)
    de_dataset = de_dataset.repeat(repeat_num)

    return de_dataset, dataset_len
