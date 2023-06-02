# Copyright 2023 Huawei Technologies Co., Ltd
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
from glob import glob

import librosa
import numpy as np
from tqdm import tqdm


class DatasetGenerator:
    def __init__(self, path, sr=16000, hop_length=512, groups=None, sequence_length=None, seed=42,
                 max_midi=108, min_midi=21):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.random = np.random.RandomState(seed)
        self.sample_rate = sr
        self.hop_length = hop_length
        self.max_midi = max_midi
        self.min_midi = min_midi

        self.data = []
        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                self.data.append(self.load(*input_files))

    def __getitem__(self, index):
        data = self.data[index]
        result = dict(path=data['path'])

        if self.sequence_length is not None:
            audio_length = len(data['audio'])
            step_begin = self.random.randint(audio_length - self.sequence_length) // self.hop_length
            n_steps = self.sequence_length // self.hop_length
            step_end = step_begin + n_steps

            begin = step_begin * self.hop_length
            end = begin + self.sequence_length

            result['audio'] = data['audio'][begin:end]
            result['onset'] = data['onset'][step_begin:step_end, :]
            result['offset'] = data['offset'][step_begin:step_end, :]
            result['frame'] = data['frame'][step_begin:step_end, :]
        else:
            result['audio'] = data['audio']
            result['onset'] = data['onset']
            result['offset'] = data['offset']
            result['frame'] = data['frame']
        try:
            res = [result['audio'], result['onset'], result['offset'], result['frame']]
        except KeyError:
            res = []
        return res

    def __len__(self):
        return len(self.data)

    @staticmethod
    def available_groups():
        return ['test']

    def files(self, group):
        audio_files = glob(os.path.join(self.path, group, '*.wav'))
        label_files = [f.replace('.wav', '.tsv') for f in audio_files]

        assert all(os.path.isfile(audio_file) for audio_file in audio_files)
        assert all(os.path.isfile(label_file) for label_file in label_files)

        return sorted(zip(audio_files, label_files))

    def load(self, audio_path, tsv_path):
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        audio_length = len(audio)

        n_keys = self.max_midi - self.min_midi + 1
        n_steps = (audio_length - 1) // self.hop_length + 1

        onsets = np.zeros((n_steps, n_keys), dtype=np.float32)
        offsets = np.zeros((n_steps, n_keys), dtype=np.float32)
        frames = np.zeros((n_steps, n_keys), dtype=np.float32)

        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

        for onset, offset, note in midi:
            left = int(round(onset * self.sample_rate / self.hop_length))

            onset_right = min(n_steps, left + 1)
            frame_right = int(round(offset * self.sample_rate / self.hop_length))

            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + 1)

            f = int(note) - self.min_midi
            offsets[frame_right:offset_right, f] = 1
            frames[left:frame_right, f] = 1
            onsets[left:onset_right, f] = 1

        data = dict(path=audio_path, audio=audio, onset=onsets, offset=offsets, frame=frames)
        return data
