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
''' define dataset and sampler function '''
import time
import math
import numpy as np
import h5py
from src.hparams import hparams as hps


class Sampler():
    ''' sampler '''
    def __init__(self, sample_nums, rank, group_size, random_seed=0):
        self.batch_size = hps.batch_size
        self.rank = rank
        self.group_size = group_size
        self.seed = random_seed
        self.sample_nums = sample_nums

        self.sample_indexes = np.arange(self.sample_nums).tolist()
        self.total_sample_nums = int(
            math.ceil(self.sample_nums / self.batch_size)) * self.batch_size

        self.sample_indexes += self.sample_indexes[:(
            self.total_sample_nums - self.sample_nums)]
        print('total training samples: {}'.format(len(self.sample_indexes)))
        print('batch_size : {}'.format(self.batch_size))
        num_steps_per_epoch = int(
            math.ceil(
                self.sample_nums /
                self.batch_size))
        self.total_steps = int(
            math.ceil(
                num_steps_per_epoch / self.group_size)) * self.group_size
        self.step_indexes = np.arange(num_steps_per_epoch).tolist()
        self.step_indexes += self.step_indexes[:(
            self.total_steps - len(self.step_indexes))]

    def __iter__(self):
        self.seed = (self.seed + 1) & 0xffffffff
        np.random.seed(self.seed)
        np.random.shuffle(self.step_indexes)
        step_indexes = self.step_indexes
        np.random.seed(self.seed)
        np.random.shuffle(self.sample_indexes)
        sample_indexes = self.sample_indexes
        sample_indexes_bins = [sample_indexes[i:i + self.batch_size]
                               for i in range(0, len(sample_indexes), self.batch_size)]

        index = []
        for step_idx in step_indexes[self.rank::self.group_size]:
            index.extend(sample_indexes_bins[step_idx])
        return iter(index)


class ljdataset():
    ''' ljspeech-1.1 dataset'''
    def __init__(self, hdf5_pth, group_size):
        self.max_text_len = 0
        self.max_mel_len = 0
        load_time = time.time()
        self.dataset = []
        with h5py.File(hdf5_pth, 'r') as ff:
            self.dsname = sorted(ff.keys())
            self.length = len(ff.keys()) // 2
            for idx in range(0, self.length * 2, 2):
                self.dataset.append({
                    'mel': ff[self.dsname[idx]][:],
                    'text': ff[self.dsname[idx + 1]][:]
                })
                self.max_mel_len = max(
                    ff[self.dsname[idx]][:].shape[1], self.max_mel_len)
                self.max_text_len = max(
                    len(ff[self.dsname[idx + 1]][:]), self.max_text_len)
        self.sample_nums = len(self.dataset)

        hps.max_text_len = self.max_text_len

        print('Training number: {}'.format(self.sample_nums))
        print('Load target time: {:.3f} s'.format(time.time() - load_time))
        print('max text length : {}'.format(self.max_text_len))
        print('max mel length : {}'.format(self.max_mel_len))
        self.n_frames_per_step = hps.n_frames_per_step
        if self.max_mel_len % self.n_frames_per_step != 0:
            self.max_mel_len += self.n_frames_per_step - \
                self.max_mel_len % self.n_frames_per_step
            assert self.max_mel_len % self.n_frames_per_step == 0

        num_steps_per_epoch = int(math.ceil(self.sample_nums / hps.batch_size))
        self.group_size = group_size
        print('{} steps per epoch'.format(num_steps_per_epoch))

    def __getitem__(self, index):
        meta = self.dataset[index]
        text_padded, input_lengths, mel_padded, gate_padded, text_mask, mel_mask, rnn_mask = self.sort_and_pad(
            meta)

        return text_padded, input_lengths, mel_padded, gate_padded, text_mask, mel_mask, rnn_mask

    def __len__(self):
        return int(math.ceil(int(math.ceil(self.sample_nums / \
                   hps.batch_size)) / self.group_size)) * hps.batch_size

    def sort_and_pad(self, meta):
        ''' pad text sequence and mel spectrogram'''
        text = meta['text']
        mel = meta['mel']

        text_len = len(text)

        input_lengths = np.array(text_len, np.int32)

        n_mels, n_frames = mel.shape

        max_input_len = self.max_text_len

        text_padded = np.ones((max_input_len), np.int32)
        text_padded[:text_len] = text

        max_target_len = self.max_mel_len

        mel_padded = np.zeros((n_mels, max_target_len), np.float32)
        mel_padded[:, :n_frames] = mel
        gate_padded = np.zeros((max_target_len), np.float32)
        gate_padded[n_frames - 1:] = 1

        text_mask = np.zeros((max_input_len)).astype(np.bool)
        text_mask[:text_len] = True
        mel_mask = np.zeros((max_target_len)).astype(np.bool)
        mel_mask[:n_frames] = True
        mel_mask = np.expand_dims(mel_mask, 0).repeat(80, 0)

        rnn_mask = np.zeros((max_input_len)).astype(np.bool)
        rnn_mask[:text_len] = True
        rnn_mask = np.expand_dims(rnn_mask, 1).repeat(512, 1)

        return text_padded, input_lengths, mel_padded, gate_padded, text_mask, mel_mask, rnn_mask
