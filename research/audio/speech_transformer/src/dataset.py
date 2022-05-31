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
"""Data operations, will be used in train.py."""

import json
import pickle
from math import ceil
from pathlib import Path

import mindspore.common.dtype as mstype
import mindspore.dataset as de
import mindspore.dataset.transforms as deC
import numpy as np

from .model_utils.config import config

de.config.set_seed(1)


class MsAudioDataset:
    """
    Audio dataset for AISHELL dataset.

    Args:
        data_json_path (str|Path): Path to dataset json.
        chars_dict_path (str|Path): Path to dataset character dictionary.
        lfr_m (int): preprocessing param, number of frames to stack. Default: 4.
        lfr_n (int): preprocessing param, number of frames to skip. Default: 3.
    """
    IGNORE_ID = -1

    def __init__(self, data_json_path, chars_dict_path, lfr_m=4, lfr_n=3):
        self.data_json_path = Path(data_json_path)
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.chars_dict_path = Path(chars_dict_path)
        self.char_list, self.sos_id, self.eos_id = self.process_dict(self.chars_dict_path)
        with self.data_json_path.open('r') as file:
            self.data = json.load(file)
        self.max_input_len, self.max_output_len = self.get_max_lens()
        self.data_samples = list(self.data.values())

    @staticmethod
    def read_pickle(file_path):
        """read pickle data"""
        with Path(file_path).open('rb') as file:
            data = pickle.load(file)

        return data

    @staticmethod
    def process_dict(dict_path):
        """process character dict"""
        with Path(dict_path).open('r') as files:
            dictionary = files.readlines()
        char_list = [entry.split(' ')[0] for entry in dictionary]
        sos_id = char_list.index('<sos>')
        eos_id = char_list.index('<eos>')
        return char_list, sos_id, eos_id

    def __getitem__(self, item):
        """get preprocessed data"""
        sample = self.data_samples[item]
        output_tokens = [int(token) for token in sample['output'][0]['tokenid'].split(' ')]

        decoder_input_tokens = np.asarray([self.sos_id] + output_tokens)
        padded_decoder_input_tokens = np.full((self.max_output_len,), self.eos_id, dtype=np.int64)
        padded_decoder_input_tokens[:decoder_input_tokens.shape[0]] = decoder_input_tokens
        padded_decoder_input_mask = (padded_decoder_input_tokens != self.eos_id).astype(np.float32)

        decoder_output_tokens = np.asarray(output_tokens + [self.eos_id])
        padded_decoder_output_tokens = np.full((self.max_output_len,), self.IGNORE_ID, dtype=np.int64)
        padded_decoder_output_tokens[:decoder_output_tokens.shape[0]] = decoder_output_tokens
        padded_decoder_output_mask = (padded_decoder_output_tokens != self.IGNORE_ID).astype(np.float32)

        input_features = self.read_pickle(sample['input'][0]['feat'])
        input_features = self.build_lfr_features(input_features, m=self.lfr_m, n=self.lfr_n)
        padded_input_features = np.full((self.max_input_len, input_features.shape[1]), 0, dtype=np.float32)
        padded_input_features[:input_features.shape[0]] = input_features
        padded_input_features_mask = np.full((self.max_input_len,), 0, dtype=np.int32)
        padded_input_features_mask[:input_features.shape[0]] = 1

        result = (
            padded_input_features, padded_input_features_mask,
            padded_decoder_input_tokens, padded_decoder_input_mask,
            padded_decoder_output_tokens, padded_decoder_output_mask,
        )

        return result

    def __len__(self):
        """num of samples"""
        return len(self.data_samples)

    @staticmethod
    def build_lfr_features(inputs, m, n):
        """
        Actually, this implements stacking frames and skipping frames.
        if m = 1 and n = 1, just return the origin features.
        if m = 1 and n > 1, it works like skipping.
        if m > 1 and n = 1, it works like stacking but only support right frames.
        if m > 1 and n > 1, it works like LFR.

        Args:
            inputs_batch: inputs is T x D np.ndarray
            m: number of frames to stack
            n: number of frames to skip
        """
        LFR_inputs = []
        T = inputs.shape[0]
        T_lfr = int(np.ceil(T / n))
        for i in range(T_lfr):
            if m <= T - i * n:
                LFR_inputs.append(np.hstack(inputs[i * n:i * n + m]))
            else:  # process last LFR frame
                num_padding = m - (T - i * n)
                frame = np.hstack(inputs[i * n:])
                for _ in range(num_padding):
                    frame = np.hstack((frame, inputs[-1]))
                LFR_inputs.append(frame)
        return np.vstack(LFR_inputs)

    def get_max_lens(self):
        """get maximum sequence len"""
        input_max_len = 0
        output_max_len = 0
        for sample in self.data.values():
            input_cur_len = sample['input'][0]['shape'][0]
            output_cur_len = sample['output'][0]['shape'][0]
            input_max_len = input_cur_len if input_cur_len > input_max_len else input_max_len
            output_max_len = output_cur_len if output_cur_len > output_max_len else output_max_len

        return ceil(input_max_len / self.lfr_n) + 1, output_max_len + 1


def create_transformer_dataset(
        data_json_path,
        chars_dict_path,
        lfr_m=4,
        lfr_n=3,
        do_shuffle='true',
        rank_size=1,
        rank_id=0,
        epoch_count=1,
        batch_size=None,
):
    """
    Create Audio dataset for AISHELL dataset.

    Args:
        data_json_path (str|Path): Path to dataset json.
        chars_dict_path (str|Path): Path to dataset character dictionary.
        lfr_m (int): preprocessing param, number of frames to stack. Default: 4.
        lfr_n (int): preprocessing param, number of frames to skip. Default: 3.
        do_shuffle (str): if true shuffle dataset. Default: 'true'.
        rank_size (int): distributed training rank size. Default: 1.
        rank_id (int): distributed training rank id. Default: 0.
        epoch_count (int): number of dataset repeats. Default: 1.
        batch_size (int): dataset batch size, if none get batch size info from config. Default: None.
    """
    dataset = MsAudioDataset(
        data_json_path,
        chars_dict_path,
        lfr_m,
        lfr_n,
    )

    ds = de.GeneratorDataset(
        source=dataset,
        column_names=[
            'source_eos_features',
            'source_eos_mask',
            'target_sos_ids',
            'target_sos_mask',
            'target_eos_ids',
            'target_eos_mask',
        ],
        shuffle=(do_shuffle == "true"),
        num_shards=rank_size,
        shard_id=rank_id,
    )
    type_cast_op_int32 = deC.TypeCast(mstype.int32)
    type_cast_op_float32 = deC.TypeCast(mstype.float32)
    ds = ds.map(operations=type_cast_op_float32, input_columns="source_eos_features")
    ds = ds.map(operations=type_cast_op_int32, input_columns="source_eos_mask")
    ds = ds.map(operations=type_cast_op_int32, input_columns="target_sos_ids")
    ds = ds.map(operations=type_cast_op_int32, input_columns="target_sos_mask")
    ds = ds.map(operations=type_cast_op_int32, input_columns="target_eos_ids")
    ds = ds.map(operations=type_cast_op_int32, input_columns="target_eos_mask")
    batch_size = batch_size if batch_size is not None else config.batch_size
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat(epoch_count)
    return ds
