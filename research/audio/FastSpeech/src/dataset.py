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
"""Data preprocessing."""
import os
from pathlib import Path

import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype

from src.cfg.config import config as hp
from src.text import text_to_sequence
from src.utils import pad_1d_tensor
from src.utils import pad_2d_tensor
from src.utils import process_text


def get_data_to_buffer():
    """
    Put data to memory, for faster training.
    """
    with Path(hp.dataset_path, 'train_indices.txt').open('r') as file:
        train_part = np.array([i[:-1] for i in file.readlines()], np.int32)
        train_part.sort()

    buffer = list()
    raw_text = process_text(os.path.join(hp.dataset_path, "metadata.txt"))

    for i in train_part:
        mel_gt_name = os.path.join(hp.dataset_path, 'mels', "ljspeech-mel-%05d.npy" % (i+1))
        mel_gt_target = np.load(mel_gt_name)

        duration = np.load(os.path.join(hp.dataset_path, 'alignments', str(i)+".npy"))

        character = raw_text[i][: len(raw_text[i])-1]
        character = np.array(text_to_sequence(character, hp.text_cleaners))

        buffer.append(
            {
                "text": character,
                "duration": duration,
                "mel_target": mel_gt_target
            }
        )

    return buffer


def reprocess_tensor(data_dict):
    """
    Prepare data for training.
    Apply padding for all samples, in reason of static graph.

    Args:
        data_dict (dict): Dictionary of np.array type data.

    Returns:
        out (dict): Dictionary with prepared data for training, np.array type.
    """
    text = data_dict["text"]
    mel_target = data_dict["mel_target"]
    duration = data_dict["duration"]

    max_len = hp.character_max_length
    length_text = text.shape[0]
    src_pos = np.pad([i+1 for i in range(int(length_text))], (0, max_len-int(length_text)), 'constant')

    max_mel_len = hp.mel_max_length
    length_mel = mel_target.shape[0]
    mel_pos = np.pad([i+1 for i in range(int(length_mel))], (0, max_mel_len-int(length_mel)), 'constant')

    text = pad_1d_tensor(text)
    duration = pad_1d_tensor(duration)
    mel_target = pad_2d_tensor(mel_target)

    out = {
        "text": text,  # shape (hp.character_max_length)
        "src_pos": src_pos,  # shape (hp.character_max_length)
        "mel_pos": mel_pos,  # shape (hp.mel_max_length)
        "duration": duration,  # shape (hp.character_max_length)
        "mel_target": mel_target,  # shape (hp.mel_max_length, hp.num_mels)
        "mel_max_len": max_mel_len,
    }

    return out


def preprocess_data(buffer):
    """
    Prepare data for training.

    Args:
        buffer (list): Raw data inputs.

    Returns:
        preprocessed_data (list): Padded and converted data, ready for training.
    """
    preprocessed_data = []
    for squeeze_data in buffer:
        db = reprocess_tensor(squeeze_data)

        preprocessed_data.append(
            (
                db["text"].astype(np.float32),
                db["src_pos"].astype(np.float32),
                db["mel_pos"].astype(np.float32),
                db["duration"].astype(np.int32),
                db["mel_target"].astype(np.float32),
                db["mel_max_len"],
            )
        )

    return preprocessed_data


class BufferDataset:
    """
    Dataloader.
    """
    def __init__(self, buffer):
        self.length_dataset = len(buffer)
        self.preprocessed_data = preprocess_data(buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.preprocessed_data[idx]


def get_val_data(data_url):
    """Get validation data."""
    data_list = list()
    with Path(data_url, 'validation.txt').open('r') as file:
        data_paths = file.readlines()

    root_wav_path = os.path.join(data_url, 'wavs')
    wav_paths = [root_wav_path + '/' + raw_path.split('|')[0] + '.wav' for raw_path in data_paths]
    val_txts = [raw_path.split('|')[1][:-1] for raw_path in data_paths]

    for orig_text, wav_path in zip(val_txts, wav_paths):
        sequence = text_to_sequence(orig_text, hp.text_cleaners)
        sequence = np.expand_dims(sequence, 0)

        src_pos = np.array([i + 1 for i in range(sequence.shape[1])])
        src_pos = np.expand_dims(src_pos, 0)

        sequence = Tensor([np.pad(sequence[0], (0, hp.character_max_length - sequence.shape[1]))], mstype.float32)
        src_pos = Tensor([np.pad(src_pos[0], (0, hp.character_max_length - src_pos.shape[1]))], mstype.float32)

        data_list.append([sequence, src_pos, wav_path])

    return data_list
