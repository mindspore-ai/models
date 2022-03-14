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
"""Utilities."""
from pathlib import Path

import numpy as np

from src.cfg.config import config as hp


def process_text(train_text_path):
    """
    Read .txt data.
    """
    metadata_path = Path(train_text_path)
    with metadata_path.open("r", encoding="utf-8") as file:
        txt = []
        for line in file.readlines():
            txt.append(line)

        return txt


def pad_1d_tensor(inputs):
    """
    Pad 1d tensor to fixed size.
    """
    max_len = hp.character_max_length
    padded = np.pad(inputs, (0, max_len - inputs.shape[0]))

    return padded


def pad_2d_tensor(inputs):
    """
    Pad 2d tensor to fixed size.
    """
    max_len = hp.mel_max_length
    s = inputs.shape[1]
    padded = np.pad(inputs, (0, max_len - inputs.shape[0]))[:, :s]

    return padded
