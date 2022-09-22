# Copyright 2020 Huawei Technologies Co., Ltd
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
"""process output"""

import json
from collections import UserDict
import numpy as np
from mindspore import Tensor
from src.model_utils.config import config


def to_py_obj(obj):
    """
    Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a python list.
    """
    if isinstance(obj, (dict, UserDict)):
        return {k: to_py_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py_obj(o) for o in obj]
    if isinstance(obj, Tensor):
        return obj.asnumpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def process_output(token_ids, tokenizer, skip_special_tokens=False, white_space=False):
    """ process output """
    sentences = []
    for batch_out in token_ids:
        tokens = []
        for i in range(config.batch_size):
            if batch_out.ndim == 3:
                batch_out = batch_out[:, 0]
            tokens = tokenizer.convert_ids_to_tokens(
                batch_out[i], skip_special_tokens)
            if white_space:
                sen = " ".join(tokens)
            else:
                sen = "".join(tokens)
            sen = sen.replace("Ġ", " ")
            if not sen:
                sen = " "
                print('sen is null')
            sentences.append(sen)
    return sentences


def get_target(file, num_samples=-1):
    """ get target """
    f = open(file, 'r', encoding='utf-8')
    origin_data = json.load(f)
    num = 0
    data = []
    if num_samples == -1:
        num_samples = len(origin_data)
    for i in range(num_samples):
        num += 1
        summary = origin_data[i]['summary']
        data.append(summary)
    return data
