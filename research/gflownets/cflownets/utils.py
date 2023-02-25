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
"""
utils
"""
from datetime import datetime
import pickle

from scipy.special import softmax
import numpy as np
import mindspore as ms


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def save_model(critic):
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    param_list = []
    for (key, value) in critic.parameters_dict().items():
        each_param = {}
        each_param["name"] = key
        if isinstance(value.data, ms.Tensor):
            param_data = value.data
        else:
            param_data = ms.Tensor(value.data)
        each_param["data"] = param_data
        param_list.append(each_param)
    ms.save_checkpoint(param_list, 'runs/gfn_MA_' + now_time + '.ckpt')


def select_action_base_probability(sample_action, prob_data):
    action = np.random.choice(sample_action.reshape(-1), p=prob_data)
    return np.array([action])


def softmax_matrix(data):
    """
    The element in the matrix is greater than 0; and sum equals 1
    """
    data = softmax(data, axis=0)
    data_sum = np.sum(data)
    data /= data_sum
    return data
