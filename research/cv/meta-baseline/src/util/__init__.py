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
util
"""
import os
import shutil
import time
import numpy as np
from mindspore import Tensor
from mindspore import dtype as ms
from mindspore import ops

_log_path = None


def set_log_path(path):
    """
    :param path:
    :return:
    """
    global _log_path
    _log_path = path


class Averager:
    """
    Averager
    """

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        """
        :param v:
        :param n:
        :return:
        """
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        """
        :return: v
        """
        return self.v


class Timer:
    """
    Timer
    """

    def __init__(self):
        self.v = time.time()

    def s(self):
        """
        :return:None
        """
        self.v = time.time()

    def t(self):
        """
        :return: time
        """
        return time.time() - self.v


def set_gpu(gpu):
    """
    :param gpu: gpu
    :return: None
    """
    print('set gpu:', gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def ensure_path(path):
    """
    ensure_path
    :param path:
    :param remove:
    :return:
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)


def time_str(t):
    """
    time to str
    :param t:
    :return: time
    """
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t >= 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


def make_nk_label(n, k, batch_size):
    """
    get label
    :param n:
    :param k:
    :param batch_size:
    :return: label
    """
    label = ops.BroadcastTo((k, n))(Tensor(np.arange(n), ms.float32))
    label = ops.Transpose()(label, (1, 0))
    label = ops.Reshape()(label, (-1,))
    label = ops.Tile()(label, (batch_size, 1))
    return label
