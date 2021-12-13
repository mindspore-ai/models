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
"""utils.py"""
import os
import os.path as osp
import sys
import numpy as np
import mindspore as ms
import mindspore.dataset as ds

from mindspore import nn
from mindspore import Tensor


def get_param_list(save_obj):
    """
    Params:
        save_obj: mindspore.nn.module object
    Returns:
        A list of parameters of save_obj
    """
    if isinstance(save_obj, nn.Cell):
        param_dict = {}
        for _, param in save_obj.parameters_and_names():
            param_dict[param.name] = param
        param_list = []
        for (key, value) in param_dict.items():
            each_param = {"name": key}
            param_data = Tensor(value.data)
            each_param["data"] = param_data
            param_list.append(each_param)
        return param_list

    return -1


def genidx(train_color_label, train_thermal_label):
    """
    Generate
    """
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(
            train_color_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(
            train_thermal_label) if v == unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)

    return color_pos, thermal_pos


class IdentitySampler(ds.Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize):
        super(IdentitySampler, self).__init__()
        # np.random.seed(0)
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)
        N = np.maximum(len(train_color_label), len(train_thermal_label))
        for j in range(int(N/(batchSize * num_pos))+1):
            batch_idx = np.random.choice(uni_label, batchSize, replace=False)
            for i in range(batchSize):
                sample_color = np.random.choice(
                    color_pos[batch_idx[i]], num_pos)
                sample_thermal = np.random.choice(
                    thermal_pos[batch_idx[i]], num_pos)

                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))

        self.index1 = index1
        self.index2 = index2
        self.N = N
        self.num_samples = N

    def __iter__(self):
        # return iter(np.arange(len(self.index1)))
        for i in range(len(self.index1)):
            yield i

    def __len__(self):
        return self.N


class AverageMeter():
    """Computers and stores the average & current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Logger():
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w', encoding='utf-8')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class LRScheduler():
    r"""
    Gets learning rate warming up + decay.

    Args:
        learning_rate (float): The initial value of learning rate.
        warmup_steps (int): The warm up steps of learning rate.
        weight_decay (int): The weight decay steps of learning rate.

    Inputs:
        Tensor. The current step number.

    Outputs:
        Tensor. The learning rate value for the current step.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    def __init__(self, learning_rate, steps_per_epoch, args):
        super(LRScheduler, self).__init__()
        self.learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.epochs = args.epoch
        self.warmup_steps = args.warmup_steps
        self.start_decay = args.start_decay
        self.end_decay = args.end_decay

        assert (self.epochs > self.warmup_steps) and (self.start_decay > self.warmup_steps)\
            and (self.end_decay > self.start_decay)

    def getlr(self):
        """
        return lr tensor sized self.epochs x steps_per_epoch
        """
        lr = []
        final_lr = 0.0
        for epoch in range(1, self.epochs + 1):
            # calculate the lr for current epoch
            if epoch < self.warmup_steps:
                warmup_percent = epoch / self.warmup_steps
                current_lr = self.learning_rate * warmup_percent
            elif epoch < self.start_decay:
                current_lr = self.learning_rate
            elif epoch <= self.end_decay:
                current_lr = self.learning_rate
                for decay in range(self.start_decay, self.end_decay + 1):
                    if decay <= epoch:
                        current_lr = current_lr * 0.1
                    else:
                        break
                if epoch == self.end_decay:
                    final_lr = current_lr
            else:
                current_lr = final_lr

            # fill the steps in current epoch

            for _ in range(self.steps_per_epoch):
                lr.append(current_lr)
        lr = Tensor(lr, dtype=ms.float32)

        return lr
