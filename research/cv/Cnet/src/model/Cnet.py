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
"""Cnet"""
import os
import time
import numpy as np
import mindspore.nn as nn
from mindspore.common import initializer as init
import mindspore.ops as ops
from mindspore.train.callback import Callback
from mindspore import Tensor
from mindspore import save_checkpoint
from src.Utils import L2_norm, orthogonal
from src.EvalMetrics import inference
from src.model.CBAM import CBAM


class Cnet(nn.Cell):

    def __init__(self):
        super(Cnet, self).__init__()
        self.features = nn.SequentialCell(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                pad_mode='pad',
                padding=1,
                has_bias=False,
            ),
            nn.BatchNorm2d(num_features=32, momentum=0.9, affine=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      pad_mode='pad',
                      padding=1,
                      has_bias=False),
            nn.BatchNorm2d(num_features=32, momentum=0.9, affine=False),
            CBAM(32, reduction_ratio=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=2,
                      pad_mode='pad',
                      padding=1,
                      has_bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9, affine=False),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                pad_mode='pad',
                padding=1,
                has_bias=False,
            ),
            nn.BatchNorm2d(num_features=64, momentum=0.9, affine=False),
            CBAM(64, reduction_ratio=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=2,
                      pad_mode='pad',
                      padding=1,
                      has_bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9, affine=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      pad_mode='pad',
                      padding=1,
                      has_bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9, affine=False),
            CBAM(128, reduction_ratio=4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            # Global Pooling
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=8,
                      pad_mode='pad',
                      has_bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9, affine=False),
        )
        self.L2_norm = L2_norm()
        # # TODO init weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Init the weight of Conv2d and Dense in the net.
        """
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(orthogonal(cell.weight.shape, 0.6))
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer(init.Constant(0.01), cell.bias.shape,
                                         cell.bias.dtype))

    def input_norm(self, x):
        expand_dims = ops.ExpandDims()
        flat = x.view(x.shape[0], -1)
        mp = ops.ReduceMean()(flat, 1)
        sp = flat.std(axis=1) + 1e-7
        return (x - expand_dims(expand_dims(expand_dims(mp, -1), -1), -1).expand_as(x)) / expand_dims(
            expand_dims(expand_dims(sp, -1), -1), 1).expand_as(x)

    def construct(self, patch):
        x_features = self.features(self.input_norm(patch))
        x = x_features.view(x_features.shape[0], -1)
        return self.L2_norm(x)


class EvalCallBack(Callback):
    """Precision verification using callback function."""

    # define the operator required
    def __init__(self, network, eval_dataset, acc_fn, ckpt_save_dir):
        super(EvalCallBack, self).__init__()
        self.network = network
        self.eval_dataset = eval_dataset
        self.acc_fn = acc_fn
        self.ckpt_save_dir = ckpt_save_dir
        self.sqrt = ops.Sqrt()
        self.sum = ops.ReduceSum()
        self.reshape = ops.Reshape()

    # define operator function in epoch end
    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        self.network.set_train(False)
        for dataset in self.eval_dataset:
            acc = inference(self.network, dataset['dataloader'], self.acc_fn)
            print("epoch:{}, {}, Accuracy(FPR95): {:.8f}".format(cur_epoch,
                                                                 dataset['name'], acc))
        save_checkpoint(self.network, os.path.join(self.ckpt_save_dir, "checkpoint_Cnet_" + str(cur_epoch) + ".ckpt"))


class TimeLossMonitor(Callback):

    def __init__(self, lr_base=None):
        super(TimeLossMonitor, self).__init__()
        self.lr_base = lr_base
        self.total_loss = []
        self.epoch_time = 0
        self.step_time = 0

    def epoch_begin(self, run_context):
        """Epoch begin."""
        self.total_loss = []
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        """Epoch end."""
        cb_params = run_context.original_args()

        epoch_seconds = (time.time() - self.epoch_time) * 1000
        per_step_seconds = epoch_seconds / cb_params.batch_num
        print("epoch: [{:3d}/{:3d}], epoch time: {:5.3f}, steps: {:5d}, per step time: {:5.3f}, avg loss: {:5.3f}, "
              "lr:[{:8.6f}]".format(cb_params.cur_epoch_num, cb_params.epoch_num, epoch_seconds,
                                    cb_params.batch_num, per_step_seconds, np.mean(self.total_loss), self.lr_base[
                                        cb_params.cur_step_num - 1]), flush=True)

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        onestep_loss = cb_params.net_outputs

        if isinstance(onestep_loss, (tuple, list)) and isinstance(onestep_loss[0], Tensor):
            onestep_loss = onestep_loss[0]
        if isinstance(onestep_loss, Tensor):
            onestep_loss = np.mean(onestep_loss.asnumpy())

        self.total_loss.append(onestep_loss)


class CustomWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn, load_triplets):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self._load_triplets = load_triplets

    def construct(self, data_a, data_p, data_n):
        out_n = None
        out_a = self._backbone(data_a)
        out_p = self._backbone(data_p)
        if self._load_triplets:
            out_n = self._backbone(data_n)
        return self._loss_fn(out_a, out_p, out_n)
