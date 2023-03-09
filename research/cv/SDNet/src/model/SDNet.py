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
"""SDNet"""
import os
import time
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.common import initializer as init
import mindspore.ops as ops
from mindspore.train.callback import Callback
from mindspore import Tensor
from mindspore import save_checkpoint
from mindspore import ParameterTuple
import mindspore.ops.functional as F
from src.Utils import L2_norm, orthogonal
from src.EvalMetrics import inference


class Focalnet(nn.Cell):

    def __init__(self):
        super(Focalnet, self).__init__()
        self.features_opt = nn.SequentialCell(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      pad_mode='pad',
                      padding=1,
                      has_bias=False),
            nn.BatchNorm2d(num_features=32, affine=False),
            nn.ReLU(),

            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      pad_mode='pad',
                      padding=1,
                      has_bias=False),
            nn.BatchNorm2d(num_features=32, affine=False),
        )
        self.features_sar = nn.SequentialCell(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      pad_mode='pad',
                      padding=1,
                      has_bias=False),
            nn.BatchNorm2d(num_features=32, affine=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      pad_mode='pad',
                      padding=1,
                      has_bias=False),
            nn.BatchNorm2d(num_features=32, affine=False),
        )

        self.embedding = nn.SequentialCell(
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=2,
                      pad_mode='pad',
                      padding=1,
                      has_bias=False),
            nn.BatchNorm2d(num_features=64, affine=False),
            nn.ReLU(),

            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      pad_mode='pad',
                      padding=1,
                      has_bias=False),
            nn.BatchNorm2d(num_features=64, affine=False),
            nn.ReLU(),

            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=2,
                      pad_mode='pad',
                      padding=1,
                      has_bias=False),
            nn.BatchNorm2d(num_features=128, affine=False),
            nn.ReLU(),

            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      pad_mode='pad',
                      padding=1,
                      has_bias=False),
            nn.BatchNorm2d(num_features=128, affine=False),
            nn.ReLU(),

            nn.Dropout(p=0.3),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=8,
                      pad_mode='pad',
                      has_bias=False),
            nn.BatchNorm2d(num_features=128, affine=False),
        )
        self.L2_norm = L2_norm()
        self.expand_dims = ops.ExpandDims()
        self.mean = ops.ReduceMean()

    def input_norm(self, x):
        flat = x.view(x.shape[0], -1)
        mp = self.mean(flat, 1)
        sp = flat.std(axis=1) + 1e-7
        return (x - self.expand_dims(self.expand_dims(self.expand_dims(mp, -1), -1), -1).expand_as(
            x)) / self.expand_dims(self.expand_dims(self.expand_dims(sp, -1), -1), 1).expand_as(x)

    def construct(self, opt_img, sar_img):
        opt_feature = self.features_opt(self.input_norm(opt_img))
        sar_feature = self.features_sar(self.input_norm(sar_img))

        opt_embeddings = self.embedding(opt_feature)
        sar_embeddings = self.embedding(sar_feature)
        opt_feat = opt_embeddings.view((opt_embeddings.shape[0], -1))
        sar_feat = sar_embeddings.view((sar_embeddings.shape[0], -1))
        return self.L2_norm(opt_feat), opt_feature, self.L2_norm(sar_feat), sar_feature


class SDNet(nn.Cell):

    def __init__(self, model, model_L, model_D):
        super(SDNet, self).__init__()
        self.model = model
        self.model_L = model_L
        self.model_D = model_D
        # # TODO init weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Init the weight of Conv2d in the net.
        """
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(orthogonal(cell.weight.shape, 0.6))
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer(init.Constant(0.01), cell.bias.shape,
                                         cell.bias.dtype))

    def construct(self, opt_img, sar_img):
        opt_e, opt_feat, sar_e, sar_feat = self.model(opt_img, sar_img)
        opt_lower_nor, opt_lower, sar_lower_nor, sar_lower = self.model_L(opt_feat, sar_feat)
        opt_recon = self.model_D(opt_lower)
        sar_recon = self.model_D(sar_lower)

        return opt_e, sar_e, opt_lower_nor, sar_lower_nor, opt_recon, sar_recon


class EvalCallBack(Callback):
    """Precision verification using callback function."""

    # define the operator required
    def __init__(self, network, eval_dataset, ckpt_save_dir, acc_fn, fpr95):
        super(EvalCallBack, self).__init__()
        self.network = network
        self.eval_dataset = eval_dataset
        self.ckpt_save_dir = ckpt_save_dir
        self.acc_fn = acc_fn
        self.fpr95 = fpr95

    # define operator function in epoch end
    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        self.network.set_train(False)
        acc_fpr95, acc = inference(self.network, self.eval_dataset, self.fpr95, self.acc_fn)
        print('\33[91mAccuracy(FPR95): {:.8f}  Acc:{:.8f}\33[0m'.format(acc_fpr95, acc))
        save_checkpoint(self.network, os.path.join(self.ckpt_save_dir, "checkpoint_SDNet_" + str(cur_epoch) + ".ckpt"))
        if ms.get_context("mode"):
            self.network.set_train(True)


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
    def __init__(self, backbone, loss_fn):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, opt_img, sar_img, label):
        opt_e, sar_e, opt_lower_nor, sar_lower_nor, opt_recon, sar_recon = self._backbone(opt_img, sar_img)
        return self._loss_fn(opt_img, sar_img, opt_e, sar_e, opt_lower_nor, sar_lower_nor, opt_recon, sar_recon)


class CustomTrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer, optimizer_L, optimizer_D):
        super(CustomTrainOneStepCell, self).__init__()
        self.network = network
        self.optimizer = optimizer
        self.optimizer_L = optimizer_L
        self.optimizer_D = optimizer_D
        self.weights = ParameterTuple(self.optimizer.parameters)
        self.weights_L = ParameterTuple(self.optimizer_L.parameters)
        self.weights_D = ParameterTuple(self.optimizer_D.parameters)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *args):
        loss = self.network(*args)
        grads = self.grad(self.network, self.weights)(*args)
        grads_L = self.grad(self.network, self.weights_L)(*args)
        grads_D = self.grad(self.network, self.weights_D)(*args)
        loss = F.depend(loss, self.optimizer(grads))
        loss = F.depend(loss, self.optimizer_L(grads_L))
        return F.depend(loss, self.optimizer_D(grads_D))
