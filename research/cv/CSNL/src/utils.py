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
import os
import time
from mindspore import ops
from mindspore import nn
import mindspore
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.train.serialization import save_checkpoint
from mindspore.communication import get_rank


def mul(x1, x2):
    m = ops.Mul()
    return m(x1, x2)


def reshape(target, shape):
    rs = ops.Reshape()
    return rs(target, shape).astype(mindspore.float32)


def eye(w):
    e = ops.Eye()
    return e(w, w, mindspore.int32)


def size(tensor):
    s = ops.Shape()
    return s(tensor)


def matmul(x1, x2):
    mat = nn.MatMul()
    return mat(x1, x2)


def split(input_x, axis):
    sp = ops.Split(axis=axis)
    return sp(input_x)


def reduce_sum(x, axis=None, keepdim=False):
    if axis is None:
        axis = range(len(x.shape))
    op = ops.ReduceSum(keepdim)
    for i in axis:
        x = op(x, i)
    return x


class Trainer():
    """Trainer"""

    def __init__(self, args, loader, my_model):
        self.args = args
        self.scale = args.scale
        self.trainloader = loader
        self.model = my_model
        self.model.set_train()
        self.criterion = nn.L1Loss()
        self.optimizer = nn.Adam(self.model.trainable_params(), learning_rate=args.lr_init, loss_scale=1024.0)
        self.loss_net = nn.WithLossCell(self.model, self.criterion)
        self.net = nn.TrainOneStepCell(self.loss_net, self.optimizer)

    def train(self):
        """Trainer"""
        losses = 0
        batch_idx = 0
        for batch_idx, imgs in enumerate(self.trainloader):
            lr = imgs["LR"]
            hr = imgs["HR"]
            lr = Tensor(lr, mindspore.float32)  # [16,3,24,24]
            hr = Tensor(hr, mindspore.float32)  # [16,3,48,48]
            t1 = time.time()
            loss = self.net(lr, hr)
            t2 = time.time()
            losses += loss.asnumpy()
            print('Step: %g, losses: %f, time: %f ' % \
                  (batch_idx, loss.asnumpy(), t2 - t1), flush=True)
        print("the epoch losses is", losses / (batch_idx + 1), flush=True)
        os.makedirs(self.args.save, exist_ok=True)
        if self.args.distribute:
            if get_rank() == 0 and self.epoch % 5 == 0:
                save_checkpoint(self.net, self.args.save + "model_" + str(self.epoch) + '.ckpt')
        else:
            save_checkpoint(self.net, self.args.save + "model_" + str(self.epoch) + '.ckpt')

    def update_learning_rate(self, epoch):
        """Update learning rates for all the networks; called at the end of every epoch.
        :param epoch: current epoch
        :type epoch: int
        :param lr: learning rate of cyclegan
        :type lr: float
        :param niter: number of epochs with the initial learning rate
        :type niter: int
        :param niter_decay: number of epochs to linearly decay learning rate to zero
        :type niter_decay: int
        """
        self.epoch = epoch
        print("*********** epoch: {} **********".format(epoch))
        lr = self.args.lr_init / (2 ** ((epoch + 1) // 200))
        self.adjust_lr('model', self.optimizer, lr)
        print("*********************************")

    def adjust_lr(self, name, optimizer, lr):
        """Adjust learning rate for the corresponding model.
        :param name: name of model
        :type name: str
        :param optimizer: the optimizer of the corresponding model
        :type optimizer: torch.optim
        :param lr: learning rate to be adjusted
        :type lr: float
        """
        lr_param = optimizer.get_lr()
        lr_param.assign_value(Tensor(lr, mstype.float32))
        print('==> ' + name + ' learning rate: ', lr_param.asnumpy())
