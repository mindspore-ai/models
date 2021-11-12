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
"""Train NetG and NetF/NetLoss"""
from mindspore import save_checkpoint
from mindspore.train.callback import Callback
from mindspore import ops
from mindspore import Tensor
from mindspore import dtype as mstype


class SaveCallbackNETG(Callback):
    """
    SavedCall for NetG to print loss and save checkpoint

    Args:
        net(NetG): The instantiation of NetG
        path(str): The path to save the checkpoint of NetG
    """
    def __init__(self, net, path):
        super(SaveCallbackNETG, self).__init__()
        self.loss = 1e5
        self.net = net
        self.path = path
        self.print = ops.Print()

    def step_end(self, run_context):
        """print info and save checkpoint per 100 steps"""
        cb_params = run_context.original_args()
        if bool(cb_params.net_outputs < self.loss) and cb_params.cur_epoch_num % 100 == 0:
            self.loss = cb_params.net_outputs
            save_checkpoint(self.net, self.path)
        if cb_params.cur_epoch_num % 100 == 0:
            self.print(
                f"NETG epoch : {cb_params.cur_epoch_num}, loss : {cb_params.net_outputs}")


class SaveCallbackNETLoss(Callback):
    """
    SavedCall for NetG to print loss and save checkpoint

    Args:
        net(NetG): The instantiation of NetF
        path(str): The path to save the checkpoint of NetF
        x(np.array): valid dataset
        ua(np.array): Label of valid dataset
    """
    def __init__(self, net, path, x, l, g, ua):
        super(SaveCallbackNETLoss, self).__init__()
        self.loss = 1e5
        self.error = 1e5
        self.net = net
        self.path = path
        self.l = l
        self.x = x
        self.g = g
        self.ua = ua
        self.print = ops.Print()

    def step_end(self, run_context):
        """print info and save checkpoint per 100 steps"""
        cb_params = run_context.original_args()
        u = (Tensor(self.g, mstype.float32) + Tensor(self.l, mstype.float32)
             * self.net(Tensor(self.x, mstype.float32))).asnumpy()
        self.tmp_error = (((u - self.ua)**2).sum()/self.ua.sum())**0.5
        if self.error > self.tmp_error and cb_params.cur_epoch_num % 100 == 0:
            self.error = self.tmp_error
            save_checkpoint(self.net, self.path)
        self.loss = cb_params.net_outputs
        if cb_params.cur_epoch_num % 100 == 0:
            self.print(
                f"NETF epoch : {cb_params.cur_epoch_num}, loss : {self.loss}, error : {self.tmp_error}")
