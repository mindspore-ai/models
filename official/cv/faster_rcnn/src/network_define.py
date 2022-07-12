 # Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""FasterRcnn training network wrapper."""

import time
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.train.callback import Callback

time_stamp_init = False
time_stamp_first = 0


class LossCallBack(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF terminating training.

    Note:
        If per_print_times is 0 do not print loss.

    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1, rank_id=0, lr=None):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.count = 0
        self.loss_sum = 0
        self.rank_id = rank_id
        self.lr = lr

        global time_stamp_init, time_stamp_first
        if not time_stamp_init:
            time_stamp_first = time.time()
            time_stamp_init = True

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs.asnumpy()
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        self.count += 1
        self.loss_sum += float(loss)

        if self.count >= 1:
            global time_stamp_first
            time_stamp_current = time.time()
            total_loss = self.loss_sum / self.count

            loss_file = open("./loss_{}.log".format(self.rank_id), "a+")
            loss_file.write("%lu s | epoch: %s step: %s total_loss: %.5f  lr: %.6f" %
                            (time_stamp_current - time_stamp_first, cb_params.cur_epoch_num, cur_step_in_epoch,
                             total_loss, self.lr[cb_params.cur_step_num - 1]))
            loss_file.write("\n")
            loss_file.close()

            self.count = 0
            self.loss_sum = 0


class LossNet(nn.Cell):
    """FasterRcnn loss method"""

    def construct(self, x1, x2, x3, x4, x5, x6):
        return x1 + x2


class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to compute loss.

    Args:
        backbone (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.
    """

    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, x, img_shape, gt_bboxe, gt_label, gt_num):
        loss1, loss2, loss3, loss4, loss5, loss6 = self._backbone(x, img_shape, gt_bboxe, gt_label, gt_num)
        return self._loss_fn(loss1, loss2, loss3, loss4, loss5, loss6)

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, return backbone network.
        """
        return self._backbone


_grad_scale = ops.MultitypeFuncGraph("grad_scale")


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.cast(ops.Reciprocal()(scale), ops.dtype(grad))


class TrainOneStepCell(nn.TrainOneStepWithLossScaleCell):
    """
    Network training package class.

    Append an optimizer to the training network after that the construct function
    can be called to create the backward graph.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default value is 1.0.
        grad_clip (bool): Whether clip gradients. Default value is False.
    """

    def __init__(self, network, optimizer, scale_sense=1, grad_clip=False):
        if isinstance(scale_sense, (int, float)):
            scale_sense = ms.Tensor(scale_sense, ms.float32)
        super(TrainOneStepCell, self).__init__(network, optimizer, scale_sense)
        self.grad_clip = grad_clip

    def construct(self, x, img_shape, gt_bboxe, gt_label, gt_num):
        weights = self.weights
        loss = self.network(x, img_shape, gt_bboxe, gt_label, gt_num)
        scaling_sens = self.scale_sense

        _, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = ops.ones_like(loss) * ops.cast(scaling_sens, ops.dtype(loss))
        grads = self.grad(self.network, weights)(x, img_shape, gt_bboxe, gt_label, gt_num, scaling_sens_filled)
        grads = self.hyper_map(ops.partial(_grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        if self.grad_clip:
            grads = ops.clip_by_global_norm(grads)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss
