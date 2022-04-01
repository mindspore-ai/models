# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""hypertext train model"""
from mindspore.ops import Squeeze, Argmax, Cast
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.parameter import ParameterTuple
import mindspore.numpy as mnp
import mindspore.common.dtype as mstype
from mindspore import nn, save_checkpoint
from mindspore.train.callback import Callback
from src.hypertext import HModel

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0

clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


class HModelWithLoss(nn.Cell):
    """loss model"""
    def __init__(self, config):
        """init"""
        super(HModelWithLoss, self).__init__()
        self.hmodel = HModel(config).to_float(mstype.float16)
        self.loss_func = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.squeeze = Squeeze(axis=1)

    def construct(self, x1, x2, label):
        """class construction"""
        out = self.hmodel(x1, x2)
        label = self.squeeze(label)
        predict_score = self.loss_func(out, label)
        return predict_score

class EvalCallBack(Callback):
    """eval"""

    def __init__(self, model, eval_dataset, epoch_per_eval, save_ckpk):
        """init function"""
        self.model = model
        self.eval_dataset = eval_dataset
        self.epoch_per_eval = epoch_per_eval
        self.save_ckpk = save_ckpk
        self.dev_curr = 0

    def step_end(self, run_context):
        """per setp to eval"""
        cb_param = run_context.original_args()
        cur_step = cb_param.cur_step_num
        if cur_step % (self.epoch_per_eval) == 0:
            print(cur_step)
            acc = self.eval_net()
            print(acc)
            if acc > 0.5:
                if self.dev_curr < acc:
                    self.dev_curr = acc
                    save_checkpoint(self.model, self.save_ckpk)

    def eval_net(self):
        """eval net"""
        squ = Squeeze(-1)
        argmax = Argmax(output_type=mstype.int32)
        cur, total = 0, 0
        print('----------start eval model-------------')
        net_work = self.model
        n = 0
        for d in self.eval_dataset.create_dict_iterator():
            if n == 200:
                break
            n += 1
            net_work.set_train(False)
            out = net_work(d['ids'], d['ngrad_ids'])
            predict = argmax(out)
            acc = predict == squ(d['label'])
            acc = mnp.array(acc, dtype=mnp.float32)
            cur += (mnp.sum(acc, -1))
            total += len(acc)
        return cur / total


class HModelTrainOneStepCell(nn.Cell):
    """train loss"""
    def __init__(self, network, optimizer, sens=1.0):
        """init fun"""
        super(HModelTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True)
        self.sens = sens
        self.hyper_map = C.HyperMap()
        self.cast = Cast()
        self.hyper_map = C.HyperMap()
        self.cast = Cast()

    def set_sens(self, value):
        """set sense"""
        self.sens = value

    def construct(self, x1, x2, label):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(x1, x2, label)
        gradient_function = self.grad(self.network, weights)
        grads = gradient_function(x1, x2, label)
        self.optimizer(grads)
        return loss
