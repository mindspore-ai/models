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
"""util"""
import os
import random
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.train.callback import Callback

class LossMonitor_mine(Callback):
    """LossMonitor"""
    def __init__(self, per_print_times, learning_rate):
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.loss_list = []
        self.learning_rate = learning_rate

    def epoch_begin(self, run_context):
        """epoch begin"""
        cb_params = run_context.original_args()
        print("epoch:%d lr: %s" % (cb_params.cur_epoch_num, \
            self.learning_rate[cb_params.cur_step_num]))

    def step_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))

        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            self.loss_list.append(loss)
            print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num, \
                cur_step_in_epoch, loss))
            print("average loss is %s" % (np.mean(self.loss_list)))
            print()

    def epoch_end(self, run_context):
        """epoch end"""
        self.loss_list = []

def getBool(string):
    """from str to bool"""
    if string == "true":
        b = True
    elif string == "false":
        b = False
    else:
        raise RuntimeError('string should be "true" or "false"')
    return b

def seed_seed(seed=2):
    """set random seed"""
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    mindspore.set_seed(seed)

def getLR(maxLR, start_epoch, epoch_num, epoch_step_num, run_distribute=False, \
    global_size=1, repeat=1):
    """generate learning rate"""
    if run_distribute:
        epoch_step_num = int(repeat * epoch_step_num / global_size)
    LR = np.arange(epoch_num).repeat(epoch_step_num, 0) + 1
    LR = np.power(1 - ((LR - 1) / epoch_num), 0.9) * maxLR
    LR = LR[start_epoch * epoch_step_num:]
    LR_1 = np.zeros(epoch_step_num)
    LR_1[:] = LR[-1]
    LR = np.concatenate((LR, LR_1), axis=0)
    return Tensor(LR, dtype=mindspore.float32)

def getCityLossWeight(encode):
    """class weight for balance"""
    # calculate weights by processing dataset histogram
    # create a loder to run all images and calculate histogram of labels,
    # then create weight array using class balancing
    weight = Tensor(np.zeros(20, dtype=np.float32))
    if encode:
        weight[0] = 2.3653597831726
        weight[1] = 4.4237880706787
        weight[2] = 2.9691488742828
        weight[3] = 5.3442072868347
        weight[4] = 5.2983593940735
        weight[5] = 5.2275490760803
        weight[6] = 5.4394111633301
        weight[7] = 5.3659925460815
        weight[8] = 3.4170460700989
        weight[9] = 5.2414722442627
        weight[10] = 4.7376127243042
        weight[11] = 5.2286224365234
        weight[12] = 5.455126285553
        weight[13] = 4.3019247055054
        weight[14] = 5.4264230728149
        weight[15] = 5.4331531524658
        weight[16] = 5.433765411377
        weight[17] = 5.4631009101868
        weight[18] = 5.3947434425354
    else:
        weight[0] = 2.8149201869965
        weight[1] = 6.9850029945374
        weight[2] = 3.7890393733978
        weight[3] = 9.9428062438965
        weight[4] = 9.7702074050903
        weight[5] = 9.5110931396484
        weight[6] = 10.311357498169
        weight[7] = 10.026463508606
        weight[8] = 4.6323022842407
        weight[9] = 9.5608062744141
        weight[10] = 7.8698215484619
        weight[11] = 9.5168733596802
        weight[12] = 10.373730659485
        weight[13] = 6.6616044044495
        weight[14] = 10.260489463806
        weight[15] = 10.287888526917
        weight[16] = 10.289801597595
        weight[17] = 10.405355453491
        weight[18] = 10.138095855713
    weight[19] = 0
    return weight
