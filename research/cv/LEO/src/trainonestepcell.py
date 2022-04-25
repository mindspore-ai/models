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
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, ParameterTuple
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size


def init_group_params(weights):
    decayed_params = []
    no_decayed_params = []
    lr_list = ["leo.inner_lr", "leo.finetuning_lr"]
    for param in weights:
        if param.name in lr_list:
            print(f"======= append {param.name} in no_decayed_params ========")
            no_decayed_params.append(param)
        else:
            decayed_params.append(param)

    group_params = [{'params': no_decayed_params, 'weight_decay': 0.0},
                    {'params': decayed_params},
                    {'order_params': weights}]
    return group_params

class TrainOneStepCell(nn.Cell):
    def __init__(self, TrainNet, outer_lr, outer_weight_decay, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = TrainNet
        self.outer_lr = outer_lr
        self.outer_weight_decay = outer_weight_decay
        self.weights = ParameterTuple(self.network.trainable_params())
        self.group_params = init_group_params(self.weights)
        self.leo_parm_opt = nn.Adam(self.group_params, learning_rate=self.outer_lr,
                                    weight_decay=self.outer_weight_decay)

        self.grad = ops.GradOperation(get_by_list=True, sens_param=False)

        self.gradient_threshold = 0.1
        self.gradient_norm_threshold = 0.1

        self.clip_by_value = ops.clip_by_value
        self.clip_by_global_norm = ops.clip_by_global_norm

        self.reducer_flag = False
        self.grad_reducer = None

        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(self.weights, mean, degree)

    def construct(self, train_inputs, train_labels, val_inputs, val_labels, train):
        val_loss, val_acc = self.network(train_inputs, train_labels, val_inputs, val_labels, train)

        gradient_function = self.grad(self.network, self.weights)
        grads = gradient_function(train_inputs, train_labels, val_inputs, val_labels, train)

        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)

        if self.gradient_threshold > 0:
            grads_weights = ()
            for g in grads:
                clipped_g = self.clip_by_value(g, -self.gradient_threshold, self.gradient_threshold)
                grads_weights += (clipped_g,)
            grads = grads_weights
        if self.gradient_norm_threshold > 0:
            grads_weights = self.clip_by_global_norm(grads, self.gradient_norm_threshold)
            grads = grads_weights

        self.leo_parm_opt(grads)

        return val_loss, val_acc
