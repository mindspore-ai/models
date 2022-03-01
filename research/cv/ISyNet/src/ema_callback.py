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
"""custom callbacks for ema and loss"""
from copy import deepcopy

from mindspore.train.callback import Callback
from mindspore.common.parameter import Parameter
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor


def load_nparray_into_net(net, array_dict):
    """
    Loads dictionary of numpy arrays into network.

    Args:
        net (Cell): Cell network.
        array_dict (dict): dictionary of numpy array format model weights.
    """
    param_not_load = []
    for _, param in net.parameters_and_names():
        if param.name in array_dict:
            new_param = array_dict[param.name]
            param.set_data(Parameter(Tensor(deepcopy(new_param)), name=param.name))
        else:
            param_not_load.append(param.name)
    return param_not_load


class EmaEvalCallBack(Callback):
    """
    Call back that will evaluate the model and save model checkpoint at
    the end of training epoch.

    Args:
        network: tinynet network instance.
        ema_network: step-wise exponential moving average of network.
        eval_dataset: the evaluation daatset.
        decay (float): ema decay.
        save_epoch (int): defines how often to save checkpoint.
        dataset_sink_mode (bool): whether to use data sink mode.
        start_epoch (int): which epoch to start/resume training.
    """

    def __init__(self, network, ema_network, loss_fn, decay=0.9,
                 save_epoch=1, dataset_sink_mode=True, start_epoch=0, save_path='./'):
        self.save_path = save_path
        self.network = network
        self.ema_network = ema_network
        self.loss_fn = loss_fn
        self.decay = decay
        self.save_epoch = save_epoch
        self.shadow = {}
        self._start_epoch = start_epoch
        self.dataset_sink_mode = dataset_sink_mode

    def begin(self, run_context):
        """Initialize the EMA parameters """
        _ = run_context # Not used
        for _, param in self.network.parameters_and_names():
            self.shadow[param.name] = deepcopy(param.data.asnumpy())

    def step_end(self, run_context):
        """Update the EMA parameters"""
        _ = run_context # Not used
        for _, param in self.network.parameters_and_names():
            new_average = (1.0 - self.decay) * param.data.asnumpy().copy() + \
                self.decay * self.shadow[param.name]
            self.shadow[param.name] = new_average

    def epoch_end(self, run_context):
        """evaluate the model and ema-model at the end of each epoch"""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num + self._start_epoch - 1

        save_ckpt = (cur_epoch % self.save_epoch == 0)
        output = [{"name": k, "data": Tensor(v)}
                  for k, v in self.shadow.items()]

        if save_ckpt:
            # Save the ema_model checkpoints
            ckpt = f'{self.save_path}/ema-{cur_epoch}.ckpt'
            save_checkpoint(output, ckpt)
