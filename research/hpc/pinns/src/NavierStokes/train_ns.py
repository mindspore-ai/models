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
"""Train PINNs for Navier-Stokes equation scenario"""
import os
import numpy as np
from mindspore import Model, context, nn, save_checkpoint
from mindspore.common import set_seed
from mindspore.train.callback import LossMonitor, TimeMonitor, Callback
from src.NavierStokes.dataset import generate_training_set_navier_stokes
from src.NavierStokes.loss import PINNs_loss_navier
from src.NavierStokes.net import PINNs_navier


class EvalCallback(Callback):
    """eval callback."""
    def __init__(self, net, ckpt_dir, per_eval_epoch, eval_begin_epoch=1000):
        super(EvalCallback, self).__init__()
        if not isinstance(per_eval_epoch, int) or per_eval_epoch <= 0:
            raise ValueError("per_eval_epoch must be int and > 0")
        self.network = net
        self.ckpt_dir = ckpt_dir
        self.per_eval_epoch = per_eval_epoch
        self.eval_begin_epoch = eval_begin_epoch
        self.best_result = None
        self.error1 = 0.3
        self.error2 = 1.0

    def epoch_end(self, run_context):
        """epoch end function."""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        if cur_epoch % self.per_eval_epoch == 0 and cur_epoch >= self.eval_begin_epoch:
            lambda1_pred = self.network.lambda1.asnumpy()
            lambda2_pred = self.network.lambda2.asnumpy()
            error1 = np.abs(lambda1_pred - 1.0) * 100
            error2 = np.abs(lambda2_pred - 0.01) / 0.01 * 100
            print(f'Error of lambda 1 is {error1[0]:.6f}%')
            print(f'Error of lambda 2 is {error2[0]:.6f}%')
            if self.best_result is None or (error1 < self.error1 and error2 < self.error2):
                if self.best_result is not None:
                    self.error1 = error1
                    self.error2 = error2
                self.best_result = error1 + error2
                save_checkpoint(self.network, os.path.join(self.ckpt_dir, "best_result.ckpt"))


def train_navier(epoch, lr, batch_size, n_train, path, noise, num_neuron, ck_path, seed=None):
    """
    Train PINNs for Navier-Stokes equation

    Args:
        epoch (int): number of epochs
        lr (float): learning rate
        batch_size (int): amount of data per batch
        n_train(int): amount of training data
        noise (float): noise intensity, 0 for noiseless training data
        path (str): path of dataset
        num_neuron (int): number of neurons for fully connected layer in the network
        ck_path (str): path to store the checkpoint file
        seed (int): random seed
    """
    if seed is not None:
        np.random.seed(seed)
        set_seed(seed)
    os.makedirs(ck_path, exist_ok=True)
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU', runtime_num_threads=60)

    layers = [3, num_neuron, num_neuron, num_neuron, num_neuron, num_neuron, num_neuron, num_neuron,
              num_neuron, 2]

    training_set, lb, ub = generate_training_set_navier_stokes(batch_size, n_train, path, noise)
    n = PINNs_navier(layers, lb, ub)
    opt = nn.Adam(n.trainable_params(), learning_rate=lr)
    loss = PINNs_loss_navier()
    eval_cb = EvalCallback(n, ckpt_dir=ck_path, per_eval_epoch=2)
    model = Model(network=n, loss_fn=loss, optimizer=opt)
    model.train(epoch=epoch // 10, train_dataset=training_set,
                callbacks=[LossMonitor(1), TimeMonitor(1), eval_cb], dataset_sink_mode=True,
                sink_size=training_set.get_dataset_size() * 10)
    print('Training complete')
