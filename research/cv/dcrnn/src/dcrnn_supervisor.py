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
import random
import numpy as np
from src import utils
from src.dcrnn_model import DCRNNModel
from src.loss import MyMAE, MaskedMAELoss
import mindspore
from mindspore import context, Model
from mindspore import dtype as mstype
from mindspore.context import ParallelMode
import mindspore.nn as nn
from mindspore.communication.management import init, get_rank
from mindspore.ops import operations as P
import mindspore.ops as ops
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, LearningRateScheduler

class WithEvalCell(nn.Cell):
    def __init__(self, network, add_cast_fp32=False):
        super(WithEvalCell, self).__init__(auto_prefix=False)
        self.num_nodes = 207
        self.output_dim = 1
        self.horizon = 12  # for the decoder
        self.cast = P.Cast()
        self.transpose = ops.Transpose()
        self._network = network

    def construct(self, data, label):
        outputs = self._network(data)

        label = self.cast(label, mstype.float32) #y_true (64, 12, 207, 2)
        label = self.transpose(label, (1, 0, 2, 3))
        label = label[..., :self.output_dim].view(self.horizon, 64, self.num_nodes * self.output_dim)

        outputs = self.cast(outputs, mstype.float32) #y_pred
        return outputs, label


class DCRNNSupervisor:
    def __init__(self, config, adj_mx, **kwargs):

        self._kwargs = kwargs
        self.config = config
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)
        if config.distributed:
            if config.context == 'py':
                context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target,
                                    device_id=int(os.environ["DEVICE_ID"]))
            else:
                context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target,
                                    device_id=int(os.environ["DEVICE_ID"]))

            config.device_id = int(os.environ["DEVICE_ID"])
            init()
            context.set_auto_parallel_context(device_num=config.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
        else:
            if config.context == 'py':
                self.device = mindspore.context.set_context(mode=context.PYNATIVE_MODE,
                                                            device_target=config.device_target,
                                                            device_id=config.device_id)
            else:
                self.device = mindspore.context.set_context(mode=context.GRAPH_MODE,
                                                            device_target=config.device_target,
                                                            device_id=config.device_id)

        # data set
        self._data = utils.get_loader_dataset(config)
        self.eval_data = utils.get_eval_dataset(config)
        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder

        # is_fp16
        self.is_fp16 = config.is_fp16

        # setup model
        dcrnn_model = DCRNNModel(adj_mx,
                                 self.is_fp16, kwargs.get('data').get('batch_size'), **self._model_kwargs)
        self.dcrnn_model = dcrnn_model
        self.criterion = MaskedMAELoss(self.horizon, self.num_nodes,
                                       self.output_dim, self._kwargs.get('data').get('batch_size'))

        self._epoch_num = self._train_kwargs.get('epoch', 0)

    def load_model(self):
        self._setup_graph()
        assert os.path.exists('models/epo%d.tar' % self._epoch_num), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = mindspore.load_checkpoint('models/epo%d.tar' % self._epoch_num, map_location='cpu')
        self.dcrnn_model.load_state_dict(checkpoint['model_state_dict'])

    def _setup_graph(self):
        self.dcrnn_model = self.dcrnn_model.eval()
        val_iterator = self._data['val_loader'].get_iterator()

        for _, (x, y) in enumerate(val_iterator):
            x, y = self._prepare_data(x, y)
            # output = self.dcrnn_model(x)
            break

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def eval(self):
        eval_net = WithEvalCell(self.dcrnn_model)
        eval_net.set_train(False)

        mae = MyMAE(self.horizon, self.num_nodes, self.output_dim, self._kwargs.get('data').get('batch_size'))
        eval_dataset = self.eval_data
        mae.clear()

        for data in eval_dataset.create_dict_iterator():
            outputs = eval_net(data["data"], data["label"])
            mae.update(outputs[0], outputs[1])

        mae_result = mae.eval()
        print("mae: ", mae_result)

    def _train(self, base_lr, epsilon=1e-8, **kwargs):

        mindspore.set_seed(2022)
        random.seed(2022)
        np.random.seed(2022)
        mindspore.dataset.config.set_seed(2022)

        optimizer = mindspore.nn.Adam(self.dcrnn_model.get_parameters(), learning_rate=base_lr, eps=epsilon)
        criterion = self.criterion
        dataset = self._data
        step_size = dataset.get_dataset_size()
        print('Step size per epoch:', step_size)
        lr_de_steps = step_size * self.config.lr_de_epochs

        def learning_rate_function(lr, cur_step_num):
            if cur_step_num % lr_de_steps == 0:
                lr = lr * 0.1
            return lr

        time_cb = TimeMonitor(data_size=step_size)
        loss_cb = LossMonitor()
        lr_cb = LearningRateScheduler(learning_rate_function)
        loss_scale_manager = DynamicLossScaleManager()
        config_ck = CheckpointConfig(save_checkpoint_steps=self.config.checkpoint_frequency * step_size,
                                     keep_checkpoint_max=self.config.checkpoints_num_keep)
        cb = [time_cb, loss_cb, lr_cb]
        ckpt_cb = ModelCheckpoint(prefix="dcrnn", directory=self.config.save_dir, config=config_ck)
        if self.config.distributed:
            if get_rank() == 0:
                cb.append(ckpt_cb)
        else:
            cb.append(ckpt_cb)
        model = Model(self.dcrnn_model, optimizer=optimizer, loss_fn=criterion,
                      amp_level=self.config.amp_level, loss_scale_manager=loss_scale_manager)
        model.train(epoch=self._epoch_num, train_dataset=dataset,
                    callbacks=cb, dataset_sink_mode=self.config.sink_mode)

        if self.config.distributed:
            if get_rank() == 0:
                print('Evaluation separately ...')
        else:
            self.eval()
