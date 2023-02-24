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

import time
import random
import argparse
import yaml
import numpy as np
from src import utils
from src.utils import load_graph_data
from src.dcrnn_model import DCRNNModel
from src.loss import MaskedMAELoss, MyMAE
import mindspore
from mindspore import context, Tensor
from mindspore import dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P
import mindspore.ops as ops


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


class DCRNNEVAL:
    def __init__(self, arg, adj_mx, **kwargs):
        self.device = mindspore.context.set_context(mode=context.GRAPH_MODE, device_target=arg.device)
        self._kwargs = kwargs
        self.config = arg
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)

        # data set
        self._data = utils.get_loader_dataset(config)
        self.eval_data = utils.get_eval_dataset(config)
        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder

        # setup model
        dcrnn_model = DCRNNModel(adj_mx, False,
                                 kwargs.get('data').get('batch_size'), **self._model_kwargs)
        self.dcrnn_model = dcrnn_model
        self.criterion = MaskedMAELoss(self.horizon, self.num_nodes,
                                       self.output_dim, self._kwargs.get('data').get('batch_size'))

        self._epoch_num = self._train_kwargs.get('epoch', 0)

    def eval(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._eval(**kwargs)

    def _eval(self, base_lr, epsilon, **kwargs):

        print("Model has been completed")
        self.dcrnn_model.set_train(False)
        pretrained_weights = mindspore.load_checkpoint(config.ckpt_path)
        param_not_load, _ = mindspore.load_param_into_net(self.dcrnn_model, pretrained_weights)
        print('param not load:', param_not_load)
        total_steps = self.eval_data.get_dataset_size()
        print('total steps:', total_steps)
        print('running evaluation')

        eval_net = WithEvalCell(self.dcrnn_model, self.criterion)
        eval_net.set_train(False)

        mae = MyMAE(self.horizon, self.num_nodes, self.output_dim, self._kwargs.get('data').get('batch_size'))
        eval_dataset = self.eval_data
        mae.clear()

        for data in eval_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            outputs = eval_net(Tensor(data["data"]), Tensor(data["label"]))
            mae.update(outputs[0].asnumpy(), outputs[1].asnumpy())

        mae_result = mae.eval()
        print("mae: ", mae_result)


def main(arg):
    tic = time.time()
    mindspore.dataset.config.set_seed(2022)
    mindspore.set_seed(2022)
    np.random.seed(2022)
    random.seed(2022)

    assert arg.context in ['py', 'gr']
    if arg.context == 'py':
        context.set_context(mode=context.PYNATIVE_MODE, device_target=arg.device,
                            device_id=arg.device_id)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=arg.device, device_id=arg.device_id)

    with open(arg.config_filename) as f:
        print('Start Reading Config File')
        print(arg.device_id)
        supervisor_config = yaml.safe_load(f)
        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        _, _, adj_mx = load_graph_data(graph_pkl_filename)

        supervisorEval = DCRNNEVAL(arg, adj_mx=adj_mx, **supervisor_config)
        supervisorEval.eval()

    toc = time.time()
    total_time = toc - tic
    print('total_time:', total_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='data/model/dcrnn_la.yaml', type=str)
    parser.add_argument('--context', default='py', type=str)
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--save_dir', default='./output_standalone', type=str, help='Where to save training outputs')
    parser.add_argument('--ckpt_path', default='dcrnn-1_375.ckpt', type=str, help='where to put ckpt file')
    parser.add_argument('--device', default='Ascend', help='Device')
    parser.add_argument('--device_id', default=4, type=int, help='ID of the target device')
    parser.add_argument('--is_fp16', default=False, type=str, help='cast to fp16 or not')
    parser.add_argument('--distributed', type=str, default=False, help='distribute train')
    parser.add_argument('--num_workers', type=int, default=1, help='workers')

    config = parser.parse_args()

    main(config)
