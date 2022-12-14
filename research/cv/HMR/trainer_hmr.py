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
import os
from src.config import config
from src.dataset import Data
from src.cal_loss import CalcLossG, CalcLossD
from src.model import HMRNetBase, Discriminator
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import save_checkpoint, context, nn, ops, load_checkpoint, set_seed
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
set_seed(1234)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class CustomTrainOneStepCell(nn.Cell):

    def __init__(self, network, optimizer, sens=1.0):
        super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = float(sens)
        self.depend = ops.Depend()
        self.reducer_flag = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [
                ParallelMode.DATA_PARALLEL,
                ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True

        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = context.get_auto_parallel_context("device_num")
            self.grad_reducer = nn.DistributedGradReducer(
                self.optimizer.parameters, mean, degree)

    def construct(self, *args_):
        """opt"""
        weights = self.weights
        loss = self.network(*args_)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args_, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        return self.depend(loss, self.optimizer(grads))


class NetWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(NetWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label, label1):
        out = self._backbone(data)
        return self._loss_fn(out, label, label1)


class HMRTrainer(Data):
    def __init__(self, rank_id=None, device_num=None):
        self._build_model()
        self.pixelformat = 'NCHW'
        self.Normalization = True
        self.pro_flip = 0.5
        self.rank_id = rank_id
        self.device_num = device_num
        self.is_flip = True
        self.op_0 = ops.Concat(0)
        self.loss_G = CalcLossG(self.discriminator)
        self.loss_D = CalcLossD(self.discriminator)
        self.e_opt = nn.Adam(self.generator.trainable_params(),
                             learning_rate=config.e_lr,
                             weight_decay=config.e_wd)

        self.d_opt = nn.Adam(self.discriminator.get_parameters(),
                             learning_rate=config.d_lr,
                             weight_decay=config.d_wd)
        self._create_data_loader()
        print(
            'sself.rank_id',
            self.rank_id,
            'self.device_num ',
            self.device_num)

    def _build_model(self):
        '''
            load  model
        '''
        print('start building model.')
        self.generator = HMRNetBase()
        self.discriminator = Discriminator()
        print('finished build model.')

    def train(self):

        loader_2d, loader_3d, loader_mosh = iter(
            self.loader_2d.create_dict_iterator()), iter(
                self.loader_3d.create_dict_iterator()), iter(
                    self.loader_mosh.create_dict_iterator())

        loss_net_G = NetWithLossCell(self.generator, self.loss_G)
        train_network_G = CustomTrainOneStepCell(loss_net_G, self.e_opt)

        train_network_G.set_train()

        loss_net_D = nn.WithLossCell(self.discriminator, self.loss_D)
        train_network_D = CustomTrainOneStepCell(loss_net_D, self.d_opt)

        train_network_D.set_train()
        if config.checkpoint_file_path:
            load_checkpoint(config.checkpoint_file_path, self.generator)

        for iter_index in range(config.iter_count):
            try:
                data_ = next(loader_2d)
                data_2d_data, data_2d_label = data_['data'], data_['label']
            except StopIteration:
                loader_2d = iter(self.loader_2d.create_dict_iterator())
                data_ = next(loader_2d)
                data_2d_data, data_2d_label = data_['data'], data_['label']
            try:
                data_ = next(loader_3d)
                data_3d_data, data_3d_label = data_['data'], data_['label']
            except StopIteration:
                loader_3d = iter(self.loader_3d.create_dict_iterator())
                data_ = next(loader_3d)
                data_3d_data, data_3d_label = data_['data'], data_['label']
            try:
                data_ = next(loader_mosh)
                data_mosh_data = data_['data']
            except StopIteration:
                loader_mosh = iter(self.loader_mosh.create_dict_iterator())
                data_ = next(loader_mosh)
                data_mosh_data = data_['data']
            net_time1 = time.time()
            images = self.op_0((data_2d_data, data_3d_data))
            generator_outputs = self.generator(images)
            dis_input = self.op_0((generator_outputs[0], data_mosh_data))
            grads_G = train_network_G(images, data_2d_label, data_3d_label)
            grads_D = train_network_D(dis_input, None)
            net_time4 = time.time()
            net_cost = (net_time4 - net_time1) * 1000
            info_ = {
                "iter: ":
                iter_index,
                ", G Loss: ":
                grads_G.asnumpy(),
                ", D Loss: ":
                grads_D.asnumpy(),
                ", per step time: ":
                str(net_cost) + "(ms)"}
            print(info_)
            if iter_index == 20000:
                if not os.path.exists(config.model_save):
                    os.mkdir(config.model_save)
                else:
                    print('the file exists')
                gen_path = os.path.join(
                    config.model_save,
                    'generator-' +
                    str(iter_index) +
                    '.ckpt')
                save_checkpoint(self.generator, gen_path)


if __name__ == '__main__':

    if config.run_distribute == 'True':
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
        init()
        rank_id_ = get_rank()
        device_num_ = get_group_size()
        context.set_auto_parallel_context(
            device_num=device_num_,
            gradients_mean=False,
            parallel_mode=ParallelMode.DATA_PARALLEL)

    else:
        context.set_context(
            mode=context.GRAPH_MODE,
            device_target="Ascend")
        rank_id_ = None
        device_num_ = None
    trainer = HMRTrainer(rank_id=rank_id_, device_num=device_num_)
    trainer.train()
