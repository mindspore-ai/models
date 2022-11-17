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
"""train"""
import os
import time

import mindspore
import mindspore.context as context
from mindspore import Tensor
from mindspore import nn
from mindspore.communication import get_rank, init, get_group_size
from mindspore.nn import DynamicLossScaleUpdateCell
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, _InternalCallbackParam, RunContext

from model_utils.config import config as cfg
from src.dataset import create_dataset
from src.loss_scale import TrainOneStepWithLossScaleCell
from src.model_reposity import Resnet18_8s, NetworkWithLossCell
from src.net_utils import AverageMeter, adjust_learning_rate

mindspore.set_seed(0)
loss_rec = AverageMeter()
recs = [loss_rec]


class Train:
    """PVNet Train class"""

    def __init__(self, cls_list=None):
        """__init__"""
        self.cls_num = 1 + len(cls_list.split(','))
        self.dataset = create_dataset(
            cls_list=cls_list,
            batch_size=cfg.batch_size,
            workers=cfg.workers_num,
            devices=cfg.group_size,
            rank=cfg.rank
        )
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        if str(cfg.pretrained_path).lower() == 'none':
            self.pretrained_path = None
        else:
            self.pretrained_path = os.path.join(self.current_dir, cfg.pretrained_path)
        self.step_per_epoch = self.dataset.get_dataset_size()
        self.dataset = self.dataset.create_tuple_iterator(output_numpy=True, do_copy=False)
        print("cls:{}, device_num:{}, rank:{}, data_size:{}".format(cls_list, cfg.group_size, cfg.rank,
                                                                    self.step_per_epoch))

    def _build_net(self):
        """ build pvnet network"""
        lr = mindspore.Tensor(adjust_learning_rate(global_step=0,
                                                   lr_init=cfg.learning_rate,
                                                   lr_decay_rate=cfg.learning_rate_decay_rate,
                                                   lr_decay_epoch=cfg.learning_rate_decay_epoch,
                                                   total_epochs=cfg.epoch_size,
                                                   steps_per_epoch=self.step_per_epoch))

        net = Resnet18_8s(ver_dim=cfg.vote_num * 2, pretrained_path=self.pretrained_path)
        self.opt = nn.Adam(net.trainable_params(), learning_rate=lr)
        self.net = NetworkWithLossCell(net, cls_num=self.cls_num)
        scale_manager = DynamicLossScaleUpdateCell(loss_scale_value=cfg.loss_scale_value,
                                                   scale_factor=cfg.scale_factor,
                                                   scale_window=cfg.scale_window)
        self.net = TrainOneStepWithLossScaleCell(self.net, self.opt, scale_sense=scale_manager)
        self.net.set_train()

    def train_net(self):
        """ train pvnet network"""
        self._build_net()
        if cfg.rank == 0:
            self._train_begin()

        for i in range(cfg.epoch_size):
            start = time.time()
            iter_start = time.time()
            for idx, data in enumerate(self.dataset):
                for rec in recs:
                    rec.reset()
                cost_time = time.time() - iter_start
                image, mask, vertex, vertex_weight = data

                image = Tensor.from_numpy(image)
                mask = Tensor.from_numpy(mask)
                vertex = Tensor.from_numpy(vertex)
                vertex_weight = Tensor.from_numpy(vertex_weight)
                total_loss = self.net(image, mask, vertex, vertex_weight)

                for rec, val in zip(recs, total_loss):
                    rec.update(val)

                if idx % 80 == 0:
                    log_str = "Rank:{}/{}, Epoch:[{}/{}], Step[{}/{}] cost:{}.s total:{}".format(
                        cfg.rank, cfg.group_size, i + 1, cfg.epoch_size, idx, self.step_per_epoch, cost_time,
                        recs[0].avg)
                    print(log_str)
                iter_start = time.time()

                if cfg.rank == 0:
                    self._cb_params.output = total_loss
                    self._cb_params.cur_step_num += 1
                    self._ckpt_saver.step_end(self._run_context)

            print('Epoch Cost:{}'.format(time.time() - start), "seconds.")
            if cfg.rank == 0:
                self._cb_params.cur_epoch_num += 1

    def _train_begin(self):
        """ the step before training """
        begin_epoch = 0
        cb_params = _InternalCallbackParam()
        cb_params.epoch_num = cfg.epoch_size
        cb_params.batch_num = self.step_per_epoch
        cb_params.cur_epoch_num = begin_epoch
        cb_params.cur_step_num = begin_epoch * self.step_per_epoch
        cb_params.train_network = self.net
        self._cb_params = cb_params
        self._run_context = RunContext(cb_params)

        ckpt_config = CheckpointConfig(save_checkpoint_steps=self.step_per_epoch,
                                       keep_checkpoint_max=cfg.keep_checkpoint_max)
        self._ckpt_saver = ModelCheckpoint(
            prefix="train",
            directory=os.path.join(cfg.train_url, cfg.cls_name),
            config=ckpt_config
        )
        self._ckpt_saver.begin(self._run_context)


def network_init(argvs):
    """ init distribute training """
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=argvs.device_target,
                        save_graphs=False,
                        device_id=int(os.getenv('DEVICE_ID', '0')),
                        reserve_class_name_in_scope=False)
    # Init distributed
    if argvs.distribute:
        init()
        argvs.rank = get_rank()
        argvs.group_size = get_group_size()
        context.reset_auto_parallel_context()
        parallel_mode = context.ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=argvs.group_size)


if __name__ == '__main__':
    network_init(cfg)
    train = Train(cls_list=cfg.cls_name)
    train.train_net()
