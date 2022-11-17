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
"""train"""
import os
import time
import glob
import argparse
import ast
import numpy as np


import mindspore
import mindspore.context as context
from mindspore import Tensor
from mindspore import nn
from mindspore.communication import get_rank, init, get_group_size
from mindspore.nn import DynamicLossScaleUpdateCell
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, _InternalCallbackParam, RunContext

# from model_utils.config import config as cfg
from src.dataset import create_dataset
from src.loss_scale import TrainOneStepWithLossScaleCell
from src.model_reposity import Resnet18_8s, NetworkWithLossCell
from src.net_utils import AverageMeter, adjust_learning_rate
import moxing as mox


from model_utils.config import config as cfg


loss_rec = AverageMeter()
recs = [loss_rec]
print(os.system('env'))


def export_AIR(args_opt):
    """start modelarts export"""
    ckpt_list = glob.glob(os.path.join(args_opt.modelarts_result_dir, args_opt.cls_name, "train*.ckpt"))
    if not ckpt_list:
        print("ckpt file not generated.")

    ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1]
    print("checkpoint path", ckpt_model)

    # if args.device_target == "Ascend":
    #     context.set_context(device_id=args.rank)
    net = Resnet18_8s(ver_dim=args.vote_num * 2)
    param_dict = mindspore.load_checkpoint(ckpt_model)
    mindspore.load_param_into_net(net, param_dict)
    net.set_train(False)
    input_data = Tensor(np.zeros([1, 3, args.img_height, args.img_width]), mindspore.float32)
    mindspore.export(net, input_data, file_name=args.file_name, file_format=args.file_format)


class Train:
    """PVNet Train class"""

    def __init__(self, arg):
        """__init__"""
        self.cls_num = 1 + len(arg.cls_name.split(','))
        self.arg = arg
        self.dataset = create_dataset(
            cls_list=arg.cls_name,
            batch_size=arg.batch_size,
            workers=arg.workers_num,
            devices=arg.group_size,
            rank=arg.rank
        )
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        if arg.pretrained_path is None:
            self.pretrained_path = None
        else:
            self.pretrained_path = os.path.join(self.current_dir, arg.pretrained_path)
        self.step_per_epoch = self.dataset.get_dataset_size()
        self.dataset = self.dataset.create_tuple_iterator(output_numpy=True, do_copy=False)
        print("cls:{}, device_num:{}, rank:{}, data_size:{}".format(arg.cls_name, arg.group_size, arg.rank,
                                                                    self.step_per_epoch))

    def _build_net(self):
        """ build pvnet network"""
        lr = mindspore.Tensor(adjust_learning_rate(global_step=0,
                                                   lr_init=self.arg.lr,
                                                   lr_decay_rate=self.arg.learning_rate_decay_rate,
                                                   lr_decay_epoch=self.arg.learning_rate_decay_epoch,
                                                   total_epochs=self.arg.epoch_size,
                                                   steps_per_epoch=self.step_per_epoch))

        net = Resnet18_8s(ver_dim=self.arg.vote_num * 2, pretrained_path=self.pretrained_path)
        self.opt = nn.Adam(net.trainable_params(), learning_rate=lr)
        self.net = NetworkWithLossCell(net, cls_num=self.cls_num)
        scale_manager = DynamicLossScaleUpdateCell(loss_scale_value=self.arg.loss_scale_value,
                                                   scale_factor=self.arg.scale_factor,
                                                   scale_window=self.arg.scale_window)
        self.net = TrainOneStepWithLossScaleCell(self.net, self.opt, scale_sense=scale_manager)
        self.net.set_train()

    def train_net(self):
        """ train pvnet network"""
        self._build_net()
        if self.arg.rank == 0:
            self._train_begin()

        for i in range(self.arg.epoch_size):
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
                        self.arg.rank, self.arg.group_size, i + 1, self.arg.epoch_size, idx, self.step_per_epoch,
                        cost_time,
                        recs[0].avg)
                    print(log_str)
                iter_start = time.time()

                if self.arg.rank == 0:
                    self._cb_params.output = total_loss
                    self._cb_params.cur_step_num += 1
                    self._ckpt_saver.step_end(self._run_context)

            print('Epoch Cost:{}'.format(time.time() - start), "seconds.")
            if self.arg.rank == 0:
                self._cb_params.cur_epoch_num += 1

    def _train_begin(self):
        """ the step before training """
        begin_epoch = 0
        cb_params = _InternalCallbackParam()
        cb_params.epoch_num = self.arg.epoch_size
        cb_params.batch_num = self.step_per_epoch
        cb_params.cur_epoch_num = begin_epoch
        cb_params.cur_step_num = begin_epoch * self.step_per_epoch
        cb_params.train_network = self.net
        self._cb_params = cb_params
        self._run_context = RunContext(cb_params)

        ckpt_config = CheckpointConfig(save_checkpoint_steps=self.step_per_epoch,
                                       keep_checkpoint_max=self.arg.keep_checkpoint_max)
        self._ckpt_saver = ModelCheckpoint(
            prefix="train",
            directory=os.path.join(self.arg.modelarts_result_dir, self.arg.cls_name),
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


def parse_args():
    parser = argparse.ArgumentParser('PVNet')
    parser.add_argument("--train_url", type=str, default="./output")
    parser.add_argument("--data_url", type=str, default="./dataset")
    parser.add_argument("--modelarts_data_dir", type=str, default="/cache/dataset")
    parser.add_argument("--modelarts_result_dir", type=str,
                        default="/cache/result")  # modelarts train result: /cache/result
    parser.add_argument('--random_seed', type=int, default=0, help='random_seed')

    parser.add_argument('--cls_name', type=str, default="cat",
                        help='Sub-Dataset to train, for example, cat,ape,cam ')

    parser.add_argument('--epoch_size', type=int, default=1, help='epoch_size')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--pretrain_epoch_size', type=int, default=0, help='pretrain_epoch_size, use with pre_trained')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.5, help='learning_rate_decay_rate')
    parser.add_argument('--learning_rate_decay_epoch', type=int, default=20, help='learning_rate_decay_epoch')
    parser.add_argument('--vote_num', type=int, default=9, help='vote num')
    parser.add_argument('--workers_num', type=int, default=16, help='workers_num')
    parser.add_argument('--group_size', type=int, default=1, help='group_size')
    parser.add_argument('--rank', type=int, default=0, help='rank')
    parser.add_argument('--loss_scale_value', type=int, default=1024, help='loss_scale_value')
    parser.add_argument('--scale_factor', type=int, default=2, help='scale_factor')
    parser.add_argument('--scale_window', type=int, default=1000, help='scale_window')
    parser.add_argument('--keep_checkpoint_max', type=int, default=10, help='keep_checkpoint_max')

    # Do not change the following two hyper-parameter, it conflicts with pvnet_linemod_config.yaml
    parser.add_argument('--img_height', type=int, default=480, help='img_height')
    parser.add_argument('--img_width', type=int, default=640, help='img_width')


    parser.add_argument('--distribute', type=ast.literal_eval, default=False, help='Run distribute')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                        help="Device target, support Ascend, GPU and CPU.")
    parser.add_argument('--pretrained_path', type=str, default="./resnet18-5c106cde.ckpt",
                        help='Pretrained checkpoint path')
    parser.add_argument('--file_name', type=str, default='pvnet', help='output air file name')
    parser.add_argument('--file_format', type=str, default='AIR', help='file_format')
    return parser.parse_args()


# _CACHE_DATA_URL = "/cache/data_url"
# _CACHE_TRAIN_URL = "/cache/train_url"
if __name__ == '__main__':
    args = parse_args()
    cfg.data_url = args.data_url
    mindspore.set_seed(args.random_seed)
    network_init(args)
    ## copy dataset from obs to modelarts
    os.makedirs(args.modelarts_data_dir, exist_ok=True)
    os.makedirs(args.modelarts_result_dir, exist_ok=True)
    mox.file.copy_parallel(args.data_url, args.modelarts_data_dir)
    train = Train(args)
    train.train_net()
    ## start export air
    export_AIR(args)
    ## copy result from modelarts to obs
    mox.file.copy_parallel(args.modelarts_result_dir, args.train_url)
    air_file = args.file_name + ".air"
    mox.file.copy(src_url=air_file, dst_url=os.path.join(args.train_url, air_file))
