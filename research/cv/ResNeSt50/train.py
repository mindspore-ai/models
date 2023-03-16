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
"""training script"""

import os
import time
import datetime
import ast
import argparse
import numpy as np
from mindspore import Tensor, context, nn
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, LossMonitor
from mindspore.train.callback import CheckpointConfig, Callback, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
from mindspore.common import set_seed

from src.datasets.dataset import ImageNet
from src.models.utils import get_lr, get_param_groups
from src.models.resnest import get_network
from src.config import config_train as config
from src.logging import get_logger
from src.crossentropy import CrossEntropy
from src.eval_callback import EvalCallBack


class ProgressMonitor(Callback):
    """monitor loss and time"""
    def __init__(self, args_, config_):
        super().__init__()
        self.me_epoch_start_time = 0
        self.me_epoch_start_step_num = 0
        self.args = args_
        self.config = config_
        self.ckpt_history = []
        self.outputs_dir = os.path.join(args_.outdir,
                                        datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
        self.rank = config_.rank
        self.logger = get_logger(self.outputs_dir, self.rank)

    def begin(self, run_context):
        self.logger.info('start network train...')

    def epoch_begin(self, run_context):
        self.me_epoch_start_time = time.time()

    def epoch_end(self, run_context, *me_args):
        """describe network construct"""
        cb_params = run_context.original_args()
        me_step = cb_params.cur_step_num

        real_epoch = me_step // self.config.steps_per_epoch
        time_used = time.time() - self.me_epoch_start_time
        fps_mean = (self.config.batch_size * (me_step-self.me_epoch_start_step_num))
        print("fps_mean {} time_used {}".format(fps_mean, time_used))
        fps_mean = fps_mean * self.config.group_size
        fps_mean = fps_mean / time_used
        self.logger.info('epoch[{}], iter[{}], loss:{}, '
                         'mean_fps:{:.2f}'
                         'imgs/sec'.format(real_epoch,
                                           me_step,
                                           cb_params.net_outputs,
                                           fps_mean))
        self.me_epoch_start_step_num = me_step
        self.me_epoch_start_time = time.time()

    def step_begin(self, run_context):
        pass

    def step_end(self, run_context, *me_args):
        pass

    def end(self, run_context):
        self.logger.info('end network train...')


def apply_eval(eval_param):
    """apply evaluation"""
    eval_model = eval_param["model"]
    eval_ds = eval_param["dataset"]
    metrics_name = eval_param["metrics_name"]
    res = eval_model.eval(eval_ds)
    return res[metrics_name]


def parse(arguments=None):
    """parameters"""
    parser = argparse.ArgumentParser('mindspore resnest training')
    # path arguments
    parser.add_argument('--outdir', type=str, default='output', help='logger output directory')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--resume', type=bool, default=False,
                        help='whether to resume the pretrained model')
    parser.add_argument('--pretrained_ckpt_path', type=str, default="/home/lidongsheng/ckpt/resnest-30_2502.ckpt",
                        help='put the path to resuming file if needed')
    # training parameters
    parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
    parser.add_argument('--run_eval', action='store_true', default=True,
                        help='evaluating')
    parser.add_argument('--save_ckpt', action='store_true', default=True,
                        help='ckpt')
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="Evaluation interval when run_eval is True, default is 10.")
    parser.add_argument("--eval_start_epoch", type=int, default=1,
                        help="Evaluation start epoch when run_eval is True, default is 120.")
    parser.add_argument("--device_num", type=int, default=1)
    # modelarts
    parser.add_argument('--is_model_arts', type=ast.literal_eval, default=False)
    parser.add_argument('--data_url', type=str)
    parser.add_argument('--train_url', type=str)

    arguments = parser.parse_args()

    return arguments

set_seed(1)


class LossCallBack(LossMonitor):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    """

    def __init__(self, has_trained_epoch=0):
        super(LossCallBack, self).__init__()
        self.has_trained_epoch = has_trained_epoch

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss1 = cb_params.net_outputs

        if isinstance(loss1, (tuple, list)):
            if isinstance(loss1[0], Tensor) and isinstance(loss1[0].asnumpy(), np.ndarray):
                loss1 = loss1[0]

        if isinstance(loss1, Tensor) and isinstance(loss1.asnumpy(), np.ndarray):
            loss1 = np.mean(loss1.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss1, float) and (np.isnan(loss1) or np.isinf(loss1)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num + int(self.has_trained_epoch),
                                                      cur_step_in_epoch, loss1), flush=True)
if __name__ == "__main__":
    print("================Start training================")

    args = parse()
    target = args.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=target)
    context.set_context(enable_graph_kernel=True)
    config.lr = config.lr*args.device_num
    if args.run_distribute:
        if target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            device_num = int(os.getenv('RANK_SIZE'))
            context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
            # init parallel training parameters
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
            config.rank = get_rank()
            config.group_size = get_group_size()
            print("rank {}".format(config.rank))
            print("group_size {}".format(config.group_size))
        else:
            init()
            config.rank = get_rank()
            config.group_size = get_group_size()
            context.set_auto_parallel_context(device_num=args.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    else:
        try:
            device_id = int(os.getenv('DEVICE_ID'))
            config.rank = 0
            config.group_size = 1
        except TypeError:
            device_id = 0
            config.rank = 0
            config.group_size = 1

    # dataset

    if args.is_model_arts:
        import moxing as mox
        train_dataset_path = '/cache/dataset/'
        config.root = train_dataset_path
        mox.file.copy_parallel(src_url=args.data_url, dst_url=train_dataset_path)
        args.outdir = '/cache/output/'
        dataset = ImageNet(train_dataset_path, mode="train",
                           img_size=config.base_size, crop_size=config.crop_size,
                           rank=config.rank, group_size=config.group_size, epoch=config.epochs,
                           batch_size=config.batch_size, num_parallel_workers=config.num_workers)
    else:
        dataset = ImageNet(config.root, mode="train",
                           img_size=config.base_size, crop_size=config.crop_size,
                           rank=config.rank, group_size=config.group_size, epoch=config.epochs,
                           batch_size=config.batch_size, num_parallel_workers=config.num_workers)
    config.steps_per_epoch = dataset.get_dataset_size()


    # net
    model_kwargs = {}
    if config.final_drop > 0.0:
        model_kwargs['final_drop'] = config.final_drop
    if config.last_gamma:
        model_kwargs['last_gamma'] = True

    # initialize weight
    if args.resume:
        if args.is_model_arts:
            pretrained_ckpt_path = "/cache/pretrained/resnest50-270_2502.ckpt"
            mox.file.copy_parallel(args.pretrained_ckpt_path, pretrained_ckpt_path)
        else:
            pretrained_ckpt_path = args.pretrained_ckpt_path
        net = get_network(config.net_name, args.resume, pretrained_ckpt_path, **model_kwargs)
    else:
        net = get_network(config.net_name, **model_kwargs)

    # initialize learning rate
    lr = get_lr(config)
    lr = Tensor(lr)

    # optimizer
    if config.disable_bn_wd:
        param_groups = get_param_groups(net)
    else:
        param_groups = net.trainable_params()
    optimizer = nn.Momentum(params=param_groups,
                            learning_rate=lr,
                            momentum=config.momentum,
                            weight_decay=config.weight_decay)

    # loss
    loss = CrossEntropy(smooth_factor=config.label_smoothing, num_classes=config.num_classes)
    if config.is_dynamic_loss_scale == 1:
        loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536,
                                                     scale_factor=2,
                                                     scale_window=2000)
    else:
        loss_scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    model = Model(net, loss_fn=loss, optimizer=optimizer, loss_scale_manager=loss_scale_manager,
                  metrics={'acc'}, amp_level="O2", keep_batchnorm_fp32=False)

    # checkpoint
    progress_cb = ProgressMonitor(args, config)
    callbacks = [progress_cb]

    time_cb = TimeMonitor(data_size=config.steps_per_epoch)
    callbacks.append(time_cb)

    if target == "GPU":
        loss_cb = LossCallBack(0)
        callbacks.append(loss_cb)

    # eval
    if args.run_eval:
        if args.device_target == "Ascend":
            if config.root is None or not os.path.isdir(config.root):
                raise ValueError("{} is not a existing path.".format(config.root))
            eval_dataset = ImageNet(config.root, mode="val",
                                    img_size=config.base_size, crop_size=config.crop_size,
                                    rank=config.rank, group_size=config.group_size, epoch=1,
                                    batch_size=config.batch_size, num_parallel_workers=config.num_workers)
            val_step_size = eval_dataset.get_dataset_size()
            eval_param_dict = {"model": model, "dataset": eval_dataset, "metrics_name": "acc"}
            eval_callback = EvalCallBack(apply_eval,
                                         eval_param_dict,
                                         interval=args.eval_interval,
                                         eval_start_epoch=args.eval_start_epoch,
                                         metrics_name="acc"
                                         )
            callbacks.append(eval_callback)

    if args.save_ckpt and config.rank == 0:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=config.steps_per_epoch,
                                       keep_checkpoint_max=3)
        save_ckpt_path = os.path.join(args.outdir, 'ckpt_' + str(config.rank) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix=config.net_name)
        callbacks.append(ckpt_cb)

    model.train(config.epochs, dataset, callbacks=callbacks, dataset_sink_mode=True)

    if args.is_model_arts:
        print("Copy output (ckpt and log info) from cloud server to local...")
        mox.file.copy_parallel(src_url=args.outdir, dst_url=args.train_url)
