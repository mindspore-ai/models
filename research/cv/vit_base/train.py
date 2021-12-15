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
"""
Train.
"""
import argparse
import math
import os

import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore import nn
from mindspore.common import set_seed
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.train.callback import Callback
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import LossMonitor
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net
from mindspore.train.serialization import save_checkpoint

import src.net_config as configs
from src.config import cifar10_cfg
from src.dataset import create_dataset_cifar10
from src.modeling_ms import VisionTransformer
from src.npz_converter import npz2ckpt

set_seed(2)


def lr_steps_imagenet(_cfg, steps_per_epoch):
    """init lr step"""
    if _cfg.lr_scheduler == 'cosine_annealing':
        _lr = warmup_cosine_annealing_lr(_cfg.lr_init,
                                         steps_per_epoch,
                                         _cfg.warmup_epochs,
                                         _cfg.epoch_size,
                                         _cfg.T_max,
                                         _cfg.eta_min)
    else:
        raise NotImplementedError(_cfg.lr_scheduler)

    return _lr


def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    """warmup lr at the end of training"""
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr1 = float(init_lr) + lr_inc * current_step
    return lr1


def warmup_cosine_annealing_lr(lr5, steps_per_epoch, warmup_epochs, max_epoch, T_max, eta_min=0):
    """warmup cosine annealing lr"""
    base_lr = lr5
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    lr_each_step = []
    for i in range(total_steps):
        last_epoch = i // steps_per_epoch
        if i < warmup_steps:
            lr5 = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr5 = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * last_epoch / T_max)) / 2
        lr_each_step.append(lr5)

    return np.array(lr_each_step).astype(np.float32)


class EvalCallBack(Callback):
    """evaluation callback function"""
    def __init__(self, model0, eval_dataset, eval_per_epoch, epoch_per_eval0):
        self.model = model0
        self.eval_dataset = eval_dataset
        self.eval_per_epoch = eval_per_epoch
        self.epoch_per_eval = epoch_per_eval0
        self.best_acc = 0
        self.best_result = 0

    def best_ckpt(self, epoch, accuracy, network):
        """save best checkpoint after eval during training"""
        if epoch // self.eval_per_epoch == 1:
            self.best_acc = accuracy
        elif epoch // self.eval_per_epoch > 1 and self.best_acc <= accuracy:
            self.best_acc = accuracy
            save_checkpoint(
                network,
                f"./{ckpt_save_dir}/best_acc.ckpt",
            )

    def epoch_end(self, run_context):
        """set evaluation at the end of each eval_per_epoch"""
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epoch == 0:
            acc = self.model.eval(self.eval_dataset)
            self.best_ckpt(cur_epoch, acc["acc"], cb_param.network)
            self.epoch_per_eval["epoch"].append(cur_epoch)
            self.epoch_per_eval["acc"].append(acc)
            print(acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10'],
                        help='dataset name.')
    parser.add_argument('--sub_type', type=str, default='ViT-B_16',
                        choices=['ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32', 'ViT-H_14', 'testing'])
    parser.add_argument('--device_target', type=str, default=None, help='device GPU or Ascend. (Default: None)')
    parser.add_argument('--device_id', type=int, default=None, help='device id of GPU or Ascend. (Default: None)')
    parser.add_argument('--device_start', type=int, default=0, help='start device id. (Default: 0)')
    parser.add_argument('--lr_init', type=float, default=None, help='start lr. (Default: None)')
    parser.add_argument('--data_url', type=str, default=None, help='location of data.')
    parser.add_argument('--ckpt_url', type=str, default=None, help='location of ckpt.')
    parser.add_argument('--modelarts', type=bool, default=False, help='use ModelArts or not.')
    parser.add_argument('--logs_dir', type=str, default='./ckpt/', help='dir to save logs and ckpt. (Default: ./ckpt/)')
    parser.add_argument('--do_val', type=bool, default=False, help='do evaluation. (Default: False)')
    args_opt = parser.parse_args()

    if args_opt.modelarts:
        import moxing as mox
        local_data_path = '/cache/data'
        local_ckpt_path = '/cache/data/pre_ckpt'

    if args_opt.dataset_name == 'cifar10':
        cfg = cifar10_cfg
    else:
        raise ValueError("Unsupported dataset.")

    # set context
    if args_opt.device_target is None:
        device_target = cfg.device_target
    else:
        device_target = args_opt.device_target

    if args_opt.lr_init is not None:
        cfg.lr_init = args_opt.lr_init

    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
    device_num = int(os.getenv('RANK_SIZE', '1'))

    if device_target == 'Ascend':
        amp = 'O3'
        device_id = int(os.getenv('DEVICE_ID', '0'))
        if args_opt.device_id is None:
            context.set_context(device_id=cfg.device_id)
        else:
            context.set_context(device_id=args_opt.device_id)

        if device_num > 1:
            if args_opt.modelarts:
                context.set_context(device_id=int(os.getenv('DEVICE_ID')))
            init(backend_name='hccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(
                device_num=device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
            )
            if args_opt.modelarts:
                local_data_path = os.path.join(local_data_path, str(device_id))
    elif device_target == 'GPU':
        amp = 'O0'
        if device_num > 1:
            init(backend_name='nccl')
            device_num = get_group_size()
            device_id = get_rank()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(
                device_num=device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
            )
        else:
            device_num = 1
            device_id = args_opt.device_id
            context.set_context(device_id=device_id)
    else:
        raise ValueError("Unsupported platform.")

    if args_opt.modelarts:
        mox.file.copy_parallel(src_url=args_opt.data_url, dst_url=local_data_path)

    if args_opt.dataset_name == "cifar10":
        if args_opt.modelarts:
            data_home = local_data_path
        else:
            data_home = cfg.data_path

        dataset = create_dataset_cifar10(
            device=device_target,
            data_home=data_home,
            repeat_num=1,
            device_num=device_num,
            training=True,
        )
    else:
        raise ValueError("Unsupported dataset.")

    batch_num = dataset.get_dataset_size()

    CONFIGS = {'ViT-B_16': configs.get_b16_config,
               'ViT-B_32': configs.get_b32_config,
               'ViT-L_16': configs.get_l16_config,
               'ViT-L_32': configs.get_l32_config,
               'ViT-H_14': configs.get_h14_config,
               'R50-ViT-B_16': configs.get_r50_b16_config,
               'testing': configs.get_testing}

    net = VisionTransformer(CONFIGS[args_opt.sub_type], num_classes=cfg.num_classes)

    if args_opt.modelarts:
        mox.file.copy_parallel(src_url=args_opt.ckpt_url, dst_url=local_ckpt_path)

    if cfg.pre_trained:
        if args_opt.modelarts:
            param_dict = load_checkpoint(os.path.join(
                local_ckpt_path,
                "cifar10_pre_checkpoint_based_imagenet21k.ckpt",
            ))
        else:
            if cfg.checkpoint_path.endswith(".ckpt"):
                param_dict = load_checkpoint(cfg.checkpoint_path)
            elif cfg.checkpoint_path.endswith(".npz"):
                if args_opt.sub_type == 'ViT-B_16':
                    param_dict = npz2ckpt(cfg.checkpoint_path)
                else:
                    raise NotImplementedError('Unsupported model type for convert.')
            else:
                raise ValueError("Unsupported checkpoint format.")

        load_param_into_net(net, param_dict)
        print("Load pre_trained ckpt: {}".format(cfg.checkpoint_path))

    loss_scale_manager = None

    lr = lr_steps_imagenet(cfg, batch_num)

    opt = nn.Momentum(
        params=net.trainable_params(),
        learning_rate=Tensor(lr),
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    model = Model(
        net,
        loss_fn=loss,
        optimizer=opt,
        metrics={'acc'},
        amp_level=amp,
        keep_batchnorm_fp32=False,
        loss_scale_manager=loss_scale_manager,
    )

    config_ck = CheckpointConfig(
        save_checkpoint_steps=batch_num,
        keep_checkpoint_max=cfg.keep_checkpoint_max,
    )
    time_cb = TimeMonitor(data_size=batch_num)

    ckpt_save_dir = args_opt.logs_dir

    ckpoint_cb = ModelCheckpoint(
        prefix="train_vit_" + args_opt.dataset_name,
        directory=ckpt_save_dir,
        config=config_ck,
    )
    loss_cb = LossMonitor(per_print_times=batch_num)

    if args_opt.modelarts:
        cbs = [time_cb, ModelCheckpoint(prefix="train_vit_" + args_opt.dataset_name, config=config_ck), loss_cb]
    else:
        epoch_per_eval = {"epoch": [], "acc": []}
        eval_cb = EvalCallBack(
            model,
            create_dataset_cifar10(
                device=device_target,
                data_home=cfg.val_data_path,
                repeat_num=1,
                device_num=1,
                training=False,
            ),
            eval_per_epoch=2,
            epoch_per_eval0=epoch_per_eval,
        )
        if args_opt.do_val:
            cbs = [time_cb, ckpoint_cb, loss_cb, eval_cb]
        else:
            cbs = [time_cb, ckpoint_cb, loss_cb]

        if device_num > 1 and device_id != args_opt.device_start:  # Set card for eval if distribute
            if args_opt.do_val:
                cbs = [loss_cb]
            else:
                cbs = [time_cb, loss_cb]
    model.train(cfg.epoch_size, dataset, callbacks=cbs)
    print("train success")
