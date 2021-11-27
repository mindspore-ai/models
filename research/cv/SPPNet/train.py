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
######################## train SPPNet example ########################
train SPPnet and get network model files(.ckpt) :
python train.py --train_path /YourDataPath --eval_path /YourValPath --device_id YourAscendId --train_model model
"""
import ast
import argparse
import os
import mindspore.nn as nn
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore import dataset as de
from mindspore import context
from mindspore import Tensor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.common import set_seed
from src.config import sppnet_mult_cfg, sppnet_single_cfg, zfnet_cfg
from src.dataset import create_dataset_imagenet
from src.generator_lr import warmup_cosine_annealing_lr
from src.sppnet import SppNet
from src.eval_callback import EvalCallBack, EvalCallBackMult


set_seed(44)
de.config.set_seed(44)
parser = argparse.ArgumentParser(description='MindSpore SPPNet')
parser.add_argument('--sink_size', type=int, default=-1, help='control the amount of data in each sink')
parser.add_argument('--train_model', type=str, default='sppnet_single', help='chose the training model',
                    choices=['sppnet_single', 'sppnet_mult', 'zfnet'])
parser.add_argument('--device_target', type=str, default="Ascend", help='chose the device for train',
                    choices=['Ascend', 'GPU'])
parser.add_argument('--train_path', type=str,
                    default="./imagenet_original/train",
                    help='path where the train dataset is saved')
parser.add_argument('--eval_path', type=str,
                    default="./imagenet_original/val",
                    help='path where the validate dataset is saved')
parser.add_argument('--is_distributed', type=int, default=0, help='distributed training')
parser.add_argument('--ckpt_path', type=str, default="./ckpt", help='if is test, must provide\
                        path where the trained ckpt file')
parser.add_argument('--dataset_sink_mode', type=ast.literal_eval,
                    default=True, help='dataset_sink_mode is False or True')
parser.add_argument('--device_id', type=int, default=0, help='device id of Ascend. (Default: 0)')
parser.add_argument('--device_num', type=int, default=1)
args = parser.parse_args()


def apply_eval(eval_param):
    """construct eval function"""
    eval_model = eval_param["model"]
    eval_ds = eval_param["dataset"]
    res = eval_model.eval(eval_ds)
    return res


if __name__ == "__main__":

    device_num = args.device_num
    device_target = args.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    context.set_context(save_graphs=False)

    if args.is_distributed:
        if args.device_target == "Ascend":
            context.set_context(device_id=device_id)
            init("hccl")
        else:
            assert args.device_target == "GPU"
            init("nccl")
        args.device_id = get_rank()
        args.device_num = get_group_size()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=args.device_num, parallel_mode=ParallelMode.DATA_PARALLEL)
    else:
        context.set_context(device_id=args.device_id)

    if args.train_model == "zfnet":
        cfg = zfnet_cfg
        ds_train = create_dataset_imagenet(args.train_path, 'zfnet', cfg.batch_size)
        network = SppNet(cfg.num_classes, phase='train', train_model=args.train_model)
        prefix = "checkpoint_zfnet"
    elif args.train_model == "sppnet_single":
        cfg = sppnet_single_cfg
        ds_train = create_dataset_imagenet(args.train_path, cfg.batch_size)
        network = SppNet(cfg.num_classes, phase='train', train_model=args.train_model)
        prefix = "checkpoint_sppnet"
    else:
        cfg = sppnet_mult_cfg
        ds_train = create_dataset_imagenet(args.train_path, 'sppnet_mult', cfg.batch_size)
        network = SppNet(cfg.num_classes, phase='train', train_model=args.train_model)
        prefix = "checkpoint_sppnet"

    if ds_train.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    loss_scale_manager = None
    metrics = {'top_1_accuracy', 'top_5_accuracy'}
    step_per_epoch = ds_train.get_dataset_size() if args.sink_size == -1 else args.sink_size

    # loss function
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # learning rate generator
    lr = Tensor(warmup_cosine_annealing_lr(lr=cfg.lr_init, steps_per_epoch=step_per_epoch,
                                           warmup_epochs=cfg.warmup_epochs, max_epoch=cfg.epoch_size,
                                           iteration_max=cfg.iteration_max, lr_min=cfg.lr_min))

    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith('.bias'):
            no_decay_params.append(x)
        elif parameter_name.endswith('.gamma'):
            no_decay_params.append(x)
        elif parameter_name.endswith('.beta'):
            no_decay_params.append(x)
        else:
            decay_params.append(x)

    params = [{'params': no_decay_params, 'weight_decay': 0.0, "lr": lr}, {'params': decay_params, "lr": lr}]

    # Optimizer
    opt = nn.Momentum(params=params,
                      learning_rate=lr,
                      momentum=cfg.momentum,
                      weight_decay=cfg.weight_decay,
                      loss_scale=cfg.loss_scale)

    if cfg.is_dynamic_loss_scale == 1:
        loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
    else:
        loss_scale_manager = FixedLossScaleManager(cfg.loss_scale, drop_overflow_update=False)

    model = Model(network, loss_fn=loss, optimizer=opt, metrics=metrics, amp_level="O2", keep_batchnorm_fp32=False,
                  loss_scale_manager=loss_scale_manager)

    if device_num > 1:
        ckpt_save_dir = os.path.join(args.ckpt_path + "_" + str(get_rank()))
    else:
        ckpt_save_dir = os.path.join(args.ckpt_path)

    # callback
    eval_dataset = create_dataset_imagenet(args.eval_path, cfg.batch_size, training=False)
    evalParamDict = {"model": model, "dataset": eval_dataset}
    if args.train_model == "sppnet_mult":
        eval_cb = EvalCallBackMult(apply_eval, evalParamDict, eval_start_epoch=1, ckpt_directory=ckpt_save_dir)
    else:
        eval_cb = EvalCallBack(apply_eval, evalParamDict, eval_start_epoch=1, train_model_name=args.train_model,
                               ckpt_directory=ckpt_save_dir)
    loss_cb = LossMonitor(per_print_times=step_per_epoch)
    time_cb = TimeMonitor(data_size=step_per_epoch)

    print("============== Starting Training ==============")

    if args.train_model == "sppnet_mult":
        ds_train_180 = create_dataset_imagenet(args.train_path, 'sppnet_mult', cfg.batch_size,
                                               training=True, image_size=180)
        for per_epoch in range(cfg.epoch_size):
            print("================ Epoch:{} ==================".format(per_epoch+1))
            if per_epoch % 2 == 0:
                cb = [time_cb, loss_cb, eval_cb]
                model.train(1, ds_train, callbacks=cb, dataset_sink_mode=False, sink_size=args.sink_size)
            else:
                cb = [time_cb, loss_cb]
                model.train(1, ds_train_180, callbacks=cb, dataset_sink_mode=False, sink_size=args.sink_size)
    else:
        cb = [time_cb, loss_cb, eval_cb]
        model.train(cfg.epoch_size, ds_train, callbacks=cb, dataset_sink_mode=True, sink_size=args.sink_size)
