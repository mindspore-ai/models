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
import ast
import argparse
import numpy as np
import moxing as mox
import mindspore.nn as nn
import mindspore as ms
from mindspore.communication.management import init, get_group_size
from mindspore import dataset as de
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.common import set_seed
from mindspore import context, Tensor, load_checkpoint, load_param_into_net, export, save_checkpoint
from src.dataset import create_dataset_imagenet
from src.generator_lr import warmup_cosine_annealing_lr
from src.config import sppnet_mult_cfg, sppnet_single_cfg, zfnet_cfg
from src.sppnet import SppNet

set_seed(44)
de.config.set_seed(44)
parser = argparse.ArgumentParser(description='MindSpore SPPNet')
parser.add_argument('--sink_size', type=int, default=-1, help='control the amount of data in each sink')
parser.add_argument('--train_model', type=str, default='sppnet_single', help='chose the training model',
                    choices=['sppnet_single', 'sppnet_mult', 'zfnet'])
parser.add_argument('--device_target', type=str, default="Ascend",
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--train_path', type=str,
                    default="./imagenet_original/train",
                    help='path where the train dataset is saved')
parser.add_argument('--ckpt_path', type=str, default="./ckpt", help='if is test, must provide\
                        path where the trained ckpt file')
parser.add_argument('--dataset_sink_mode', type=ast.literal_eval,
                    default=True, help='dataset_sink_mode is False or True')
parser.add_argument('--device_id', type=int, default=0, help='device id of Ascend. (Default: 0)')
parser.add_argument('--device_num', type=int, default=1)
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epoch_size', type=int, default=16)
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
args = parser.parse_args()


def apply_eval(eval_param):
    """construct eval function"""
    eval_model = eval_param["model"]
    eval_ds = eval_param["dataset"]
    res = eval_model.eval(eval_ds)
    return res


def run_export(train_model, ckpt_path):
    if train_model == "zfnet":
        cfg = zfnet_cfg
        network = SppNet(args.num_classes, train_model=train_model)

    elif train_model == "sppnet_single":
        cfg = sppnet_single_cfg
        network = SppNet(args.num_classes, train_model=train_model)

    else:
        cfg = sppnet_mult_cfg
        network = SppNet(args.num_classes, train_model=train_model)

    ckpt_file = "best.ckpt"
    param_dict = load_checkpoint(ckpt_file)
    load_param_into_net(network, param_dict)
    input_arr = Tensor(np.zeros([args.batch_size, 3, cfg.image_height, cfg.image_width]), ms.float32)
    export(network, input_arr, file_name="/cache/out/"+train_model, file_format=args.file_format)
    mox.file.copy_parallel("/cache/out/", ckpt_path)

def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break

def get_no_decay_params(network):
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
    return  decay_params, no_decay_params

def get_network():
    if args.train_model == "zfnet":
        cfg = zfnet_cfg
        ds_train = create_dataset_imagenet(args.train_path, 'zfnet', args.batch_size)
        network = SppNet(args.num_classes, phase='train', train_model=args.train_model)
    elif args.train_model == "sppnet_single":
        cfg = sppnet_single_cfg
        ds_train = create_dataset_imagenet(args.train_path, args.batch_size)
        network = SppNet(args.num_classes, phase='train', train_model=args.train_model)
    else:
        cfg = sppnet_mult_cfg
        ds_train = create_dataset_imagenet(args.train_path, 'sppnet_mult', args.batch_size)
        network = SppNet(args.num_classes, phase='train', train_model=args.train_model)

    if ds_train.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    return network, ds_train, cfg

def run_train():
    device_num = args.device_num
    device_target = args.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    context.set_context(save_graphs=False)

    if device_target == "Ascend":
        context.set_context(device_id=args.device_id)

        if device_num > 1:
            init()
            device_num = get_group_size()
            print("device_num:", device_num)
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              global_rank=args.device_id,
                                              gradients_mean=True)
    else:
        raise ValueError("Unsupported platform.")

    network, ds_train, cfg = get_network()

    #trans_learn
    if args.pretrained is not None:
        param_dict = load_checkpoint(args.pretrained)
        filter_list = ['fc3.weight', 'fc3.bias']
        filter_checkpoint_parameter_by_list(param_dict, filter_list)
        load_param_into_net(network, param_dict)


    metrics = {'top_1_accuracy', 'top_5_accuracy'}
    step_per_epoch = ds_train.get_dataset_size() if args.sink_size == -1 else args.sink_size

    # loss function
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # learning rate generator
    lr = Tensor(warmup_cosine_annealing_lr(lr=cfg.lr_init, steps_per_epoch=step_per_epoch,
                                           warmup_epochs=cfg.warmup_epochs, max_epoch=cfg.epoch_size,
                                           iteration_max=cfg.iteration_max, lr_min=cfg.lr_min))

    decay_params, no_decay_params = get_no_decay_params(network)

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

    if device_target == "Ascend":
        model = Model(network, loss_fn=loss, optimizer=opt, metrics=metrics, amp_level="O2", keep_batchnorm_fp32=False,
                      loss_scale_manager=loss_scale_manager)
    else:
        raise ValueError("Unsupported platform.")

    # callback
    loss_cb = LossMonitor(per_print_times=step_per_epoch)
    time_cb = TimeMonitor(data_size=step_per_epoch)
    cb = [time_cb, loss_cb]
    print("============== Starting Training ==============")

    if args.train_model == "sppnet_mult":
        ds_train_mult_size = create_dataset_imagenet(args.train_path, 'sppnet_mult', args.batch_size,
                                                     training=True, image_size=180)

        for per_epoch in range(args.epoch_size):
            print("================ Epoch:{} ==================".format(per_epoch + 1))
            model.train(1, ds_train_mult_size, callbacks=cb, dataset_sink_mode=False, sink_size=args.sink_size)
    else:
        for per_epoch in range(args.epoch_size):
            print("================ Epoch:{} ==================".format(per_epoch + 1))
            model.train(1, ds_train, callbacks=cb, dataset_sink_mode=True, sink_size=args.sink_size)

    save_checkpoint(network, 'best.ckpt')

    mox.file.copy_parallel("best.ckpt", args.ckpt_path+"/best.ckpt")

if __name__ == '__main__':
    run_train()
    print("============== Starting Exporting ==============")
    run_export(args.train_model, args.ckpt_path)
