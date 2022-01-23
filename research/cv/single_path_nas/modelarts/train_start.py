# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
#################train spnasnet example########################
python train.py
"""
import argparse
import os
import glob
import ast
import numpy as np
from mindspore import load_checkpoint, load_param_into_net, export
from mindspore import Tensor
from mindspore import context
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
from mindspore.train.model import Model
import moxing as mox
from easydict import EasyDict as edict
from src import spnasnet
from src.CrossEntropySmooth import CrossEntropySmooth
from src.dataset import create_dataset_imagenet

set_seed(1)

imagenet_cfg = edict({
    'name': 'imagenet',
    'lr_init': 1.5,  # 1p:0.26  8p:1.5
    'batch_size': 128,
    'momentum': 0.9,
    'weight_decay': 1e-5,
    'image_height': 224,
    'image_width': 224,
    'data_path': '/data/ILSVRC2012_train/',
    'val_data_path': '/data/ILSVRC2012_val/',
    'device_target': 'Ascend',
    'device_id': 0,
    'keep_checkpoint_max': 40,
    'checkpoint_path': None,
    'onnx_filename': 'single-path-nas',
    'air_filename': 'single-path-nas',

    # optimizer and lr related
    'lr_scheduler': 'cosine_annealing',
    'lr_epochs': [30, 60, 90],
    'lr_gamma': 0.3,
    'eta_min': 0.0,
    'T_max': 150,
    'warmup_epochs': 0,

    # loss related
    'is_dynamic_loss_scale': 1,
    'loss_scale': 1024,
    'label_smooth_factor': 0.1,
    'use_label_smooth': True,
})

def model_export(arguments):
    """export air"""
    output_dir = arguments.local_output_dir
    epoch_size = str(args.epoch_size)
    ckpt_file = glob.glob(output_dir + '/' + '*' + epoch_size + '*' + '.ckpt')[0]
    print("ckpt_file: ", ckpt_file)
    network = spnasnet.spnasnet(num_classes=args.num_classes)

    param_dic = load_checkpoint(ckpt_file)
    load_param_into_net(network, param_dic)

    input_arr = Tensor(np.zeros([args.batch_size, 3, args.height, args.width], np.float32))
    export_file = os.path.join(output_dir, args.file_name)
    export(network, input_arr, file_name=export_file, file_format=args.file_format)
    return 0

def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    """remove useless parameters according to filter_list"""
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break

def lr_steps_imagenet(_cfg, steps_per_epoch):
    """lr step for imagenet"""
    from src.lr_scheduler.warmup_step_lr import warmup_step_lr
    from src.lr_scheduler.warmup_cosine_annealing_lr import warmup_cosine_annealing_lr
    if _cfg.lr_scheduler == 'exponential':
        _lr = warmup_step_lr(_cfg.lr_init,
                             _cfg.lr_epochs,
                             steps_per_epoch,
                             _cfg.warmup_epochs,
                             args.epoch_size,
                             gamma=_cfg.lr_gamma,
                             )
    elif _cfg.lr_scheduler == 'cosine_annealing':
        _lr = warmup_cosine_annealing_lr(_cfg.lr_init,
                                         steps_per_epoch,
                                         _cfg.warmup_epochs,
                                         args.epoch_size,
                                         _cfg.T_max,
                                         _cfg.eta_min)
    else:
        raise NotImplementedError(_cfg.lr_scheduler)

    return _lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single-Path-NAS Training')
    parser.add_argument('--dataset_name', type=str, default='imagenet', choices=['imagenet',],
                        help='dataset name.')
    parser.add_argument('--train_url', required=True, default=None, help='obs browser path')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data')
    parser.add_argument('--filter_prefix', type=str, default='huawei', help='filter_prefix name.')
    parser.add_argument('--device_id', type=int, default=None, help='device id of Ascend. (Default: None)')
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--file_name", type=str, default="single-path-nas", help="output file name.")
    parser.add_argument('--width', type=int, default=224, help='input width')
    parser.add_argument('--height', type=int, default=224, help='input height')
    parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR",
                        help="file format")
    parser.add_argument('--local_data_dir', type=str, default="/cache")
    parser.add_argument('--local_output_dir', type=str, default="/cache/train_output")
    parser.add_argument("--filter_weight", type=ast.literal_eval, default=False,
                        help="Filter head weight parameters, default is False.")
    parser.add_argument('--load_path', type=str, default='/cache/data/pretrained_dir',
                        help='The location of input data')
    parser.add_argument('--pretrained_ckpt', type=str, default='',
                        help='pretrained checkpoint file name')
    parser.add_argument('--pre_trained', type=ast.literal_eval, default=False, help='need pre_trained')
    parser.add_argument('--num_classes', type=int, default=1000, help='num_classes')
    parser.add_argument('--epoch_size', type=int, default=180, help='epoch_size')
    args = parser.parse_args()

    local_data_path = args.local_data_dir
    train_output_path = args.local_output_dir

    mox.file.copy_parallel(args.data_url, local_data_path)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~file copy success~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    print(args.pre_trained)
    print(args.num_classes)
    print(args.epoch_size)

    if args.dataset_name == "imagenet":
        cfg = imagenet_cfg
    else:
        raise ValueError("Unsupported dataset.")

    # set context
    device_target = cfg.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target, enable_graph_kernel=True)

    device_num = int(os.environ.get("DEVICE_NUM", 1))

    rank = 0
    if device_target == "Ascend":
        if args.device_id is not None:
            context.set_context(device_id=args.device_id)
        else:
            context.set_context(device_id=cfg.device_id)

        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
            rank = get_rank()
    else:
        raise ValueError("Unsupported platform.")

    if args.dataset_name == "imagenet":
        dataset = create_dataset_imagenet(args.data_url, 1)
    else:
        raise ValueError("Unsupported dataset.")

    batch_num = dataset.get_dataset_size()

    net = spnasnet.get_spnasnet(num_classes=args.num_classes)
    net.update_parameters_name(args.filter_prefix)


    def get_param_groups(network):
        """ get param groups """
        decay_params = []
        no_decay_params = []
        for x in network.trainable_params():
            parameter_name = x.name
            if parameter_name.endswith('.bias'):
                # all bias not using weight decay
                no_decay_params.append(x)
            elif parameter_name.endswith('.gamma'):
                # bn weight bias not using weight decay, be carefully for now x not include BN
                no_decay_params.append(x)
            elif parameter_name.endswith('.beta'):
                # bn weight bias not using weight decay, be carefully for now x not include BN
                no_decay_params.append(x)
            else:
                decay_params.append(x)

        return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]

    # load checkpoint
    if args.pre_trained:
        ckpt__file = glob.glob(args.local_data_dir + '/' + '*.ckpt')[0]
        if os.path.isfile(ckpt__file):
            print("ckpt_file: ", ckpt__file)
            param_dict = load_checkpoint(ckpt__file)
            if args.filter_weight:
                filter_list = [y.name for y in net.output.get_parameters()]
                filter_checkpoint_parameter_by_list(param_dict, filter_list)
            load_param_into_net(net, param_dict)


    loss_scale_manager = None
    if args.dataset_name == 'imagenet':
        lr = lr_steps_imagenet(cfg, batch_num)

        if cfg.is_dynamic_loss_scale:
            cfg.loss_scale = 1

        opt = Momentum(params=net.get_parameters(),
                       learning_rate=Tensor(lr),
                       momentum=cfg.momentum,
                       weight_decay=cfg.weight_decay,
                       loss_scale=cfg.loss_scale)

        if not cfg.use_label_smooth:
            cfg.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=cfg.label_smooth_factor, num_classes=args.num_classes)

        if cfg.is_dynamic_loss_scale == 1:
            loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
        else:
            loss_scale_manager = FixedLossScaleManager(cfg.loss_scale, drop_overflow_update=False)

    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'top_1_accuracy', 'top_5_accuracy', 'loss'},
                  loss_scale_manager=loss_scale_manager)

    config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * 1, keep_checkpoint_max=cfg.keep_checkpoint_max)
    time_cb = TimeMonitor(data_size=batch_num)
    ckpt_save_dir = "./ckpt_" + str(rank) + "/"
    ckpoint_cb = ModelCheckpoint(prefix="train_spnasnet_" + args.dataset_name, directory=ckpt_save_dir,
                                 config=config_ck)
    loss_cb = LossMonitor()

    model.train(args.epoch_size, dataset, callbacks=[time_cb, ckpoint_cb, loss_cb])

    print("train success")
    if not os.path.exists(args.local_output_dir):
        os.mkdir(args.local_output_dir)
    args.local_output_dir = ckpt_save_dir
    model_export(args)
    print(os.listdir(args.local_output_dir))
    print(os.listdir(train_output_path))
    mox.file.copy_parallel(args.local_output_dir, args.train_url)
