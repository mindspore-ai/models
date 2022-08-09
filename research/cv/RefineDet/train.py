# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""Train RefineDet and get checkpoint files."""

import argparse
import ast
import os
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.communication.management import init, get_rank
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed, dtype
from src.config import get_config
from src.dataset import create_refinedet_dataset, create_mindrecord
from src.lr_schedule import get_lr
from src.init_params import init_net_param, filter_checkpoint_parameter_by_list
from src.refinedet import refinedet_vgg16, refinedet_resnet101
from src.refinedet_loss_cell import RefineDetLossCell, TrainingWrapper

set_seed(1)

def get_args():
    """get args for train"""
    parser = argparse.ArgumentParser(description="RefineDet training script")
    parser.add_argument("--using_mode", type=str, default="refinedet_vgg16_320",
                        choices=("refinedet_vgg16_320", "refinedet_vgg16_512",
                                 "refinedet_resnet101_320", "refinedet_resnet101_512"),
                        help="which network you want to train, we present four networks: "
                             "using vgg16 as backbone with 320x320 image size"
                             "using vgg16 as backbone with 512x512 image size"
                             "using resnet101 as backbone with 320x320 image size"
                             "using resnet101 as backbone with 512x512 image size")
    parser.add_argument("--run_online", type=ast.literal_eval, default=False,
                        help="Run on Modelarts platform, need data_url, train_url if true, default is False.")
    parser.add_argument("--data_url", type=str,
                        help="using for OBS file system")
    parser.add_argument("--train_url", type=str,
                        help="using for OBS file system")
    parser.add_argument("--pre_trained_url", type=str, default=None, help="Pretrained Checkpoint file url for OBS.")
    parser.add_argument("--run_platform", type=str, default="Ascend", choices=("Ascend", "GPU", "CPU"),
                        help="run platform, support Ascend, GPU and CPU.")
    parser.add_argument("--only_create_dataset", type=ast.literal_eval, default=False,
                        help="If set it true, only create Mindrecord, default is False.")
    parser.add_argument("--distribute", type=ast.literal_eval, default=False,
                        help="Run distribute, default is False.")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate, default is 0.05.")
    parser.add_argument("--mode", type=str, default="sink", help="Run sink mode or not, default is sink.")
    parser.add_argument("--dataset", type=str, default="coco",
                        help="Dataset, default is coco."
                             "Now we have coco, voc0712, voc0712plus")
    parser.add_argument("--epoch_size", type=int, default=500, help="Epoch size, default is 500.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default is 32.")
    parser.add_argument("--pre_trained", type=str, default=None, help="Pretrained Checkpoint file path.")
    parser.add_argument("--pre_trained_epoch_size", type=int, default=0, help="Pretrained epoch size.")
    parser.add_argument("--save_checkpoint_epochs", type=int, default=10, help="Save checkpoint epochs, default is 10.")
    parser.add_argument("--loss_scale", type=int, default=1024, help="Loss scale, default is 1024.")
    parser.add_argument("--filter_weight", type=ast.literal_eval, default=False,
                        help="Filter head weight parameters, default is False.")
    parser.add_argument('--freeze_layer', type=str, default="none", choices=["none", "backbone"],
                        help="freeze the weights of network, support freeze the backbone's weights, "
                             "default is not freezing.")
    parser.add_argument('--debug', type=str, default="0", choices=["0", "1", "2", "3"],
                        help="Active the debug mode. 0 for no debug mode,"
                             "Under debug mode 1, the network would be run in PyNative mode,"
                             "Under debug mode 2, all ascend log would be print on stdout,"
                             "Under debug mode 3, all ascend log would be print on stdout."
                             "And network will run in PyNative mode.")
    parser.add_argument("--check_point", type=str, default="./ckpt",
                        help="The directory path to save check point files")
    args_opt = parser.parse_args()
    return args_opt

def refinedet_model_build(config, args_opt):
    """build refinedet network"""
    if config.model == "refinedet_vgg16":
        refinedet = refinedet_vgg16(config=config)
        init_net_param(refinedet)
    elif config.model == "refinedet_resnet101":
        refinedet = refinedet_resnet101(config=config)
        init_net_param(refinedet)
    else:
        raise ValueError(f'config.model: {config.model} is not supported')
    return refinedet

def train_main(args_opt):
    """main code for train refinedet"""
    rank = 0
    device_num = 1
    # config with args
    config = get_config(args_opt.using_mode, args_opt.dataset)

    # run mode config
    if args_opt.debug == "1" or args_opt.debug == "3":
        network_mode = context.PYNATIVE_MODE
    else:
        network_mode = context.GRAPH_MODE

    # set run platform
    if args_opt.run_platform == "CPU":
        context.set_context(mode=network_mode, device_target="CPU")
    else:
        context.set_context(mode=network_mode, device_target=args_opt.run_platform, device_id=args_opt.device_id)
        if args_opt.distribute:
            device_num = args_opt.device_num
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                              device_num=device_num)
            init()
            rank = get_rank()

    mindrecord_file = create_mindrecord(config, args_opt.dataset, "refinedet.mindrecord", True)

    if args_opt.only_create_dataset:
        return

    loss_scale = float(args_opt.loss_scale)
    if args_opt.run_platform == "CPU":
        loss_scale = 1.0

    # When create MindDataset, using the fitst mindrecord file, such as
    # refinedet.mindrecord0.
    use_multiprocessing = (args_opt.run_platform != "CPU")
    dataset = create_refinedet_dataset(config, mindrecord_file, repeat_num=1, batch_size=args_opt.batch_size,
                                       device_num=device_num, rank=rank, use_multiprocessing=use_multiprocessing)

    dataset_size = dataset.get_dataset_size()
    print(f"Create dataset done! dataset size is {dataset_size}")
    refinedet = refinedet_model_build(config, args_opt)
    if ("use_float16" in config and config.use_float16):
        refinedet.to_float(dtype.float16)
    net = RefineDetLossCell(refinedet, config)

    # checkpoint
    ckpt_config = CheckpointConfig(save_checkpoint_steps=dataset_size * args_opt.save_checkpoint_epochs)
    ckpt_prefix = args_opt.check_point + '/ckpt_'
    save_ckpt_path = ckpt_prefix + str(rank) + '/'
    ckpoint_cb = ModelCheckpoint(prefix="refinedet", directory=save_ckpt_path, config=ckpt_config)

    if args_opt.pre_trained:
        param_dict = load_checkpoint(args_opt.pre_trained)
        if args_opt.filter_weight:
            filter_checkpoint_parameter_by_list(param_dict, config.checkpoint_filter_list)
        load_param_into_net(net, param_dict, True)

    lr = Tensor(get_lr(global_step=args_opt.pre_trained_epoch_size * dataset_size,
                       lr_init=config.lr_init, lr_end=config.lr_end_rate * args_opt.lr, lr_max=args_opt.lr,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=args_opt.epoch_size,
                       steps_per_epoch=dataset_size))

    if "use_global_norm" in config and config.use_global_norm:
        opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr,
                          config.momentum, config.weight_decay, 1.0)
        net = TrainingWrapper(net, opt, loss_scale, True)
    else:
        opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr,
                          config.momentum, config.weight_decay, loss_scale)
        net = TrainingWrapper(net, opt, loss_scale)


    callback = [TimeMonitor(data_size=dataset_size), LossMonitor(), ckpoint_cb]
    model = Model(net)
    dataset_sink_mode = False
    if args_opt.mode == "sink" and args_opt.run_platform != "CPU":
        print("In sink mode, one epoch return a loss.")
        dataset_sink_mode = True
    print("Start train RefineDet, the first epoch will be slower because of the graph compilation.")
    model.train(args_opt.epoch_size, dataset, callbacks=callback, dataset_sink_mode=dataset_sink_mode)

def main():
    args_opt = get_args()
    # copy files if online
    if args_opt.run_online:
        import moxing as mox
        args_opt.device_id = int(os.getenv('DEVICE_ID'))
        args_opt.device_num = int(os.getenv('RANK_SIZE'))
        dir_root = os.getcwd()
        data_root = os.path.join(dir_root, "data")
        ckpt_root = os.path.join(dir_root, args_opt.check_point)
        mox.file.copy_parallel(args_opt.data_url, data_root)
        if args_opt.pre_trained:
            mox.file.copy_parallel(args_opt.pre_trained_url, args_opt.pre_trained)
    # print log to stdout
    if args_opt.debug == "2" or args_opt.debug == "3":
        os.environ["SLOG_PRINT_TO_STDOUT"] = "1"
        os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"
        os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "1"
    train_main(args_opt)
    if args_opt.run_online:
        mox.file.copy_parallel(ckpt_root, args_opt.train_url)

if __name__ == '__main__':
    main()
