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

"""Train SSD and get checkpoint files."""

import argparse
import ast
from pathlib import Path

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common import dtype
from mindspore.common import set_seed
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.train import Model
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import LossMonitor
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import TimeMonitor

from src.config import config
from src.dataset import create_mindrecord
from src.dataset import create_ssd_dataset
from src.lr_schedule import get_lr
from src.ssd_resnet34 import SSDTrainWithLossCell
from src.ssd_resnet34 import TrainingWrapper

set_seed(1)


def get_args():
    """
    Get arguments
    """
    parser = argparse.ArgumentParser(description="SSD-ResNet34 training")
    parser.add_argument("--data_url", type=str, help="A path to the dataset.")
    parser.add_argument("--train_url", type=str, help="A path, where checkpoints and logs will be saved.")
    parser.add_argument("--mindrecord_url", type=str, help="A path to the mindrecord dataset.")
    parser.add_argument("--run_platform", type=str, default="CPU", choices=("GPU", "CPU"),
                        help="run platform, support GPU and CPU.")
    parser.add_argument("--only_create_dataset", type=ast.literal_eval, default=False,
                        help="If set it true, only create mindrecord, default is False.")
    parser.add_argument("--distribute", type=ast.literal_eval, default=False,
                        help="Run distribute, default is False.")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Maximum learning rate during training, default is 0.05.")
    parser.add_argument("--mode", type=str, default="sink", choices=['sink', "none"],
                        help="Run sink mode or not, default is sink.")
    parser.add_argument("--dataset", type=str, default="coco", help="Dataset, default is coco.")
    parser.add_argument("--epoch_size", type=int, default=500, help="Epoch size, default is 500.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default is 32.")
    parser.add_argument("--pre_trained", type=str, default=None,
                        help="Path to the parent folder with the checkpoints for each GPU.")
    parser.add_argument("--pre_trained_epochs", type=int,
                        default=0, help="Pretrained epoch size.")
    parser.add_argument("--save_checkpoint_epochs", type=int,
                        default=10, help="Save checkpoint epochs, default is 10.")
    parser.add_argument("--loss_scale", type=int, default=1024, help="Loss scale, default is 1024.")
    parser.add_argument("--filter_weight", type=ast.literal_eval, default=False,
                        help="Filter head weight parameters, default is False.")
    parser.add_argument('--freeze_layer', type=str, default="none", choices=["none", "backbone"],
                        help="freeze the weights of network, support freeze the backbone's weights, "
                             "default is not freezing.")
    args_opt = parser.parse_args()
    return args_opt


def _prepare_environment(args_opt):
    """Update configuration and set MindSpore context"""
    rank = 0
    device_num = args_opt.device_num

    config.coco_root = args_opt.data_url
    config.mindrecord_dir = args_opt.mindrecord_url
    config.pre_trained = args_opt.pre_trained
    config.pre_trained_epochs = args_opt.pre_trained_epochs
    config.local_train_url = args_opt.train_url
    config.batch_size = args_opt.batch_size

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.run_platform)

    if args_opt.distribute and not args_opt.only_create_dataset:
        init()
        rank = get_rank()
        device_num = get_group_size()
        context.reset_auto_parallel_context()

        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
            global_rank=0,
            all_reduce_fusion_config=[29, 58, 89],
        )

    config.device_num = device_num
    config.rank = rank

    print("Rank: {}, Device num: {}".format(rank, device_num))


def _prepare_dataset(args_opt, mindrecord_file):
    """Prepare a MindSpore dataset for training"""
    # When create MindDataset, using the first mindrecord file, such as ssd.mindrecord0.
    use_multiprocessing = (args_opt.run_platform != "CPU")
    dataset = create_ssd_dataset(
        mindrecord_file,
        repeat_num=1,
        batch_size=config.batch_size,
        device_num=config.device_num,
        rank=config.rank,
        use_multiprocessing=use_multiprocessing,
    )
    dataset_size = dataset.get_dataset_size()
    print("Create dataset done! dataset size is {}".format(dataset_size))
    return dataset, dataset_size


def _prepare_mindrecords(args_opt):
    """Prepare MindRecords and return a path"""
    mindrecord_file = create_mindrecord(
        args_opt.dataset,
        "ssd.mindrecord",
        True,
        rank=config.rank,
    )
    return mindrecord_file


def main():
    """
    Execute training process!
    """
    args_opt = get_args()
    _prepare_environment(args_opt)

    mindrecord_file = _prepare_mindrecords(args_opt)

    if args_opt.only_create_dataset:
        return

    loss_scale = float(args_opt.loss_scale)
    if args_opt.run_platform == "CPU":
        loss_scale = 1.0

    dataset, dataset_size = _prepare_dataset(args_opt, mindrecord_file)

    net = SSDTrainWithLossCell(config)

    if ("use_float16" in config and config.use_float16) and args_opt.run_platform == "GPU":
        if config.rank == 0:
            print("Converting model to the float16 format.")

        net.ssd_resnet34_network.to_float(dtype.float16)

    # checkpoint
    checkpoint_config = CheckpointConfig(
        save_checkpoint_steps=dataset_size * args_opt.save_checkpoint_epochs,
        keep_checkpoint_max=10,
    )

    if not Path(config.local_train_url).exists():
        Path(config.local_train_url).mkdir(parents=True)

    checkpoint_callback = ModelCheckpoint(
        prefix="ssd",
        directory=config.local_train_url + "/card{}".format(config.rank),
        config=checkpoint_config,
    )

    lr = Tensor(get_lr(
        global_step=config.pre_trained_epochs * dataset_size,
        lr_init=config.lr_init,
        lr_end=config.lr_end_rate * args_opt.lr,
        lr_max=args_opt.lr,
        warmup_epochs=config.warmup_epochs,
        total_epochs=args_opt.epoch_size,
        steps_per_epoch=dataset_size,
    ))

    if config.rank == 0:
        print("Total number of epochs: {}".format(lr.shape[0] / dataset_size))

    if "use_grad_clip" in config and config.use_grad_clip:
        opt = nn.Momentum(
            filter(lambda x: x.requires_grad, net.get_parameters()),
            lr,
            config.momentum,
            config.weight_decay,
            1.0,
        )
        net = TrainingWrapper(net, opt, loss_scale, True, pretrain_path=config.pre_trained)
    else:
        opt = nn.Momentum(
            filter(lambda x: x.requires_grad, net.get_parameters()),
            lr,
            config.momentum,
            config.weight_decay,
            loss_scale,
        )
        net = TrainingWrapper(net, opt, loss_scale, pretrain_path=config.pre_trained)

    if args_opt.mode == 'sink':
        print_loss_per_epoches = 1
    else:
        print_loss_per_epoches = dataset_size

    callback = [
        TimeMonitor(data_size=dataset_size),
        LossMonitor(per_print_times=print_loss_per_epoches),
        checkpoint_callback,
    ]

    model = Model(net)

    dataset_sink_mode = False

    if args_opt.mode == "sink" and args_opt.run_platform != "CPU":
        print("In sink mode, one epoch return a loss.")
        dataset_sink_mode = True
    print("Start train SSD, the first epoch will be slower because of the graph compilation.")
    model.train(args_opt.epoch_size, dataset, callbacks=callback, dataset_sink_mode=dataset_sink_mode)


if __name__ == '__main__':
    main()
