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

"""train_SSD."""

import os
import sys
import argparse
import ast
import json
import numpy as np
import moxing as mox
from src.box_utils import default_boxes
from src.init_params import init_net_param, filter_checkpoint_parameter
from src.lr_schedule import get_lr
from src.dataset import create_ssd_dataset
from src.config import config
from src.ssd import SSD320, SSDWithLossCell, SsdInferWithDecoder, \
    TrainingWrapper, ssd_mobilenet_v2

import mindspore
from mindspore import nn
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.context import ParallelMode
from mindspore.common import dtype
from mindspore import context, Tensor
from mindspore.communication.management import init, get_group_size
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

base_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(base_path + "/..")
sys.path.append(base_path + "/../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


rank_size = int(os.environ.get("RANK_SIZE", 1))
device_id = int(os.getenv('DEVICE_ID'))
print("rank_size:", rank_size)
print("device_id:", device_id)


def get_epoch(ckpt_name):
    start = ckpt_name.find('-')
    start += len('-')
    end = ckpt_name.find('_', start)
    epoch = ast.literal_eval(ckpt_name[start:end].strip())
    return epoch


def get_ckpt_epoch(ckpt_dir):
    """Get checkpoint epoch"""

    ckpt_epoch = {}
    files = os.listdir(ckpt_dir)
    if not files:
        print("No ckpt files")
        return None, None

    for file_name in files:
        file_path = os.path.join(ckpt_dir, file_name)
        if os.path.splitext(file_path)[1] == '.ckpt':
            epoch = get_epoch(file_name)
            ckpt_epoch[file_name] = epoch
    newest_ckpt = max(ckpt_epoch, key=ckpt_epoch.get)
    max_epoch = ckpt_epoch[newest_ckpt]
    return newest_ckpt, max_epoch


def training_env_set(run_platform):
    """Set training env"""

    context.set_context(mode=context.GRAPH_MODE, device_target=run_platform)
    if rank_size > 1:
        context.reset_auto_parallel_context()
        init()
        context.set_auto_parallel_context(
            all_reduce_fusion_config=[29, 58, 89])
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=get_group_size())


def read_classes(class_file, num_classes):
    """Read classes"""

    mox.file.shift('os', 'mox')
    if not os.path.isfile(class_file):
        raise RuntimeError("class file is not valid.")

    with open(class_file, "rb") as f:
        js = json.load(f)
    classes = js['classes']
    assert len(
        classes) == num_classes, "object counts in json file does not match num_classes"
    return classes


def get_args():
    """Get args"""

    parser = argparse.ArgumentParser(description="SSD training")
    parser.add_argument("--data_url", type=str, default="",
                        help="This should be set to the same directory given to the data_download's data_dir argument")
    parser.add_argument(
        "--train_url",
        type=str,
        default="",
        help="obs output path")

    # platform params
    parser.add_argument("--run_platform", type=str, default="Ascend", choices=("Ascend", "GPU"),
                        help="run platform, support Ascend, GPU.")
    # net hyper param
    parser.add_argument(
        "--epoch_size",
        type=int,
        default=500,
        help="Epoch size, default is 500.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size, default is 32.")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate, default is 0.05.")
    # resume training
    parser.add_argument(
        "--pre_trained",
        type=str,
        default=None,
        help="Pretrained Checkpoint file path.")
    parser.add_argument(
        "--pre_trained_epoch_size",
        type=int,
        default=0,
        help="Pretrained epoch size.")
    parser.add_argument("--filter_weight", type=ast.literal_eval, default=False,
                        help="Filter head weight parameters, default is False.")

    # datasets
    parser.add_argument("--only_create_dataset", type=ast.literal_eval, default=False,
                        help="If set it true, only create Mindrecord, default is False.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="coco",
        help="Dataset, default is coco.")
    parser.add_argument(
        "--class_file",
        type=str,
        default="",
        help=".json file with class names, background should always be \
            included as the first one.(For coco or voc only)")
    parser.add_argument(
        "--num_classes",
        type=int,
        default=-1,
        help="num of classes, for coco or voc only")
    # other modes
    parser.add_argument('--freeze_layer', type=str, default="none", choices=["none", "backbone"],
                        help="freeze the weights of network, support freeze the backbone's weights, "
                             "default is not freezing.")
    parser.add_argument(
        "--mode",
        type=str,
        default="sink",
        help="Run sink mode or not, default is sink.")  # --
    parser.add_argument(
        "--save_checkpoint_epochs",
        type=int,
        default=10,
        help="Save checkpoint epochs, default is 10.")  # --
    parser.add_argument(
        "--loss_scale",
        type=int,
        default=1024,
        help="Loss scale, default is 1024.")  # --

    # export
    parser.add_argument(
        '--file_format',
        type=str,
        choices=[
            "AIR",
            "ONNX",
            "MINDIR"],
        default='MINDIR',
        help='file format')
    args_opt = parser.parse_args()

    assert args_opt.epoch_size > args_opt.pre_trained_epoch_size, \
        "Total epoch_size should be bigger than pre_trained_epoch_size"

    # set config

    if args_opt.num_classes != -1:
        config.classes = read_classes(
            os.path.join(args_opt.data_url, args_opt.class_file), args_opt.num_classes)
        config.num_classes = args_opt.num_classes
        print("Self defined category names loaded.")
    else:
        print("Use default categories. ")

    config.mindrecord_dir = "/cache/dataset"

    return args_opt


def training(args_opt):
    """Train"""

    mox.file.shift('os', 'mox')

    # check dataset
    mox.file.copy_parallel(args_opt.data_url, config.mindrecord_dir)
    mindrecord_file = os.path.join(config.mindrecord_dir, "ssd.mindrecord0")
    if not os.path.exists(mindrecord_file):
        raise RuntimeError("mindrecord file is not valid.")

    # When create MindDataset, using the fitst mindrecord file, such as
    # ssd.mindrecord0.
    use_multiprocessing = (args_opt.run_platform != "CPU")
    dataset = create_ssd_dataset(mindrecord_file, repeat_num=1, batch_size=args_opt.batch_size,
                                 device_num=rank_size, rank=device_id, use_multiprocessing=use_multiprocessing)

    dataset_size = dataset.get_dataset_size()
    print("Dataset loaded")

    backbone = ssd_mobilenet_v2()
    if config.model == "ssd320":
        ssd = SSD320(backbone=backbone, config=config)
    elif config.model == "ssd_mobilenet_v2":
        ssd = ssd_mobilenet_v2(config=config)
    else:
        raise ValueError(f'config.model: {config.model} is not supported')
    if args_opt.run_platform == "GPU":
        ssd.to_float(dtype.float16)
    net = SSDWithLossCell(ssd, config)

    init_net_param(net)

    if config.feature_extractor_base_param != "":
        param_dict = load_checkpoint(config.feature_extractor_base_param)
        for x in list(param_dict.keys()):
            param_dict["network.feature_extractor.mobilenet_v1." +
                       x] = param_dict[x]
            del param_dict[x]
        load_param_into_net(
            ssd.feature_extractor.mobilenet_v1.network,
            param_dict)

    # checkpoint
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=dataset_size *
        args_opt.save_checkpoint_epochs)
    save_ckpt_path = './ckpt_' + str(device_id) + '/'
    ckpoint_cb = ModelCheckpoint(
        prefix="ssd",
        directory=save_ckpt_path,
        config=ckpt_config)

    # load pretrained ckpt
    if args_opt.pre_trained:
        resumed_ckpt = os.path.join(args_opt.data_url, args_opt.pre_trained)

        param_dict = load_checkpoint(resumed_ckpt)
        if args_opt.filter_weight:
            filter_checkpoint_parameter(param_dict)
        load_param_into_net(net, param_dict)
        print("Pretrained ckpt loaded.")

    if args_opt.freeze_layer == "backbone":
        for param in backbone.feature_1.trainable_params():
            param.requires_grad = False

    lr = Tensor(get_lr(global_step=args_opt.pre_trained_epoch_size * dataset_size,
                       lr_init=config.lr_init, lr_end=config.lr_end_rate * args_opt.lr, lr_max=args_opt.lr,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=args_opt.epoch_size,
                       steps_per_epoch=dataset_size))

    loss_scale = float(args_opt.loss_scale)
    if args_opt.run_platform == "CPU":
        loss_scale = 1.0

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
    print("Start train SSD, the first epoch will be slower because of the graph compilation.")

    try:
        model.train(
            args_opt.epoch_size -
            args_opt.pre_trained_epoch_size,
            dataset,
            callbacks=callback,
            dataset_sink_mode=dataset_sink_mode)
    finally:
        if not os.path.exists(os.path.dirname(save_ckpt_path)):
            os.makedirs(os.path.dirname(save_ckpt_path), exist_ok=True)
        mox.file.copy_parallel(save_ckpt_path, args_opt.train_url)
    # ------------------ train_eval end -----------------------


def file_export(args_opt):
    """Export to file"""

    if config.model == "ssd320":
        net = SSD320(ssd_mobilenet_v2(), config, is_training=False)
    else:
        net = ssd_mobilenet_v2(config=config)
    net = SsdInferWithDecoder(net, Tensor(default_boxes), config)

    save_ckpt_path = './ckpt_' + str(device_id) + '/'
    ckpt_file, _ = get_ckpt_epoch(save_ckpt_path)
    if ckpt_file is None:
        return
    param_dict = load_checkpoint(os.path.join(save_ckpt_path, ckpt_file))
    net.init_parameters_data()
    load_param_into_net(net, param_dict)
    net.set_train(False)

    input_shp = [1, 3] + config.img_shape
    input_array = Tensor(
        np.random.uniform(-1.0, 1.0, size=input_shp), mindspore.float32)
    export(net, input_array, file_name="ssd", file_format=args_opt.file_format)
    name = 'ssd.' + args_opt.file_format.lower()
    mox.file.copy(name, os.path.join(args_opt.train_url, name))
    print("Export finished.")


if __name__ == '__main__':
    args = get_args()  # self defined here
    training_env_set(args.run_platform)

    training(args)
    print("Training Finished.")

    # export
    if rank_size > 1:
        if device_id == 0:
            file_export(args)
    else:
        file_export(args)

    print(f"{'successful':*^30}")
