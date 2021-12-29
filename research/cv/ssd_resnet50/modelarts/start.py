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
"""start"""
import sys
import os
import argparse
import logging
import ast
import glob
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.communication.management import init, get_rank
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import export as export_model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import dtype

CACHE_TRAIN_DATA_URL = "/cache/train_data_url"
CACHE_TRAIN_OUT_URL = "/cache/train_out_url"

if CACHE_TRAIN_OUT_URL != '':
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))

    from src.ssd import SSDWithLossCell, TrainingWrapper, ssd_resnet50
    from src.config import config
    from src.dataset import create_ssd_dataset, data_to_mindrecord_byte_image, voc_data_to_mindrecord
    from src.lr_schedule import get_lr
    from src.init_params import init_net_param, filter_checkpoint_parameter_by_list
    import moxing as mox


def get_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description="SSD training")
    parser.add_argument("--run_platform", type=str, default="Ascend", choices=("Ascend", "GPU", "CPU"),
                        help="run platform, support Ascend, GPU and CPU.")

    # Model output directory
    parser.add_argument("--train_url",
                        type=str, default='', help='the path model saved')
    # 数据集目录
    parser.add_argument("--data_url",
                        type=str, default='', help='the training data')

    parser.add_argument("--only_create_dataset", type=ast.literal_eval, default=False,
                        help="If set it true, only create Mindrecord, default is False.")
    parser.add_argument("--distribute", type=ast.literal_eval, default=False,
                        help="Run distribute, default is False.")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate, default is 0.05.")
    parser.add_argument("--mode", type=str, default="sink", help="Run sink mode or not, default is sink.")
    parser.add_argument("--dataset", type=str, default="coco", help="Dataset, default is coco.")
    parser.add_argument("--epoch_size", type=int, default=10, help="Epoch size, default is 500.")
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

    # adapts parameters in config.py
    parser.add_argument("--feature_extractor_base_param", type=str, default="")
    parser.add_argument("--coco_root", type=str, default="")
    # parser.add_argument("--classes_label_path", type=str, default="labels.txt")
    parser.add_argument("--num_classes", type=int, default=81)
    parser.add_argument("--voc_root", type=str, default="")
    parser.add_argument("--voc_json", type=str, default="")
    parser.add_argument("--image_dir", type=str, default="")
    parser.add_argument("--anno_path", type=str, default="coco_labels.txt")


    args_opt = parser.parse_args()
    return args_opt


def update_config(args_opt):
    """
    补全在config中的数据集路径
    Args:
        args_opt:
        config:

    Returns:

    """
    if config.num_ssd_boxes == -1:
        num = 0
        h, w = config.img_shape
        for i in range(len(config.steps)):
            num += (h // config.steps[i]) * (w // config.steps[i]) * \
                   config.num_default[i]
        config.num_ssd_boxes = num

    data_dir = CACHE_TRAIN_DATA_URL

    # The MindRecord format dataset path is updated to the selected dataset path
    config.mindrecord_dir = data_dir

    # complete the path of the dataset
    dataset = args_opt.dataset
    if dataset == 'coco':
        coco_root = args_opt.coco_root
        config.coco_root = os.path.join(data_dir, coco_root)
        print(f"update config.coco_root {coco_root} to {config.coco_root}")
    elif dataset == 'voc':
        voc_root = args_opt.voc_root
        config.voc_root = os.path.join(data_dir, voc_root)
        print(f"update config.voc_root {voc_root} to {config.voc_root}")
    else:
        image_dir = args_opt.image_dir
        anno_path = args_opt.anno_path
        config.image_dir = os.path.join(data_dir, image_dir)
        config.anno_path = os.path.join(data_dir, anno_path)
        print(f"update config.image_dir {image_dir} to {config.image_dir}")
        print(f"update config.anno_path {anno_path} to {config.anno_path}")

    # with open(os.path.join(data_dir, args_opt.classes_label_path), 'r') as f:
    #     config.classes = [line.strip() for line in f.readlines()]
    config.num_classes = args_opt.num_classes
    print(f"config: {config}")


def get_last_ckpt():
    """get_last_ckpt"""
    ckpt_pattern = os.path.join(CACHE_TRAIN_OUT_URL, "*.ckpt")
    ckpts = glob.glob(ckpt_pattern)
    if not ckpts:
        print(f"Cant't found ckpt in {CACHE_TRAIN_OUT_URL}")
        return None
    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1]


def export(net, device_id, ckpt_file, file_format="AIR", batch_size=1):
    """export"""
    print(f"start export {ckpt_file} to {file_format}, device_id {device_id}")
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(device_id=device_id)

    param_dict = load_checkpoint(ckpt_file)
    net.init_parameters_data()
    load_param_into_net(net, param_dict)
    net.set_train(False)

    input_shp = [batch_size, 3] + config.img_shape
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp),
                         mindspore.float32)
    ckpt_file = CACHE_TRAIN_OUT_URL + '/ssdresnet50'
    export_model(net, input_array, file_name=ckpt_file, file_format=file_format)
    print(f"export {ckpt_file} to {file_format} success.")


def export_air(net, args):
    """export_air"""
    ckpt = get_last_ckpt()
    if not ckpt:
        return
    export(net, args.device_id, ckpt, "AIR")


def ssd_model_build(args_opt):
    """
    build ssd model
    """
    if config.model == "ssd_resnet50":
        ssd = ssd_resnet50(config=config)
        init_net_param(ssd)
        if args_opt.feature_extractor_base_param != "":
            print("args_opt.feature_extractor_base_param的值是****=", args_opt.feature_extractor_base_param)
            print("args_opt.pre_trained****=", args_opt.pre_trained)
            param_dict = load_checkpoint(args_opt.feature_extractor_base_param)
            for x in list(param_dict.keys()):
                param_dict["network.feature_extractor.resnet." + x] = param_dict[x]
                del param_dict[x]
            load_param_into_net(ssd.feature_extractor.resnet, param_dict)
    else:
        raise ValueError(f'config.model: {args_opt.model} is not supported')
    return ssd


def file_create(args_opt, mindrecord_dir, mindrecord_file, prefix):
    """generate mindrecord file path"""
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if args_opt.dataset == "coco":
            if os.path.isdir(config.coco_root):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("coco", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")
        elif args_opt.dataset == "voc":
            if os.path.isdir(config.voc_dir):
                print("Create Mindrecord.")
                voc_data_to_mindrecord(mindrecord_dir, True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("voc_dir not exits.")
        else:
            if os.path.isdir(config.image_dir) and os.path.exists(config.anno_path):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("other", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("image_dir or anno_path not exits.")

    print("*********mindrecord_file", mindrecord_file)


def start_main():
    """start_main"""
    args_opt = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    print("Training setting args:", args_opt)

    os.makedirs(CACHE_TRAIN_DATA_URL, exist_ok=True)
    mox.file.copy_parallel(args_opt.data_url, CACHE_TRAIN_DATA_URL)

    update_config(args_opt)

    if args_opt.distribute:
        device_num = args_opt.device_num
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        init()
        rank = args_opt.device_id % device_num
    else:
        rank = 0
        device_num = 1
    print("Start create dataset!")

    if args_opt.run_platform == "CPU":
        print(args_opt)
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.run_platform, device_id=args_opt.device_id)
        if args_opt.distribute:
            device_num = args_opt.device_num
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                              device_num=device_num)
            init()
            context.set_auto_parallel_context(all_reduce_fusion_config=[29, 58, 89])
            rank = get_rank()


    print("****args_opt.dataset=", args_opt.dataset)
    prefix = "ssd.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    file_create(args_opt, mindrecord_dir, mindrecord_file, prefix)

    if args_opt.only_create_dataset:
        return

    loss_scale = float(args_opt.loss_scale)
    if args_opt.run_platform == "CPU":
        loss_scale = 1.0

    # When create MindDataset, using the fitst mindrecord file, such as ssd.mindrecord0.
    use_multiprocessing = (args_opt.run_platform != "CPU")
    dataset = create_ssd_dataset(mindrecord_file, repeat_num=1, batch_size=args_opt.batch_size,
                                 device_num=device_num, rank=rank, use_multiprocessing=use_multiprocessing)
    print("********dataset", dataset)
    dataset_size = dataset.get_dataset_size()
    print("****-----****dataset_size", dataset)
    print(f"Create dataset done! dataset size is {dataset_size}")
    ssd = ssd_model_build(args_opt)
    print("finish ssd model building ...............")

    if ("use_float16" in config and config.use_float16) or args_opt.run_platform == "GPU":
        ssd.to_float(dtype.float16)
    net = SSDWithLossCell(ssd, config)

    ckpt_config = CheckpointConfig(save_checkpoint_steps=dataset_size * args_opt.save_checkpoint_epochs, \
        keep_checkpoint_max=60)
    ckpoint_cb = ModelCheckpoint(prefix="ssd", directory=CACHE_TRAIN_OUT_URL, config=ckpt_config)

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
    print("Start train SSD, the first epoch will be slower because of the graph compilation.")

    # change the working directory for model saving
    os.makedirs(CACHE_TRAIN_OUT_URL, exist_ok=True)
    os.chdir(CACHE_TRAIN_OUT_URL)
    model.train(args_opt.epoch_size, dataset, callbacks=callback, dataset_sink_mode=dataset_sink_mode)


    # net = SSDWithLossCell(ssd, config)
    net = ssd_resnet50(config=config)
    export_air(net, args_opt)
    mox.file.copy_parallel(CACHE_TRAIN_OUT_URL, args_opt.train_url)


if __name__ == '__main__':
    start_main()
