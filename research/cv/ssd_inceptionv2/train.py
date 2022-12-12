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
import os
import mindspore
import mindspore.nn as nn


from mindspore import context, Tensor
from mindspore.communication.management import init, get_rank
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from src.ssd import SSDWithLossCell, TrainingWrapper, ssd_inception_v2
from src.config import config
from src.dataset import create_ssd_dataset, create_mindrecord, convert_anno, coco2017_to_coco
from src.lr_schedule import get_lr
from src.init_params import filter_checkpoint_parameter_by_list

set_seed(1)

def get_args():
    """get args"""
    parser = argparse.ArgumentParser(description="SSD training")
    parser.add_argument("--run_platform", type=str, default="Ascend", choices=("Ascend"),
                        help="run platform, only support Ascend.")
    parser.add_argument("--mindrecord_url", type=str, default=None, help="mindrecord path, default is none.")
    parser.add_argument("--mindrecord_eval", type=str, default=None, help="mindrecord_eval path, default is none.")
    parser.add_argument("--eval_callback", type=ast.literal_eval, default=False,
                        help="verify or not during training.")
    parser.add_argument("--distribute", type=ast.literal_eval, default=False,
                        help="Run distribute, default is False.")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--data_url", type=str, default="", help="cocoo 14mini data path.")
    parser.add_argument("--train_url", type=str, help="path for checkpoint.")
    parser.add_argument("--run_online", type=str, default=False, help="run online,default is False.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate, default is 0.05.")
    parser.add_argument("--mode", type=str, default="sink", help="Run sink mode or not, default is sink.")
    parser.add_argument("--dataset", type=str, default="coco", help="Dataset, default is coco.")
    parser.add_argument("--epoch_size", type=int, default=100, help="Epoch size, default is 500.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default is 32.")
    parser.add_argument("--pre_trained", type=str, default="", help="Pretrained Checkpoint file path.")
    parser.add_argument("--backbone_pre_trained", type=str, default="",
                        help="BackBone Pretrained Checkpoint file path.")
    parser.add_argument("--pre_trained_epoch_size", type=int, default=0, help="Pretrained epoch size.")
    parser.add_argument("--save_checkpoint_epochs", type=int, default=1, help="Save checkpoint epochs, default is 10.")
    parser.add_argument("--loss_scale", type=int, default=1024, help="Loss scale, default is 1024.")
    parser.add_argument("--filter_weight", type=ast.literal_eval, default=False,
                        help="Filter head weight parameters, default is False.")
    parser.add_argument('--freeze_layer', type=str, default="none", choices=["none", "backbone"],
                        help="freeze the weights of network, support freeze the backbone's weights, "
                             "default is not freezing.")
    parser.add_argument("--data_complete", type=ast.literal_eval, default=True, help="dataset preparation complete.")
    parser.add_argument("--data_url_raw", type=str, default="", help="coco2017 path.")
    parser.add_argument("--number_path", type=str, default="", help="number_id_val path")
    args = parser.parse_args()
    return args

def convert_coco():
    """convert coco2017 to coco 14 mini"""
    args_opt = get_args()
    config.coco_root = args_opt.data_url
    config.coco_root_raw = args_opt.data_url_raw
    if args_opt.run_online:
        import moxing as mox
        config.coco_root = "/cache/data_url"
        config.coco_root_raw = "/cache/data_url_raw"
        mox.file.copy_parallel(args_opt.data_url_raw, config.coco_root_raw)
    print("Start convert coco2017 json file to coco 14 mini json file!")
    convert_anno(config, args_opt.number_path)
    print("Start convert coco2017 dataset to coco 14 mini dataset!")
    coco2017_to_coco(config, args_opt.number_path)
    print("Done!")
    if args_opt.run_online:
        mox.file.copy_parallel(config.coco_root, args_opt.data_url)

def ssd_model_build():
    """create network"""
    ssd = ssd_inception_v2(configs=config)
    if config.feature_extractor_base_param != "":
        ssd.init_parameters_data()
        param_dict = load_checkpoint(config.feature_extractor_base_param)
        load_param_into_net(ssd, param_dict)
    return ssd


def main():
    args_opt = get_args()
    device_id = args_opt.device_id
    rank = 0
    device_num = 1
    config.coco_root = args_opt.data_url
    config.mindrecord_dir = args_opt.mindrecord_url
    config.feature_extractor_base_param = args_opt.backbone_pre_trained
    config.checkpoint_path = args_opt.pre_trained
    local_train_url = args_opt.train_url
    mindrecord_exist = False
    mindrecord_files = []
    if args_opt.run_online:
        import moxing as mox
        config.coco_root = "/cache/data_url"
        config.mindrecord_dir = "/cache/mindrecord_url"
        mox.file.copy_parallel(args_opt.mindrecord_url, config.mindrecord_dir)
        mox.file.copy_parallel(args_opt.data_url, config.coco_root)
        if args_opt.backbone_pre_trained != "":
            config.feature_extractor_base_param = "/cache/backbone_checkpoint_path/backcheckpoint.ckpt"
            mox.file.copy_parallel(args_opt.backbone_pre_trained, config.feature_extractor_base_param)
        if args_opt.pre_trained != "":
            config.checkpoint_path = "/cache/checkpoint_path/checkpoint.ckpt"
            mox.file.copy_parallel(args_opt.pre_trained, config.checkpoint_path)
        local_train_url = "/cache/train_out_si"
    if args_opt.distribute:
        context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.run_platform)
        device_num = args_opt.device_num
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        init()
        context.set_auto_parallel_context(all_reduce_fusion_config=[29, 58, 89])
        rank = get_rank()
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.run_platform, device_id=device_id)
    # check whether there is a mindrecord file.
    file_list = os.listdir(config.mindrecord_dir)
    if file_list:
        print("Mindrecord files exist.")
        mindrecord_exist = True
        mindrecord_files = [config.mindrecord_dir + "/ssd.mindrecord{}".format(i) for i in range(8)]
    if not mindrecord_exist:
        if rank == 0:
            mindrecord_file = create_mindrecord(args_opt.dataset, "ssd.mindrecord", True)
            if args_opt.run_online:
                import moxing as mox
                mox.file.copy_parallel(config.mindrecord_dir, args_opt.mindrecord_url)
        else:
            file_num = len(os.listdir(config.mindrecord_dir))
            while file_num != 16:
                file_num = len(os.listdir(config.mindrecord_dir))
        mindrecord_files = [mindrecord_file.strip('0') + '{}'.format(i) for i in range(8)]
    loss_scale = float(args_opt.loss_scale)
    dataset = create_ssd_dataset(mindrecord_files, repeat_num=1, batch_size=args_opt.batch_size,
                                 device_num=device_num, rank=rank, use_multiprocessing=True)
    dataset_size = dataset.get_dataset_size()
    print(f"Create dataset done! dataset size is {dataset_size}")
    ssd = ssd_model_build()
    net = SSDWithLossCell(ssd, config)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=dataset_size * args_opt.save_checkpoint_epochs)
    ckpoint_cb = ModelCheckpoint(prefix="ssd", directory=local_train_url+"/card_{}".format(rank), config=ckpt_config)

    if config.checkpoint_path != "":
        print("Load Checkpoint!")
        param_dict = load_checkpoint(config.checkpoint_path)
        net.init_parameters_data()
        if args_opt.filter_weight:
            filter_checkpoint_parameter_by_list(param_dict, config.checkpoint_filter_list)
        load_param_into_net(net, param_dict)
    lr = Tensor(get_lr(global_step=args_opt.pre_trained_epoch_size, lr_init=config.lr_init, lr_end=config.lr_end_rate,
                       lr_max=args_opt.lr, warmup_epochs=config.warmup_epochs, total_epochs=args_opt.epoch_size,
                       steps_per_epoch=dataset_size), mindspore.float32)
    group_params = filter(lambda x: x.requires_grad, net.get_parameters())
    opt = nn.Momentum(group_params, lr, config.momentum, config.weight_decay, loss_scale)
    net = TrainingWrapper(net, opt, loss_scale)
    callback = [TimeMonitor(data_size=dataset_size), LossMonitor(), ckpoint_cb]
    model = Model(net)
    if args_opt.dataset == "coco":
        json_path = os.path.join(config.coco_root, config.instances_set.format(config.val_data_type))
    if args_opt.eval_callback:
        local_eval_data = ""
        if args_opt.run_online:
            local_eval_data = "/cache/data_eval"
            import moxing as mox
            mox.file.copy_parallel(args_opt.mindrecord_eval, local_eval_data)
        else:
            local_eval_data = args_opt.mindrecord_eval
        from src.callback import eval_callback
        eval_cb = eval_callback(local_eval_data, ssd, json_path, args_opt.batch_size, eval_per_epoch=10)
        callback.append(eval_cb)
    dataset_sink_mode = False
    if args_opt.mode == "sink":
        print("In sink mode, one epoch return a loss.")
        dataset_sink_mode = True
    print("Start train SSD, the first epoch will be slower because of the graph compilation.")
    model.train(args_opt.epoch_size, dataset, callbacks=callback, dataset_sink_mode=dataset_sink_mode)
    if args_opt.run_online == "True":
        mox.file.copy_parallel(local_train_url, args_opt.train_url)


if __name__ == '__main__':
    arg = get_args()
    if arg.data_complete:
        main()
    else:
        convert_coco()
