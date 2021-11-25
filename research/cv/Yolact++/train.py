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
""" Train our Model """
import os
import argparse
import ast
from mindspore.communication.management import get_rank, get_group_size #
import mindspore.common.dtype as mstype
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.nn as nn
from mindspore.common import set_seed
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor, SummaryCollector, Callback
from mindspore.common import initializer as init_p
from src.loss_monitor import LossMonitor
from src.yolact.layers.modules.loss import MultiBoxLoss
from src.yolact.yolactpp import Yolact
from src.config import yolact_plus_resnet50_config as cfg
from src.dataset import data_to_mindrecord_byte_image, create_yolact_dataset
from src.lr_schedule import dynamic_lr
from src.network_define import WithLossCell

set_seed(1)

parser = argparse.ArgumentParser(description="Yolact++ training")
parser.add_argument("--only_create_dataset", type=ast.literal_eval, default=False, help="If set it true, only create "
                    "Mindrecord, default is false.")
# Modelarts --run_distribute default is True
parser.add_argument("--run_distribute", type=ast.literal_eval, default=False, help="Run distribute, default is false.")
parser.add_argument("--do_train", type=ast.literal_eval, default=True, help="Do train or not, default is true.")
parser.add_argument("--do_eval", type=ast.literal_eval, default=False, help="Do eval or not, default is false.")
parser.add_argument("--dataset", type=str, default="coco", help="Dataset, default is coco.")
parser.add_argument("--pre_trained", type=str, default=None, help="Pretrain file path.")
parser.add_argument("--device_id", type=int, default=3, help="Device id, default is 0.")
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
parser.add_argument("--rank_id", type=int, default=0, help="Rank id, default is 0.")
parser.add_argument("--net_ckpt", type=str,
                    default="/data/yolact/yolact-20_619.ckpt", help="Do")
parser.add_argument("--run_platform", type=str, default="Ascend", choices="Ascend",
                    help="run platform, only support Ascend.")
parser.add_argument("--distribute", type=ast.literal_eval, default=False, help="Run distribute, default is False.")

parser.add_argument("--train_url", type=str, default="obs://xxx", help="ckpt output dir in obs")    # Modelarts
parser.add_argument("--data_url", type=str, default="obs://xxx", help="mindrecord file path.")      # Modelarts
parser.add_argument('--is_modelarts', type=str, default="False", help='is train on modelarts')
args_opt = parser.parse_args()

class TransferCallback(Callback):
    """Callback"""
    def __init__(self, local_train_path, obs_train_path):
        super(TransferCallback, self).__init__()
        self.local_train_path = local_train_path
        self.obs_train_path = obs_train_path

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        current_epoch = cb_params.cur_epoch_num
        if current_epoch % 10 == 0 and current_epoch != 0:
            mox.file.copy_parallel(self.local_train_path, self.obs_train_path)

def init_weights(module):

    for name, cell in module.cells_and_names():
        is_conv_layer = isinstance(cell, nn.Conv2d)

        if is_conv_layer and "backbone" not in name:
            cell.weight.set_data(init_p.initializer('XavierUniform', cell.weight.shape))

            if cell.has_bias is True:
                cell.bias.set_data(init_p.initializer('zeros', cell.bias.shape))

if __name__ == '__main__':
    print("Start train for yolact!")
    if args_opt.run_platform == "Ascend":
        if args_opt.is_modelarts == "True":
            import moxing as mox
            context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id,
                                max_call_depth=10000)
            device_id = int(os.getenv('DEVICE_ID'), 0)
            if not args_opt.do_eval and args_opt.run_distribute:
                init()
                rank = get_rank()
                device_num = get_group_size()
                context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                                  gradients_mean=True)
            else:
                rank = 0
                device_num = 1
        else:
            context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", max_call_depth=10000,
                                enable_reduce_precision=True)
            if args_opt.distribute:
                if os.getenv("DEVICE_ID", "not_set").isdigit():
                    context.set_context(device_id=int(os.getenv("DEVICE_ID")))
                init()
                device_num = int(os.getenv("DEVICE_NUM"))
                rank = int(os.getenv("RANK_ID"))
                rank_size = int(os.getenv("RANK_SIZE"))
                context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                                  device_num=device_num)
            else:
                rank = 0
                device_num = 1
                context.set_context(device_id=int(args_opt.device_id), save_graphs=True)
    else:
        raise ValueError("Unsupported platform.")

    print("Start create dataset!")
    if args_opt.is_modelarts == "True":
        ckpt_filename = "resnet50.ckpt"

        local_data_url = "/cache/mr/" + str(device_id)
        local_pretrained_url = "/cache/weights/"
        local_train_url = "/cache/ckpt"

        mox.file.make_dirs(local_data_url)
        mox.file.make_dirs(local_train_url)
        mox.file.make_dirs(local_pretrained_url)

        local_pretrained_url = local_pretrained_url + "resnet50.ckpt"

        filename = "yolact.mindrecord0"
        mox.file.copy_parallel(args_opt.data_url, local_data_url)
        if args_opt.pre_trained is not None:
            mox.file.copy(args_opt.pre_trained, local_pretrained_url)
        local_data_path = os.path.join(local_data_url, filename)
    else:
        prefix = "yolact.mindrecord"
        mindrecord_dir = cfg['mindrecord_dir']
        mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
        if rank == 0 and not os.path.exists(mindrecord_file):
            if not os.path.isdir(mindrecord_dir):
                os.makedirs(mindrecord_dir)
            if args_opt.dataset == "coco":
                if os.path.isdir(cfg['coco_root']):
                    print("Create Mindrecord.")
                    data_to_mindrecord_byte_image("coco", True, prefix)
                    print("Create Mindrecord Done, at {}".format(mindrecord_dir))
                else:
                    raise Exception("coco_root not exits.")
            else:
                if os.path.isdir(cfg['IMAGE_DIR']) and os.path.exists(cfg['ANNO_PATH']):
                    print("Create Mindrecord.")
                    data_to_mindrecord_byte_image("other", True, prefix)
                    print("Create Mindrecord Done, at {}".format(mindrecord_dir))
                else:
                    raise Exception("IMAGE_DIR or ANNO_PATH not exits.")

    if not args_opt.only_create_dataset:
        if args_opt.is_modelarts == "True":
            dataset = create_yolact_dataset(local_data_path, batch_size=cfg['batch_size'],
                                            device_num=device_num, rank_id=rank)
        else:
            dataset = create_yolact_dataset(mindrecord_file, batch_size=cfg['batch_size'],
                                            device_num=device_num, rank_id=rank)

        num_steps = dataset.get_dataset_size()
        print("pre epoch step num: ", num_steps)
        print("Create dataset done!")
        net = Yolact()
        net = net.set_train()

        if args_opt.is_modelarts == "True":
            ckpt_file_name = "resnet50.ckpt"
            backbone_path = local_pretrained_url
            if args_opt.pre_trained is not None:
                param_dict = load_checkpoint(backbone_path)
                if cfg['pretrain_epoch_size'] == 0:
                    for item in list(param_dict.keys()):
                        if not item.startswith('backbone'):
                            param_dict.pop(item)
                load_param_into_net(net, param_dict)

        init_weights(net)

        if args_opt.is_modelarts == "False":
            ckpt_path = args_opt.net_ckpt
            if ckpt_path != "":
                param_dict = load_checkpoint(ckpt_path)
                load_param_into_net(net, param_dict)


        loss = MultiBoxLoss(num_classes=cfg['num_classes'], pos_threshold=cfg['positive_iou_threshold'],
                            neg_threshold=cfg['negative_iou_threshold'], negpos_ratio=cfg['ohem_negpos_ratio'],
                            batch_size=cfg['batch_size'], num_priors=cfg['num_priors'])
        net_with_loss = WithLossCell(net, loss)

        lr = Tensor(dynamic_lr(cfg, start_epoch=0, total_epochs=cfg['epoch_size'], steps_each_epoch=num_steps),
                    mstype.float32)

        opt = nn.Momentum(params=net.trainable_params(), learning_rate=cfg['lr'], momentum=cfg['momentum'],
                          weight_decay=cfg['decay'], loss_scale=cfg['loss_scale'])

        # define model
        if args_opt.is_modelarts == "True":
            model = Model(net_with_loss, optimizer=opt, amp_level='O0')
        else:
            model = Model(net_with_loss, optimizer=opt, amp_level='O3')

        print("============== Starting Training ==============")
        time_cb = TimeMonitor(data_size=num_steps)
        loss_cb = LossMonitor()

        if args_opt.is_modelarts == "True":
            summary_collector = SummaryCollector(summary_dir='./summary_dir', collect_freq=100)
        else:
            summary_collector = SummaryCollector(summary_dir='./summary_dir', collect_freq=10)

        cb = [time_cb, loss_cb]
        if args_opt.is_modelarts == "True":
            if cfg['save_checkpoint']:
                ckptconfig = CheckpointConfig(save_checkpoint_steps=cfg['save_checkpoint_epochs'] * num_steps,
                                              keep_checkpoint_max=cfg['keep_checkpoint_max'])

                ckpoint_cb = ModelCheckpoint(prefix='yolact', directory=local_train_url, config=ckptconfig)
                transferCb = TransferCallback(local_train_url, args_opt.train_url)
                if device_id == 0:
                    cb += [ckpoint_cb, transferCb]
            model.train(cfg['epoch_size'], dataset, callbacks=cb, dataset_sink_mode=True)
        else:
            if cfg['save_checkpoint']:
                ckptconfig = CheckpointConfig(save_checkpoint_steps=cfg['save_checkpoint_epochs'] * num_steps,
                                              keep_checkpoint_max=cfg['keep_checkpoint_max'])
                save_checkpoint_path = os.path.join(cfg['save_checkpoint_path'], 'ckpt_' + str(rank) + '/')
                ckpoint_cb = ModelCheckpoint(prefix='yolact', directory=save_checkpoint_path, config=ckptconfig)
                cb += [ckpoint_cb]
            model.train(cfg['epoch_size'], dataset, callbacks=cb, dataset_sink_mode=False)

        print("============== End Training ==============")
