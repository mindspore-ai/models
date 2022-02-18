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
''' train PoseEstNet '''
import ast
import argparse
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net

from src.loss import NetWithLoss
from src.scheduler import get_lr
from src.model import get_pose_net
from src.dataset import create_dataset
from src.config import cfg, update_config

parser = argparse.ArgumentParser(description='Train PoseEstNet network')

parser.add_argument('--train_url', type=str)
parser.add_argument('--data_url', type=str)
parser.add_argument('--pre_trained', type=ast.literal_eval, default=False)
parser.add_argument('--device_target', type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),\
                    help="Device target, support Ascend, GPU and CPU.")
parser.add_argument('--cfg', help='experiment configure file name', required=False, type=str, default="config.yaml")
parser.add_argument('--pre_ckpt_path', type=str, default='./pretrain/PoseEstNet.ckpt')
parser.add_argument('--model_format', type=str, default='AIR')

args = parser.parse_args()

if __name__ == '__main__':
    update_config(cfg, args)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)

    # define dataset
    dataset = create_dataset(cfg, args.data_url, is_train=True)
    step_size = dataset.get_dataset_size()

    #define net
    network = get_pose_net(cfg)

    if args.pre_trained:
        pre_path = args.pre_ckpt_path[args.pre_ckpt_path.rfind('/'), -1]
        param_dict = load_checkpoint(pre_path)
        load_param_into_net(network, param_dict)

    net_with_loss = NetWithLoss(network, use_target_weight=True)

    #init lr
    lr = get_lr(lr=cfg.TRAIN.LR,
                total_epochs=cfg.TRAIN.END_EPOCH,
                steps_per_epoch=step_size,
                lr_step=cfg.TRAIN.LR_STEP,
                gamma=cfg.TRAIN.LR_FACTOR)
    lr = Tensor(lr, mindspore.float32)

    #define opt
    decayed_params = []
    no_decayed_params = []
    for param in network.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': cfg.TRAIN.WD},
                    {'params': no_decayed_params},
                    {'order_params': network.trainable_params()}]

    optimizer = nn.Adam(group_params,
                        learning_rate=lr,
                        weight_decay=cfg.TRAIN.WD,
                        use_nesterov=cfg.TRAIN.NESTEROV)

    model = Model(net_with_loss, optimizer=optimizer)
    # define callbacks
    time_cb = TimeMonitor(data_size=dataset.get_dataset_size())
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]

    config_ck = CheckpointConfig(save_checkpoint_steps=10 * step_size, keep_checkpoint_max=30)

    save_checkpoint_path = args.train_url

    ckpt_cb = ModelCheckpoint(prefix="PoseEstNet",
                              directory=save_checkpoint_path,
                              config=config_ck)
    cb += [ckpt_cb]

    print("=============================")
    print("Total epoch: {}".format(cfg.TRAIN.END_EPOCH))
    print("Batch size: {}".format(cfg.TRAIN.BATCH_SIZE))
    print("=======Training begin========")

    model.train(cfg.TRAIN.END_EPOCH, dataset, callbacks=cb, dataset_sink_mode=True)

    input_arr = mindspore.numpy.zeros((cfg.TRAIN.BATCH_SIZE, 3, 256, 256), mindspore.float32)
    export(network, Tensor(input_arr), file_name=args.train_url + \
            'PoseEstNet', file_format=args.model_format)
