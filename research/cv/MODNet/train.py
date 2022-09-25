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
import os
import argparse
import ast
import time
from pprint import pprint
import mindspore as ms
from mindspore import nn
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init
from mindspore.communication.management import get_group_size, get_rank
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.models import MODNet
from src.losses import MODNetLossCell
from src.dataset import create_dataset
from src.utils import load_config, merge

def preLauch():
    """parse the console argument"""
    parser = argparse.ArgumentParser(description='matting objective decomposition network !')
    parser.add_argument("-c", "--config", help="configuration file to use")
    parser.add_argument("--distribute", type=ast.literal_eval, default=False,
                        help="Run distribute, default is false.")
    parser.add_argument('--device_target', type=str, default='GPU',
                        help='device target, Ascend or GPU (Default: GPU)')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id of training (Default: 0)')
    parser.add_argument('--output_path', type=str, default='./output',
                        help='output path of training (default ./output)')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--start_epochs', type=int, default=0,
                        help='start epochs of training (Default: 0)')
    parser.add_argument('--epoch_size', type=int, default=40,
                        help='epoch size of training (Default: 40)')
    parser.add_argument('--init_weight_path', type=str,
                        default='./init_weight.ckpt', help='checkpoint dir of init_weight')
    parser.add_argument('--ckpt_path', type=str,
                        default=None, help='checkpoint path')
    parser.add_argument('--ckpt_save_interval', type=int, default=1,
                        help='save ckpt frequency, unit is epoch')
    parser.add_argument('--log_print_interval', type=int, default=10,
                        help='save ckpt frequency, unit is epoch')
    parser.add_argument('--seed', type=int, default=2022,
                        help='set seed of training (Default: 2022)')
    args = parser.parse_args()
    ms.common.set_seed(args.seed)
    cfg = merge(args, load_config(args.config))
    pprint(cfg)
    # if not exists 'ckpt_dir', make it
    cfg.ckpt_dir = os.path.join(cfg.output_path, 'ckpt')
    if not os.path.exists(cfg.output_path):
        os.makedirs(cfg.output_path, exist_ok=True)
    if not os.path.exists(cfg.ckpt_dir):
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

    context.set_context(mode=context.GRAPH_MODE,
                        save_graphs=False,
                        device_target=cfg.device_target)
    if cfg.distribute:
        init()
        cfg.device_num = get_group_size()
        cfg.rank_id = get_rank()
        context.set_auto_parallel_context(gradients_mean=True,
                                          device_num=cfg.device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL)

        if cfg.device_target == 'Ascend':
            context.set_context(device_id=cfg.device_id)
    else:
        cfg.device_num = 1
        cfg.rank_id = 0
        context.set_context(device_id=cfg.device_id)

    return cfg


def main():
    args = preLauch()

    # create network
    modnet = MODNet(3, backbone_pretrained=args['backbone_pretrained'])

    if args.ckpt_path:
        param_dict = load_checkpoint(args.ckpt_path)
        load_param_into_net(modnet, param_dict)

    # create dataset
    dataset = create_dataset(args.train_dataset, usage='train')
    dataset_size = dataset.get_dataset_size()

    # lr
    lr = []
    step_size = dataset_size
    for i in range(0, args.epoch_size):
        cur_lr = args.lr / (2 ** ((i + 1) // 10))
        lr.extend([cur_lr] * step_size)
    optim = nn.SGD(modnet.trainable_params(), learning_rate=lr, momentum=0.9)

    # create WithLossCell
    net_loss = MODNetLossCell(modnet)
    model = ms.Model(network=net_loss, optimizer=optim, amp_level="O2")
    time_cb = ms.TimeMonitor(data_size=step_size)
    loss_cb = ms.LossMonitor()
    cb = [time_cb, loss_cb]
    config_ck = ms.CheckpointConfig(save_checkpoint_steps=dataset_size, keep_checkpoint_max=20)
    ckpt_cb = ms.ModelCheckpoint(prefix="MODNet", directory=args.ckpt_dir, config=config_ck)
    if args.rank_id == 0:
        cb += [ckpt_cb]

    model.train(epoch=args.epoch_size, train_dataset=dataset, callbacks=cb, dataset_sink_mode=True)


if __name__ == '__main__':
    print("train begins------------------------------")
    train_start_time = time.time()
    main()
    train_end_time = time.time()
    print("End of the training------------------------------")
    print('Total tain time: {} s'.format(train_end_time-train_start_time))
