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
"""train MultiTaskNet"""
import os
import ast
import random
import argparse
import time
import numpy as np
import mindspore
from mindspore.common import set_seed
from mindspore import dataset as de
from mindspore import context, Tensor
from mindspore.communication.management import init, get_rank
from mindspore.train.model import Model, ParallelMode
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from src.model.DenseNet import DenseNet121
from src.loss.loss import netWithLossCell
from src.lr_scheduler.lr_scheduler import get_lr
from src.dataset.dataset import create_dataset, eval_create_dataset, _get_rank_info
from src.optimizers.optimizers import init_optim
from src.utils.save_callback import SaveCallback

parser = argparse.ArgumentParser(description='Train MultiTaskNet')

parser.add_argument('--device_target', type=str, default="Ascend", help='Device target')
parser.add_argument('--distribute', type=ast.literal_eval, default=True)
parser.add_argument('--pre_ckpt_path', type=str, default='densenet121_pretrain.ckpt')
parser.add_argument('--isModelArts', type=ast.literal_eval, default=False)
parser.add_argument('--train_url', type=str)
parser.add_argument('--data_url', type=str)

# Datasets
parser.add_argument('--root', type=str, default='', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='veri', help="name of the dataset")
parser.add_argument('--height', type=int, default=256, help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=256, help="width of an image (default: 256)")

# Optimization options
parser.add_argument('--amp-level', type=str, default='O0', help="choose O0 or O3")
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm")
parser.add_argument('--max-epoch', default=10, type=int, help="maximum epochs to run")
parser.add_argument('--train-batch', default=32, type=int, help="train batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0007, type=float, help="initial learning rate")
parser.add_argument('--stepsize', default=[30, 60], nargs='+', type=int, help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float, help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--htri-only', action='store_true', default=False, help="whether use only htri loss in training")
parser.add_argument('--lambda-xent', type=float, default=1, help="cross entropy loss weight")
parser.add_argument('--lambda-htri', type=float, default=1, help="hard triplet loss weight")
parser.add_argument('--lambda-vcolor', type=float, default=0.125, help="vehicle color classification weight")
parser.add_argument('--lambda-vtype', type=float, default=0.125, help="vehicle type classification weight")
parser.add_argument('--label-smooth', type=ast.literal_eval, default=False, help="label smoothing")

# Architecture
parser.add_argument('--heatmapaware', type=ast.literal_eval, default=False, help="embed heatmaps to images")
parser.add_argument('--segmentaware', type=ast.literal_eval, default=False, help="embed segments to images")

args = parser.parse_args()

if args.isModelArts:
    import moxing as mox

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

if __name__ == '__main__':
    set_seed(1)
    target = args.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)

    if args.distribute:
        init()
        device_num = int(os.getenv('RANK_SIZE', '1'))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True, device_num=device_num)

    else:
        device_id = int(os.getenv('DEVICE_ID', '0'))
        context.set_context(device_id=device_id)


    if args.isModelArts:
        mox.file.copy_parallel(src_url=args.data_url, dst_url='/cache/dataset/device_' + os.getenv('DEVICE_ID'))
        train_dataset_path = '/cache/dataset/device_' + os.getenv('DEVICE_ID')
    else:
        train_dataset_path = args.root

    # dataset
    dataset, num_train_vids, num_train_vcolors, num_train_vtypes = create_dataset(dataset_dir=args.dataset,
                                                                                  root=train_dataset_path,
                                                                                  width=args.width,
                                                                                  height=args.height,
                                                                                  keyptaware=True,
                                                                                  heatmapaware=args.heatmapaware,
                                                                                  segmentaware=args.segmentaware,
                                                                                  train_batch=args.train_batch)
    query_dataloader, gallery_dataloader, _, _, _, _vcolor2label, \
        _vtype2label = eval_create_dataset(dataset_dir=args.dataset,
                                           root=train_dataset_path,
                                           width=args.width,
                                           height=args.height,
                                           keyptaware=True,
                                           heatmapaware=args.heatmapaware,
                                           segmentaware=args.segmentaware,
                                           train_batch=args.train_batch)

    step_size = dataset.get_dataset_size()

    # model
    if args.isModelArts:
        pre_path = train_dataset_path + '/' + args.pre_ckpt_path
    else:
        pre_path = args.pre_ckpt_path

    _model = DenseNet121(pretrain_path=pre_path,
                         num_vids=num_train_vids,
                         num_vcolors=num_train_vcolors,
                         num_vtypes=num_train_vtypes,
                         keyptaware=True,
                         heatmapaware=args.heatmapaware,
                         segmentaware=args.segmentaware,
                         multitask=True)
    test_model = DenseNet121(pretrain_path=pre_path,
                             num_vids=num_train_vids,
                             num_vcolors=num_train_vcolors,
                             num_vtypes=num_train_vtypes,
                             keyptaware=True,
                             heatmapaware=args.heatmapaware,
                             segmentaware=args.segmentaware,
                             multitask=True)

    net_with_loss = netWithLossCell(_model,
                                    label_smooth=args.label_smooth,
                                    keyptaware=True,
                                    multitask=True,
                                    htri_only=args.htri_only,
                                    lambda_xent=args.lambda_xent,
                                    lambda_htri=args.lambda_htri,
                                    lambda_vcolor=args.lambda_vcolor,
                                    lambda_vtype=args.lambda_vtype,
                                    margin=args.margin,
                                    num_train_vids=num_train_vids,
                                    num_train_vcolors=num_train_vcolors,
                                    num_train_vtypes=num_train_vtypes,
                                    batch_size=args.train_batch)

    lr = get_lr(lr=args.lr,
                total_epochs=args.max_epoch,
                steps_per_epoch=step_size,
                lr_step=args.stepsize,
                gamma=args.gamma)
    lr = Tensor(lr, mindspore.float32)
    decayed_params = []
    no_decayed_params = []

    for param in _model.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)
    group_params = [{'params': decayed_params, 'weight_decay': args.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': _model.trainable_params()}]

    manager = FixedLossScaleManager(64, drop_overflow_update=False)

    loss_scale = 64 if target == 'Ascend' else 1
    optimizer = init_optim(args.optim, group_params, lr, args.weight_decay, loss_scale)

    model = Model(net_with_loss, optimizer=optimizer, loss_scale_manager=manager, amp_level=args.amp_level)

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    device_num, device_id = _get_rank_info()

    config_ck = CheckpointConfig(save_checkpoint_steps=1 * step_size, keep_checkpoint_max=args.max_epoch)

    if args.isModelArts:
        save_checkpoint_path = '/cache/train_output/device_' + os.getenv('DEVICE_ID') + '/'
    elif not args.distribute:
        save_checkpoint_path = './ckpt/'
    else:
        save_checkpoint_path = './ckpt_' + str(get_rank()) + '/'

    ckpt_cb = ModelCheckpoint(prefix="MultipleNet",
                              directory=save_checkpoint_path,
                              config=config_ck)
    save_cb = SaveCallback(test_model, query_dataloader, gallery_dataloader, \
            _vcolor2label, _vtype2label, 1, args.max_epoch, save_checkpoint_path, step_size, device_id)
    cb += [ckpt_cb, save_cb]

    print("##################### heatmapaware is #####################:{}".format(args.heatmapaware))
    print("##################### segmentaware is #####################:{}".format(args.segmentaware))
    print("##################### batch size is #####################:{}".format(args.train_batch))
    print("######################## start train ########################")
    begin = time.time()
    model.train(args.max_epoch, dataset, callbacks=cb, dataset_sink_mode=False)
    end = time.time()
    print('total time: ', str(end-begin))
    if args.isModelArts:
        mox.file.copy_parallel(src_url='/cache/train_output', dst_url=args.train_url)
