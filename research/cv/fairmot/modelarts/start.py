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
"""train fairmot."""
import json
import os
import glob
import numpy as np
from mindspore import context
from mindspore import Tensor, export
from mindspore import dtype as mstype
from mindspore import Model
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore.context import ParallelMode
from mindspore.train.callback import TimeMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.communication.management import init
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.opts import Opts
from src.losses import CenterNetMultiPoseLossCell
from src.backbone_dla_conv import DLASegConv
from src.infer_net import InferNet
from src.fairmot_pose import WithNetCell
from src.fairmot_pose import WithLossCell
from src.utils.lr_schedule import dynamic_lr
from src.utils.jde import JointDataset
from src.utils.callback import LossCallback
import moxing as mox


def train(opt):
    """train fairmot."""
    local_data_path = '/cache/data'
    device_id = int(os.getenv('DEVICE_ID'))
    device_num = int(os.getenv('RANK_SIZE'))
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                        save_graphs=False, max_call_depth=10000)
    context.set_context(device_id=device_id)
    context.set_auto_parallel_context(
        device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
    init()
    local_data_path = os.path.join(local_data_path, str(device_id))
    opt.data_cfg = os.path.join(local_data_path, opt.data_cfg)
    output_path = opt.train_url
    load_path = os.path.join(local_data_path, opt.load_pre_model)
    print('local_data_path:', local_data_path)
    print('mixdata_path:', opt.data_cfg)
    print('output_path:', output_path)
    print('load_path', load_path)
    # data download
    print('Download data.')
    mox.file.copy_parallel(src_url=opt.data_url, dst_url=local_data_path)
    f = open(opt.data_cfg)
    data_config = json.load(f)
    train_set_paths = data_config['train']
    dataset_root = local_data_path
    f.close()
    dataset = JointDataset(
        opt, dataset_root, train_set_paths, (1088, 608), augment=True)
    opt = Opts().update_dataset_info_and_set_heads(opt, dataset)
    if opt.is_modelarts or opt.run_distribute:
        Ms_dataset = ds.GeneratorDataset(dataset, ['input', 'hm', 'reg_mask', 'ind', 'wh', 'reg', 'ids'],
                                         shuffle=True, num_parallel_workers=8,
                                         num_shards=device_num, shard_id=device_id)
    else:
        Ms_dataset = ds.GeneratorDataset(dataset, ['input', 'hm', 'reg_mask', 'ind', 'wh', 'reg', 'ids'],
                                         shuffle=True)
    Ms_dataset = Ms_dataset.batch(
        batch_size=opt.batch_size, drop_remainder=True)
    batch_dataset_size = Ms_dataset.get_dataset_size()
    net = DLASegConv(opt.heads,
                     down_ratio=4,
                     final_kernel=1,
                     last_level=5,
                     head_conv=256)
    net = net.set_train()
    param_dict = load_checkpoint(load_path)
    load_param_into_net(net, param_dict)
    loss = CenterNetMultiPoseLossCell(opt)
    lr = Tensor(dynamic_lr(20, opt.num_epochs, batch_dataset_size),
                mstype.float32)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)
    net_with_loss = WithLossCell(net, loss)
    fairmot_net = nn.TrainOneStepCell(net_with_loss, optimizer)

    # define callback
    loss_cb = LossCallback(opt.batch_size)
    time_cb = TimeMonitor()
    config_ckpt = CheckpointConfig(saved_network=net)
    ckpoint_cb = ModelCheckpoint(prefix='Fairmot_{}'.format(device_id), directory=local_data_path + '/output/ckpt',
                                 config=config_ckpt)
    callbacks = [loss_cb, ckpoint_cb, time_cb]

    # train
    model = Model(fairmot_net)
    model.train(opt.num_epochs, Ms_dataset, callbacks=callbacks)
    export_AIR(local_data_path + "/output/ckpt", opt)
    mox.file.copy_parallel(local_data_path + "/output", output_path)
    mox.file.copy(src_url='fairmot.air', dst_url=output_path+'/fairmot.air')


def export_AIR(ckpt_path, opt):
    """start modelarts export"""
    ckpt_list = glob.glob(ckpt_path + "/Fairmot_*.ckpt")
    if not ckpt_list:
        print("ckpt file not generated.")

    ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1]
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    backbone_net = DLASegConv(opt.heads,
                              down_ratio=4,
                              final_kernel=1,
                              last_level=5,
                              head_conv=256,
                              is_training=True)
    infer_net = InferNet()
    net_ = WithNetCell(backbone_net, infer_net)
    param_dict = load_checkpoint(ckpt_model)
    load_param_into_net(net_, param_dict)
    input_arr = Tensor(np.zeros([1, 3, 608, 1088]), mstype.float32)
    export(net_, input_arr, file_name='fairmot', file_format='AIR')


if __name__ == '__main__':
    opt_ = Opts().parse()
    train(opt_)
