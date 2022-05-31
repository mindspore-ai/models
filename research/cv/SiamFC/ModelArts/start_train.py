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
"""start train """
import sys
import os
import pickle
import argparse
import lmdb
import numpy as np
import moxing as mox
import mindspore as ms
from mindspore.common import set_seed
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore import context
from mindspore.context import ParallelMode
import mindspore.dataset as ds
from mindspore import nn
from mindspore.train import Model
from mindspore import Tensor
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
import mindspore.dataset.transforms as data_trans
from mindspore.train.serialization import load_checkpoint, export, load_param_into_net
from src.config import config
from src.create_lmdb import create_lmdb
from src.alexnet import SiameseAlexNet
from src.dataset import ImagnetVIDDataset
from src.custom_transforms import ToTensor, RandomStretch, RandomCrop, CenterCrop
sys.path.append(os.getcwd())


def obs_data2modelarts(args):
    """copy code to modelarts"""
    if not mox.file.exists(args.modelarts_data_dir):
        mox.file.make_dirs(args.modelarts_data_dir)
    print("===>>>Copy files from obs:{} to modelarts dir:{}".format(
        args.data_url, args.modelarts_data_dir))
    mox.file.copy_parallel(src_url=args.data_url,
                           dst_url=args.modelarts_data_dir)
    print("===>>>Copy from obs to modelarts")
    files = os.listdir(args.modelarts_data_dir)
    print("===>>>Files:", files)


def modelarts_result2obs(args):
    """
    Copy result data from modelarts to obs.
    """
    files = os.listdir(args.modelarts_result_dir)
    print("===>>>modelarts current Files:", files)
    mox.file.copy(src_url=os.path.join(ARGS.modelarts_result_dir, 'SiamFC-1_6650.ckpt'),
                  dst_url=os.path.join(args.obs_result_dir, 'SiamFC-1_6650.ckpt'))
    mox.file.copy(src_url=os.path.join(ARGS.modelarts_result_dir, 'models1.air'),
                  dst_url=os.path.join(args.obs_result_dir, 'models1.air'))
    mox.file.copy(src_url=os.path.join(ARGS.modelarts_result_dir, 'models2.air'),
                  dst_url=os.path.join(args.obs_result_dir, 'models2.air'))
    obs_files = os.listdir(args.obs_result_dir)
    print("===>>>obs current Files:", obs_files)


def export_AIR(args):
    """
    get air model
    """
    net1 = SiameseAlexNet(train=False)
    files = os.listdir(args.modelarts_result_dir)
    print("===>>>Files:", files)
    model_path = os.path.join(args.modelarts_result_dir, 'SiamFC-1_6650.ckpt')
    load_param_into_net(net1, load_checkpoint(model_path), strict_load=True)
    net1.set_train(False)
    net2 = SiameseAlexNet(train=False)
    load_param_into_net(net2, load_checkpoint(model_path), strict_load=True)
    net2.set_train(False)
    input_data_exemplar1 = Tensor(np.zeros([1, 3, 127, 127]), ms.float32)
    input_data_instance1 = Tensor(np.zeros(1), ms.float32)
    input_data_exemplar2 = Tensor(np.ones([1, 256, 6, 6]), ms.float32)
    input_data_instance2 = Tensor(np.ones([1, 3, 255, 255]), ms.float32)
    input1 = [input_data_exemplar1, input_data_instance1]
    input2 = [input_data_exemplar2, input_data_instance2]
    file_name_export1 = os.path.join(ARGS.modelarts_result_dir, "models1")
    file_name_export2 = os.path.join(ARGS.modelarts_result_dir, "models2")
    export(net1, *input1, file_name=file_name_export1, file_format="AIR")
    export(net2, *input2, file_name=file_name_export2, file_format="AIR")


def train(args):
    """set train """
    # loading meta data
    data_dir = os.path.join(args.modelarts_data_dir, "ILSVRC_VID_CURATION")
    meta_data_path = os.path.join(data_dir, "meta_data.pkl")
    meta_data = pickle.load(open(meta_data_path, 'rb'))
    all_videos = [x[0] for x in meta_data]

    set_seed(1234)
    random_crop_size = config.instance_size - 2 * config.total_stride
    train_z_transforms = data_trans.Compose([
        RandomStretch(),
        CenterCrop((config.exemplar_size, config.exemplar_size)),
        ToTensor()
    ])
    train_x_transforms = data_trans.Compose([
        RandomStretch(),
        RandomCrop((random_crop_size, random_crop_size),
                   config.max_translate),
        ToTensor()
    ])
    db_open = lmdb.open(data_dir + '.lmdb', readonly=True, map_size=int(50e12))
    # create dataset
    train_dataset = ImagnetVIDDataset(db_open, all_videos, data_dir,
                                      train_z_transforms, train_x_transforms)
    dataset = ds.GeneratorDataset(
        train_dataset, ["exemplar_img", "instance_img"], shuffle=True)
    dataset = dataset.batch(batch_size=args.batch_size, drop_remainder=True)
    # set network
    network = SiameseAlexNet(train=True)
    decay_lr = nn.polynomial_decay_lr(args.lr_start,
                                      args.lr_end,
                                      total_step=args.epoch * config.num_per_epoch,
                                      step_per_epoch=config.num_per_epoch,
                                      decay_epoch=args.epoch,
                                      power=1.0)
    optim = nn.SGD(params=network.trainable_params(),
                   learning_rate=decay_lr,
                   momentum=config.momentum,
                   weight_decay=config.weight_decay)

    loss_scale_manager = DynamicLossScaleManager()
    model = Model(network,
                  optimizer=optim,
                  loss_scale_manager=loss_scale_manager,
                  metrics=None,
                  amp_level='O3')

    modelarts_result_dir = ARGS.modelarts_result_dir
    if not mox.file.exists(modelarts_result_dir):
        print(f"obs_result_dir[{modelarts_result_dir}] not exist!")
        mox.file.make_dirs(modelarts_result_dir)

    config_ck_train = CheckpointConfig(
        save_checkpoint_steps=6650, keep_checkpoint_max=20)
    ckpoint_cb_train = ModelCheckpoint(prefix='SiamFC',
                                       directory=ARGS.modelarts_result_dir,
                                       config=config_ck_train)
    time_cb_train = TimeMonitor(data_size=config.num_per_epoch)
    loss_cb_train = LossMonitor()

    model.train(epoch=args.epoch,
                train_dataset=dataset,
                callbacks=[time_cb_train, ckpoint_cb_train, loss_cb_train],
                dataset_sink_mode=False
                )


if __name__ == '__main__':
    ARGPARSER = argparse.ArgumentParser(description=" SiamFC Train")
    ARGPARSER.add_argument('--device_target',
                           type=str,
                           default="Ascend",
                           choices=['GPU', 'CPU', 'Ascend'],
                           help='the target device to run, support "GPU", "CPU"')
    ARGPARSER.add_argument('--data_url',
                           default="obs://siamfc-mindspore/dataset",
                           type=str,
                           help=" the path of data")
    ARGPARSER.add_argument('--sink_size',
                           type=int, default=-1,
                           help='control the amount of data in each sink')
    ARGPARSER.add_argument('--device_id',
                           type=int, default=0,
                           help='device id of GPU or Ascend')
    ARGPARSER.add_argument('--modelarts_data_dir',
                           type=str, default="/cache/dataset",
                           help='modelart input path')
    ARGPARSER.add_argument('--modelarts_result_dir',
                           type=str, default="/cache/result",
                           help='modelart result path')
    ARGPARSER.add_argument('--obs_result_dir',
                           type=str, default="./output",
                           help='obs result path')
    ARGPARSER.add_argument('--epoch',
                           type=int, default=1,
                           help='epoch number')
    ARGPARSER.add_argument('--lr_start',
                           type=float, default=1e-2,
                           help='start learning rate')
    ARGPARSER.add_argument('--lr_end',
                           type=float, default=1e-5,
                           help='end learning rate')
    ARGPARSER.add_argument('--batch_size',
                           type=int, default=8,
                           help='batch size')
    ARGS = ARGPARSER.parse_args()

    DEVICENUM = int(os.environ.get("DEVICE_NUM", 1))
    DEVICETARGET = ARGS.device_target
    if DEVICETARGET == "Ascend":
        context.set_context(
            mode=context.GRAPH_MODE,
            device_id=ARGS.device_id,
            save_graphs=False,
            device_target=ARGS.device_target)
        if DEVICENUM > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=DEVICENUM,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    Data_DIR = os.path.join(ARGS.modelarts_data_dir, 'ILSVRC_VID_CURATION')
    Output_DIR = ARGS.modelarts_data_dir + '/ILSVRC_VID_CURATION.lmdb'
    Num_Thread = 16
    print("============>>>>lmdb data dir: ", Output_DIR)
    obs_data2modelarts(ARGS)
    create_lmdb(Data_DIR, Output_DIR, Num_Thread)

    # train
    train(ARGS)
    export_AIR(ARGS)
    modelarts_result2obs(ARGS)
