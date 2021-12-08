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
"""train"""
from __future__ import division
import sys
import os
import time
import glob
import argparse
import ast
import yaml
import h5py
import numpy as np
import mindspore
from mindspore import Tensor, context, Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, Callback
from mindspore.nn.dynamic_lr import piecewise_constant_lr as pc_lr
from mindspore.nn.dynamic_lr import warmup_lr
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.context import ParallelMode
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication import get_rank, get_group_size


def parse_arguments(argv):
    """receive arguments"""
    parser = argparse.ArgumentParser()
    # Ascend device
    parser.add_argument('--device_target', default='Ascend',
                        help='device where the code will be implemented')
    parser.add_argument('--data_url', required=False, default=None, help='Location of data')
    parser.add_argument('--pre_trained', required=False, default=None, help='Ckpt file path')
    parser.add_argument('--run_distribute', type=ast.literal_eval, required=False, default=None,
                        help='If run distributed')
    # gpu device
    parser.add_argument('--config', type=str, default="src/Sony_config.yaml",
                        help='Directory of config.')
    parser.add_argument('--device_num', type=int, default=1,
                        help='number of GPUs')

    return parser.parse_args(argv)


class StepLossTimeMonitor(Callback):
    """Monitor"""
    def __init__(self, batch_size, per_print_times=1):
        """init"""
        super(StepLossTimeMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.batch_size = batch_size

    def step_begin(self, run_context):
        """runs on step begin"""
        self.step_time = time.time()

    def step_end(self, run_context):
        """runs on step end"""
        step_seconds = time.time() - self.step_time
        step_fps = self.batch_size * 1.0 / step_seconds

        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        self.losses.append(loss)
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print("step: {}, loss is {}, fps is {}".format(cur_step_in_epoch, loss, step_fps))

    def epoch_begin(self, run_context):
        """runs on epoch begin"""
        self.epoch_start = time.time()
        self.losses = []

    def epoch_end(self, run_context):
        """runs on epoch end"""
        cb_params = run_context.original_args()
        epoch_cost = time.time() - self.epoch_start
        step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        step_fps = self.batch_size * 1.0 * step_in_epoch / epoch_cost
        print("epoch: {:3d}, avg loss:{:.4f}, total cost: {:.3f} s, per step fps:{:5.3f}".format(
            cb_params.cur_epoch_num, np.mean(self.losses), epoch_cost, step_fps))


def train_gpu(args):
    from mindspore.communication import init
    from src.unet import UNet
    from src.data_utils import Sony

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.device_num > 1:
        init()
        mindspore.context.set_auto_parallel_context(device_num=args.device_num,
                                                    parameter_broadcast=True,
                                                    parallel_mode=ParallelMode.DATA_PARALLEL,
                                                    gradients_mean=True)
        rank = get_rank()

    else:
        mindspore.context.set_context(mode=config["mode"], device_target=config["device"],
                                      device_id=config["device_id"])
        rank = 0

    # dataset
    sony = Sony(config["input_dir"], config["gt_dir"])
    if args.device_num > 1:
        rank_id = get_rank()
        rank_size = get_group_size()
        dataset = ds.GeneratorDataset(sony, ["data", "label"], shuffle=False, num_shards=rank_size, shard_id=rank_id)
        dataset = dataset.batch(batch_size=config["batch_size"], drop_remainder=True)
    else:
        dataset = ds.GeneratorDataset(sony, ["data", "label"], shuffle=False)
        dataset = dataset.batch(batch_size=config["batch_size"], drop_remainder=True)

    # model
    unet = UNet()

    # loss
    loss = nn.L1Loss()
    # opt
    optimizer = nn.Adam(params=unet.trainable_params(), learning_rate=config["lr"])

    # call back
    ckpt_config = CheckpointConfig(save_checkpoint_steps=len(sony),
                                   keep_checkpoint_max=10)
    print("checkpoint dir: ", config["checkpoint_dir"]+'ckpt_{}/'.format(rank))
    ckpoint_cb = ModelCheckpoint(prefix='ckpt_sony_ms_adam',
                                 directory=config["checkpoint_dir"]+'ckpt_{}/'.format(rank),
                                 config=ckpt_config)

    callbacks = [StepLossTimeMonitor(batch_size=config["batch_size"], per_print_times=1), ckpoint_cb]

    # train
    model = Model(unet, loss_fn=loss, optimizer=optimizer)
    model.train(config["epochs"], dataset, callbacks=callbacks)


def dynamic_lr(steps_per_epoch, warmup_epochss):   # if warmup, plus warmup_epochs
    """ learning rate with warmup"""
    milestone = [(1200 + warmup_epochss) * steps_per_epoch,
                 (1300 + warmup_epochss) * steps_per_epoch,
                 (1700 + warmup_epochss) * steps_per_epoch,
                 (2500 + warmup_epochss) * steps_per_epoch]
    learning_rates = [3e-4, 1e-5, 3e-6, 1e-6]
    lrs = pc_lr(milestone, learning_rates)
    return lrs


def RandomCropAndFlip(image, label):
    """ random crop and flip """
    ps = 512
    # random crop
    h = image.shape[1]
    w = image.shape[2]
    xx = np.random.randint(0, h - ps)
    yy = np.random.randint(0, w - ps)
    image = image[:, xx:xx + ps, yy:yy + ps]
    label = label[:, xx * 2:xx * 2 + ps * 2, yy * 2:yy * 2 + ps * 2]
    # random flip
    if np.random.randint(2) == 1:  # random flip
        image = np.flip(image, axis=1)
        label = np.flip(label, axis=1)
    if np.random.randint(2) == 1:
        image = np.flip(image, axis=2)
        label = np.flip(label, axis=2)
    if np.random.randint(2) == 1:  # random transpose
        image = np.transpose(image, (0, 2, 1))
        label = np.transpose(label, (0, 2, 1))
    image = np.minimum(image, 1.0)

    return image, label


def train_ascend(args):
    from mindspore.communication.management import init
    from mindspore.nn.loss import L1Loss
    from src.myutils import pack_raw
    from src.unet_parts import UNet
    from src.configs import config
    from src.myutils import GNMTTrainOneStepWithLossScaleCell, WithLossCell

    def get_dataset(input_dir1, gt_dir1, train_ids1, num_shards=None, shard_id=None, distribute=False):
        """ get mindspore dataset from raw data """
        input_final_data = []
        gt_final_data = []
        for train_id in train_ids1:
            in_files = glob.glob(input_dir1 + '%05d_00*.hdf5' % train_id)

            gt_files = glob.glob(gt_dir1 + '%05d_00*.hdf5' % train_id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)
            gt_exposure = float(gt_fn[9: -6])
            gt = h5py.File(gt_path, 'r')
            gt_rawed = gt.get('gt')[:]
            gt_image = np.expand_dims(np.float32(gt_rawed / 65535.0), axis=0)
            gt_image = gt_image.transpose([0, 3, 1, 2])

            for in_path in in_files:
                gt_final_data.append(gt_image[0])

                in_fn = os.path.basename(in_path)
                in_exposure = float(in_fn[9: -6])
                ratio = min(gt_exposure / in_exposure, 300)
                im = h5py.File(in_path, 'r')
                in_rawed = im.get('in')[:]
                input_image = np.expand_dims(pack_raw(in_rawed), axis=0) * ratio
                input_image = np.float32(input_image)
                input_image = input_image.transpose([0, 3, 1, 2])
                input_final_data.append(input_image[0])
        data = (input_final_data, gt_final_data)
        if distribute:
            datasets = ds.NumpySlicesDataset(data, ['input', 'label'], shuffle=False,
                                             num_shards=num_shards, shard_id=shard_id)
        else:
            datasets = ds.NumpySlicesDataset(data, ['input', 'label'], shuffle=False)
        return datasets

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    if args.run_distribute:
        device_num = int(os.getenv('RANK_SIZE'))
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=device_id)
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()

    local_data_path = args.data_url

    input_dir = os.path.join(local_data_path, 'short/')
    gt_dir = os.path.join(local_data_path, 'long/')

    train_fns = glob.glob(gt_dir + '0*.hdf5')
    train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

    net = UNet(4, 12)
    net_loss = L1Loss()
    net = WithLossCell(net, net_loss)
    if args.run_distribute:
        dataset = get_dataset(input_dir, gt_dir, train_ids,
                              num_shards=device_num, shard_id=device_id, distribute=True)
    else:
        dataset = get_dataset(input_dir, gt_dir, train_ids)
    transform_list = [RandomCropAndFlip]
    dataset = dataset.map(transform_list, input_columns=['input', 'label'], output_columns=['input', 'label'])
    dataset = dataset.shuffle(buffer_size=161)
    dataset = dataset.batch(batch_size=config.batch_size, drop_remainder=True)
    batches_per_epoch = dataset.get_dataset_size()

    lr_warm = warmup_lr(learning_rate=3e-4, total_step=config.warmup_epochs * batches_per_epoch,
                        step_per_epoch=batches_per_epoch, warmup_epoch=config.warmup_epochs)
    lr = dynamic_lr(batches_per_epoch, config.warmup_epochs)
    lr = lr_warm + lr[config.warmup_epochs:]
    net_opt = nn.Adam(net.trainable_params(), lr)
    scale_manager = DynamicLossScaleManager()
    net = GNMTTrainOneStepWithLossScaleCell(net, net_opt, scale_manager.get_update_cell())

    ckpt_dir = args.pre_trained
    if ckpt_dir is not None:
        param_dict = load_checkpoint(ckpt_dir)
        load_param_into_net(net, param_dict)
    model = Model(net)

    loss_cb = LossMonitor()
    time_cb = TimeMonitor(data_size=4)
    config_ck = CheckpointConfig(save_checkpoint_steps=100 * batches_per_epoch, keep_checkpoint_max=100)
    ckpoint_cb = ModelCheckpoint(prefix='sony_trained_net', directory=config.train_output_dir, config=config_ck)
    callbacks_list = [ckpoint_cb, loss_cb, time_cb]
    model.train(epoch=config.total_epochs, train_dataset=dataset,
                callbacks=callbacks_list,
                dataset_sink_mode=True)


def train(args):
    if args.device_target == 'Ascend':
        train_ascend(args)
    elif args.device_target == 'GPU':
        train_gpu(args)


if __name__ == "__main__":
    train(parse_arguments(sys.argv[1:]))
