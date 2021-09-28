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
'''training model'''
import os
import argparse
import numpy as np
import mindspore
import mindspore.dataset as ds

from mindspore.context import ParallelMode
from mindspore.communication.management import init
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore import context
from mindspore import Model
from mindspore import Tensor
from mindspore.nn.optim import Adam
from mindspore.train.callback import SummaryCollector

from src.hparams import hparams as hps
from src.dataset import ljdataset, Sampler
from src.callback import Monitor, get_lr
from src.tacotron2 import Tacotron2, Tacotron2Loss, NetWithLossClass, TrainStepWrap

np.random.seed(0)
mindspore.common.set_seed(1024)


def prepare_dataloaders(fdir, rank, group_size, args):
    '''prepare dataloaders'''
    dataset = ljdataset(fdir, group_size)
    ds_dataset = ds.GeneratorDataset(dataset,
                                     ['text_padded',
                                      'input_lengths',
                                      'mel_padded',
                                      'gate_padded',
                                      'text_mask',
                                      'mel_mask',
                                      'rnn_mask'],
                                     num_parallel_workers=int(args.workers),
                                     sampler=Sampler(dataset.sample_nums,
                                                     rank,
                                                     group_size),
                                     shard_id=rank,
                                     num_shards=group_size)
    ds_dataset = ds_dataset.batch(hps.batch_size)
    return ds_dataset


def train(args):
    ''' training '''
    if args.is_distributed in "true":
        rank = int(os.getenv("RANK_ID"))
        group_size = int(os.getenv("RANK_SIZE"))
        device_id = int(os.getenv("DEVICE_ID"))
        context.set_context(
            mode=0,
            device_target="Ascend",
            device_id=device_id)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=group_size)
        init()
    else:
        device_id = int(args.device_id)
        context.set_context(
            mode=1,
            device_target="Ascend",
            device_id=device_id,
            reserve_class_name_in_scope=False)
        rank = 0
        group_size = 1
    train_loader = prepare_dataloaders(args.data_dir, rank, group_size, args)
    epoch_num = hps.epoch_num

    steps_per_epoch = train_loader.get_dataset_size()

    learning_rate = get_lr(hps.lr, epoch_num, steps_per_epoch, steps_per_epoch * 30)
    learning_rate = Tensor(learning_rate)

    scale_update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**12,
                                                   scale_factor=2,
                                                   scale_window=1000)
    net = Tacotron2()
    loss_fn = Tacotron2Loss()
    loss_net = NetWithLossClass(net, loss_fn)

    if args.ckpt_dir != '' and not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
        os.chmod(args.ckpt_dir, 0o775)

    resume_epoch = None
    if args.pretrained_model != '':
        resume_epoch = int(args.pretrained_model.split('-')[-1].split('_')[0])
        learning_rate = learning_rate[resume_epoch * steps_per_epoch:]
        param_dict = load_checkpoint(args.pretrained_model)
        load_param_into_net(net, param_dict)
        print(
            'Successfully loading the pretrained model {}'.format(
                args.pretrained_model))

    optimizer = Adam(params=net.trainable_params(), learning_rate=learning_rate)

    train_net = TrainStepWrap(loss_net, optimizer, scale_update_cell)
    train_net.set_train()

    model = Model(train_net)

    if args.is_distributed in 'true':
        ckpt_path = os.path.join(
            args.ckpt_dir,
            'device' + str(device_id) + '/')
        summary_collector = SummaryCollector(
            summary_dir='summary_dir/device{}/'.format(device_id), collect_freq=1)
    else:
        ckpt_path = os.path.join(args.ckpt_dir, 'single/')
        summary_collector = SummaryCollector(
            summary_dir='summary_dir/standalone/', collect_freq=1)
    config_ck = CheckpointConfig(
        save_checkpoint_steps=steps_per_epoch,
        keep_checkpoint_max=epoch_num)
    ckpt_cb = ModelCheckpoint(
        prefix='tacotron2',
        directory=ckpt_path,
        config=config_ck)
    if resume_epoch is None:
        model.train(
            epoch_num,
            train_loader,
            callbacks=[
                Monitor(learning_rate),
                summary_collector,
                ckpt_cb],
            dataset_sink_mode=False)
    else:
        model.train(
            epoch_num - resume_epoch,
            train_loader,
            callbacks=[
                Monitor(learning_rate),
                summary_collector,
                ckpt_cb],
            dataset_sink_mode=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='ljdataset.hdf5',
                        help='directory to load data')
    parser.add_argument(
        '-cd',
        '--ckpt_dir',
        type=str,
        default='mindspore_ckpt',
        help='directory to save checkpoints')
    parser.add_argument('-cp', '--ckpt_pth', type=str, default='',
                        help='path to load checkpoints')
    parser.add_argument(
        '-dist',
        '--is_distributed',
        type=str,
        default='true',
        help='distributed training')
    parser.add_argument('--device_id', type=str, default='3',
                        help='choose device id')
    parser.add_argument(
        '--workers',
        type=str,
        default='8',
        help='num parallel workers')
    parser.add_argument(
        '-p',
        '--pretrained_model',
        type=str,
        default='',
        help='pretrained model path')
    Args = parser.parse_args()

    train(Args)
