# coding: utf-8
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
''' fcn-4 train'''

import os
import argparse
import glob
import moxing as mox
import numpy as np
from mindspore.train.serialization import export
from mindspore import Tensor, context, nn
import mindspore.dataset as ds
from mindspore.train import Model
from mindspore.common import set_seed
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from src.musictagger import MusicTaggerCNN
from src.loss import BCELoss

from modelarts.utils.convert2mindrecord import prepare_train_data

set_seed(1)

def export_fcn4(config):
    """ export_fcn4 """
    net = MusicTaggerCNN(in_classes=[1, 128, 384, 768, 2048],
                         kernel_size=[3, 3, 3, 3, 3],
                         padding=[0] * 5,
                         maxpool=[(2, 4), (4, 5), (3, 8), (4, 8)],
                         has_bias=True)
    # load checkpoint
    prob_ckpt_list = os.path.join(config.checkpoint_path, "{}*.ckpt".format(config.prefix))
    ckpt_list = glob.glob(prob_ckpt_list)
    if not ckpt_list:
        print('Freezing model failed!')
        print("can not find ckpt files. ")
    else:
        ckpt_list.sort(key=os.path.getmtime)
        ckpt_name = ckpt_list[-1]
        print("checkpoint file name", ckpt_name)
        _param_dict = load_checkpoint(ckpt_name)
        load_param_into_net(network, _param_dict)
        net.set_train(False)

        image = Tensor(np.random.uniform(0.0, 1.0, size=[config.batch_size, 1, 96, 1366]).astype(np.float32))
        export_path = os.path.join(config.output_path, config.export_path)
        if not os.path.exists(export_path):
            os.makedirs(export_path, exist_ok=True)
        file_name = os.path.join(config.output_path, config.export_path, config.file_name)
        export(net, image, file_name=file_name, file_format=config.file_format)
        print('Freezing model success!')
    return 0

def create_dataset(base_path, filename, batch_size, columns_list,
                   num_consumer):
    """Create dataset"""

    path = os.path.join(base_path, filename)
    dtrain = ds.MindDataset(path, columns_list, num_consumer)
    dtrain = dtrain.shuffle(buffer_size=dtrain.get_dataset_size())
    dtrain = dtrain.batch(batch_size, drop_remainder=True)

    return dtrain

def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)

def train(model, dataset_direct, filename, columns_list, num_consumer=4,
          batch=16, epoch=50, save_checkpoint_steps=2172, keep_checkpoint_max=50,
          prefix="model", directory='./'):
    """
    train network
    """
    config_ck = CheckpointConfig(save_checkpoint_steps=save_checkpoint_steps,
                                 keep_checkpoint_max=keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix=prefix,
                                 directory=directory,
                                 config=config_ck)
    data_train = create_dataset(dataset_direct, filename, batch, columns_list,
                                num_consumer)

    model.train(epoch, data_train, callbacks=[ckpoint_cb, \
        LossMonitor(per_print_times=181), TimeMonitor()], dataset_sink_mode=True)

def _parse_args():
    """ parser arguments """
    parser = argparse.ArgumentParser('fcn-4 training args')
    # url for modelarts
    parser.add_argument('--train_url', type=str, default='',
                        help='where training log and ckpts saved')
    parser.add_argument('--data_url', type=str, default='',
                        help='path of dataset')
    # container path
    parser.add_argument('--data_path', type=str, default='/cache/data',
                        help='path of dataset')
    parser.add_argument('--output_path', type=str, default='/cache/train',
                        help='training output path')
    parser.add_argument('--pre_trained', type=str, default='False',
                        help='whether training based on the pre-trained model')
    parser.add_argument('--load_path', type=str, default='/cache/data/pretrained_ckpt/',
                        help='path to load pretrained checkpoint')
    parser.add_argument('--model_name', type=str, default='MusicTagger-10_543.ckpt',
                        help=' pretrained checkpoint file name')
    parser.add_argument('--checkpoint_path', type=str, default='/cache/train/checkpoint',
                        help=' the path to save checkpoint')

    # config of data
    parser.add_argument('--num_classes', type=int, default=50,
                        help='number of tagging classes')
    parser.add_argument('--num_consumer', type=int, default=4,
                        help='file number for mindrecord')
    parser.add_argument('--npy_path', type=str, default='Music_Tagger_Data/npy_path',
                        help='path to numpy')
    parser.add_argument('--info_path', type=str, default='Music_Tagger_Data/config',
                        help='path to info_name, which provide the label of each audio clips')
    parser.add_argument('--info_name', type=str, default='annotations_final.csv',
                        help='info name')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device ID used to train or evaluate the dataset. ')
    parser.add_argument('--mr_path', type=str, default='/cache/data/Music_Tagger_Data/mindrecord_path',
                        help='directory of mindrecord data')
    parser.add_argument('--mr_name_train', type=str, default='train',
                        help='mindrecord name of training data')
    parser.add_argument('--mr_name_val', type=str, default='val',
                        help='mindrecord name of validation data')
    # # config of music tagger
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='training batch size')
    parser.add_argument('--epoch_size', type=int, default=2,
                        help='total training epochs')
    parser.add_argument('--loss_scale', type=float, default=1024.0,
                        help='loss scale')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='if use mix precision calculation')
    parser.add_argument('--train_filename', type=str, default='train.mindrecord0',
                        help='file name of the train mindrecord data')
    parser.add_argument('--val_filename', type=str, default='val.mindrecord0',
                        help='file name of the evaluation mindrecord data')
    parser.add_argument('--keep_checkpoint_max', type=int, default=10,
                        help='only keep the last keep_checkpoint_max checkpoint')
    parser.add_argument('--save_step', type=int, default=10,
                        help='steps for saving checkpoint')
    parser.add_argument('--prefix', type=str, default='MusicTagger',
                        help='prefix of checkpoint')
    # export config
    parser.add_argument('--export_path', type=str, default='export_path',
                        help='path to export model')
    parser.add_argument('--file_name', type=str, default='fcn-4',
                        help='file name of export model')
    parser.add_argument('--file_format', type=str, default='AIR',
                        help='file format for model frozen')

    _args = parser.parse_args()

    return _args

if __name__ == "__main__":
    cfg_args = _parse_args()
    # create local path
    if not os.path.exists(cfg_args.output_path):
        os.makedirs(cfg_args.output_path, exist_ok=True)
    if not os.path.exists(os.path.join(cfg_args.output_path, cfg_args.checkpoint_path)):
        os.makedirs(cfg_args.checkpoint_path, exist_ok=True)
    if not os.path.exists(cfg_args.data_path):
        os.makedirs(cfg_args.data_path, exist_ok=True)
    if not os.path.exists(cfg_args.mr_path):
        os.makedirs(cfg_args.mr_path, exist_ok=True)
    # download data from obs
    mox.file.copy_parallel(cfg_args.data_url, cfg_args.data_path)
    # prepare training data in mindrecord format
    path1 = os.path.join(cfg_args.data_path, cfg_args.info_path)
    path2 = os.path.join(cfg_args.data_path, cfg_args.npy_path)
    prepare_train_data(path1, path2, cfg_args.mr_path, cfg_args.num_classes)
    print("-"*15, "check if data well prepared", "-"*15)
    for root, dirs, files in os.walk(cfg_args.mr_path, topdown=True):
        for name in files:
            print(os.path.join(root, name))
        for name in dirs:
            print(os.path.join(root, name))

    # set context
    context.set_context(device_target='Ascend', mode=context.GRAPH_MODE, device_id=get_device_id())
    context.set_context(enable_auto_mixed_precision=cfg_args.mixed_precision)

    # define model
    network = MusicTaggerCNN(in_classes=[1, 128, 384, 768, 2048],
                             kernel_size=[3, 3, 3, 3, 3],
                             padding=[0] * 5,
                             maxpool=[(2, 4), (4, 5), (3, 8), (4, 8)],
                             has_bias=True)
    # load ckpt
    if cfg_args.pre_trained == "True":
        param_dict = load_checkpoint(os.path.join(cfg_args.data_path, cfg_args.load_path, cfg_args.model_name))
        load_param_into_net(network, param_dict)
    net_loss = BCELoss()
    network.set_train(True)
    net_opt = nn.Adam(params=network.trainable_params(),
                      learning_rate=cfg_args.lr,
                      loss_scale=cfg_args.loss_scale)
    loss_scale_manager = FixedLossScaleManager(loss_scale=cfg_args.loss_scale,
                                               drop_overflow_update=False)
    net_model = Model(network, net_loss, net_opt, loss_scale_manager=loss_scale_manager)
    train(model=net_model,
          dataset_direct=cfg_args.mr_path,
          filename=cfg_args.train_filename,
          columns_list=['feature', 'label'],
          num_consumer=cfg_args.num_consumer,
          batch=cfg_args.batch_size,
          epoch=cfg_args.epoch_size,
          save_checkpoint_steps=cfg_args.save_step,
          keep_checkpoint_max=cfg_args.keep_checkpoint_max,
          prefix=cfg_args.prefix,
          directory=cfg_args.checkpoint_path)
    print("train success")
    # export
    export_fcn4(cfg_args)
    # synchronizing data
    mox.file.copy_parallel(cfg_args.output_path, cfg_args.train_url)
