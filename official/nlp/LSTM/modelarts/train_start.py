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

"""
######################## train lstm example ########################
train lstm and get network model files(.ckpt) :
"""

import argparse
import glob
import os
import tarfile
import time

import moxing as mox
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, context, export
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from mindspore.nn.metrics import Accuracy
from mindspore.profiler import Profiler
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.dataset import convert_to_mindrecord, lstm_create_dataset
from src.lr_schedule import get_lr
from src.lstm import SentimentNet
from src.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id, get_job_id

parser = argparse.ArgumentParser(description='Natural Language Processing')

# ModelArts config
parser.add_argument("--enable_modelarts", type=bool, default=True, help="whether training on modelarts, default: True")
parser.add_argument("--data_url", type=str, default="", help="dataset url for obs")
parser.add_argument("--checkpoint_url", type=str, default="", help="checkpoint url for obs")
parser.add_argument("--train_url", type=str, default="", help="training output url for obs")
parser.add_argument("--data_path", type=str, default="/cache/data", help="dataset path for local")
parser.add_argument("--load_path", type=str, default="/cache/checkpoint", help="dataset path for local")
parser.add_argument("--output_path", type=str, default="/cache/train", help="training output path for local")
parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/",
                    help="the path where pre-trained checkpoint file path")
parser.add_argument("--checkpoint_file", type=str, default="./checkpoint/lstm-20_390.ckpt",
                    help="the path where pre-trained checkpoint file name")
parser.add_argument("--device_target", type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
parser.add_argument("--enable_profiling", type=bool, default=False, help="whether enable modelarts profiling")

# LSTM config
parser.add_argument("--num_classes", type=int, default=2, help="output class num")
parser.add_argument("--num_hiddens", type=int, default=128, help="number of hidden unit per layer")
parser.add_argument("--num_layers", type=int, default=2, help="number of network layer")
parser.add_argument("--learning_rate", type=float, default=0.1, help="static learning rate")
parser.add_argument("--dynamic_lr", type=bool, default=False, help="dynamic learning rate")
parser.add_argument("--lr_init", type=float, default=0.05,
                    help="initial learning rate, effective when enable dynamic_lr")
parser.add_argument("--lr_end", type=float, default=0.01, help="end learning rate, effective when enable dynamic_lr")
parser.add_argument("--lr_max", type=float, default=0.1, help="maximum learning rate, effective when enable dynamic_lr")
parser.add_argument("--lr_adjust_epoch", type=int, default=6,
                    help="the epoch interval of adjusting learning rate, effective when enable dynamic_lr")
parser.add_argument("--warmup_epochs", type=int, default=1,
                    help="the epoch interval of warmup, effective when enable dynamic_lr")
parser.add_argument("--momentum", type=float, default=0.9, help="")
parser.add_argument("--num_epochs", type=int, default=20, help="")
parser.add_argument("--batch_size", type=int, default=64, help="")
parser.add_argument("--embed_size", type=int, default=300, help="")
parser.add_argument("--bidirectional", type=bool, default=True, help="whether enable bidirectional LSTM network")
parser.add_argument("--save_checkpoint_steps", type=int, default=7800, help="")
parser.add_argument("--keep_checkpoint_max", type=int, default=10, help="")

# train config
parser.add_argument("--preprocess", type=str, default='false', help="whether to preprocess data")
parser.add_argument("--preprocess_path", type=str, default="./preprocess",
                    help="path where the pre-process data is stored, "
                         "if preprocess set as 'false', you need prepared preprocessed data under data_url")
parser.add_argument("--aclImdb_zip_path", type=str, default="./aclImdb_v1.tar.gz", help="path where the dataset zip")
parser.add_argument("--aclImdb_path", type=str, default="./aclImdb", help="path where the dataset is stored")
parser.add_argument("--glove_path", type=str, default="./glove", help="path where the GloVe is stored")
parser.add_argument("--ckpt_path", type=str, default="./ckpt_lstm/",
                    help="the path to save the checkpoint file")
parser.add_argument("--pre_trained", type=str, default="", help="the pretrained checkpoint file path")
parser.add_argument("--device_num", type=int, default=1, help="the number of using devices")
parser.add_argument("--distribute", type=bool, default=False, help="enable when training with multi-devices")
parser.add_argument("--enable_graph_kernel", type=bool, default=True, help="whether accelerate by graph kernel")

# export config
parser.add_argument("--ckpt_file", type=str, default="./ckpt_lstm/lstm-20_390.ckpt", help="the export ckpt file name")
parser.add_argument("--device_id", type=int, default=0, help="")
parser.add_argument("--file_name", type=str, default="./lstm", help="the export air file name")
parser.add_argument("--file_format", type=str, default="AIR", help="the export file format")

# LSTM Postprocess config
parser.add_argument("--label_dir", type=str, default="", help="")
parser.add_argument("--result_dir", type=str, default="./result_Files", help="")

# Preprocess config
parser.add_argument("--result_path", type=str, default="./preprocess_Result/", help="")

config = parser.parse_args()

set_seed(1)
_global_sync_count = 0
profiler = None
ckpt_save_dir = ''
embedding_table = None


def unzip(file_name, dirs):
    """
    unzip dataset in tar.gz.format
    :param file_name: file to be unzipped
    :param dirs: unzip path
    :return: no return
    """
    if os.path.exists(file_name) != 1:
        raise FileNotFoundError('{} not exist'.format(file_name))

    print('unzip {} start.'.format(file_name))
    t = tarfile.open(file_name)
    t.extractall(path=dirs)
    print('unzip {} end.'.format(file_name))


def frozen_to_air(net, args):
    """
    export trained model with specific format
    :param net: model object
    :param args: frozen arguments
    :return: no return
    """
    param_dict = load_checkpoint(args.get("ckpt_file"))
    load_param_into_net(net, param_dict)
    input_arr = args.get('input_arr')
    export(net, input_arr, file_name=args.get("file_name"), file_format=args.get("file_format"))


def sync_data(from_path, to_path):
    """
    Download data from remote obs to local directory if the first url is remote url and the second one is local path
    Upload data from local directory to remote obs in contrast.
    :param from_path: source path
    :param to_path: target path
    :return: no return
    """
    global _global_sync_count
    sync_lock = "/tmp/copy_sync.lock" + str(_global_sync_count)
    _global_sync_count += 1

    # Each server contains 8 devices as most.
    if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
        print("from path: ", from_path)
        print("to path: ", to_path)
        mox.file.copy_parallel(from_path, to_path)
        print("===finish data synchronization===")
        try:
            os.mknod(sync_lock)
        except IOError:
            print("Failed to create directory")
        print("===save flag===")

    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(1)

    print("Finish sync data from {} to {}.".format(from_path, to_path))


def download_data():
    """
    sync data from data_url, train_url to data_path, output_path
    :return: no return
    """
    if config.enable_modelarts:
        if config.data_url:
            if not os.path.isdir(config.data_path):
                os.makedirs(config.data_path)
                sync_data(config.data_url, config.data_path)
                print("Dataset downloaded: ", os.listdir(config.data_path))
        if config.checkpoint_url:
            if not os.path.isdir(config.load_path):
                os.makedirs(config.load_path)
                sync_data(config.checkpoint_url, config.load_path)
                print("Preload downloaded: ", os.listdir(config.load_path))
        if config.train_url:
            if not os.path.isdir(config.output_path):
                os.makedirs(config.output_path)
            sync_data(config.train_url, config.output_path)
            print("Workspace downloaded: ", os.listdir(config.output_path))

        context.set_context(save_graphs_path=os.path.join(config.output_path, str(get_rank_id())))
        config.device_num = get_device_num()
        config.device_id = get_device_id()
        # create output dir
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)


def upload_data():
    """
    sync data from output_path to train_url
    :return: no return
    """
    if config.enable_modelarts:
        if config.train_url:
            print("Start copy data to output directory.")
            sync_data(config.output_path, config.train_url)
            print("Copy data to output directory finished.")


def modelarts_preprocess():
    """
    add path prefix, modify parameter and sync data
    :return: no return
    """
    print("============== Starting ModelArts Preprocess ==============")
    config.aclImdb_path = os.path.join(config.data_path, config.aclImdb_path)
    config.aclImdb_zip_path = os.path.join(config.data_path, config.aclImdb_zip_path)
    config.glove_path = os.path.join(config.data_path, config.glove_path)

    config.ckpt_path = os.path.join(config.output_path, config.ckpt_path)
    config.ckpt_file = os.path.join(config.output_path, config.ckpt_file)
    config.file_name = os.path.join(config.output_path, config.file_name)

    if config.preprocess == 'true':
        config.preprocess_path = os.path.join(config.output_path, config.preprocess_path)
    else:
        config.preprocess_path = os.path.join(config.data_path, config.preprocess_path)

    # create profiler
    global profiler
    if config.enable_profiling:
        profiler = Profiler()

    # download data from obs
    download_data()

    # unzip dataset zip
    if config.preprocess == 'true':
        unzip(config.aclImdb_zip_path, config.data_path)
    print("============== ModelArts Preprocess finish ==============")


def modelarts_postprocess():
    """
    convert lstm model to AIR format, sync data
    :return: no return
    """
    print("============== Starting ModelArts Postprocess ==============")
    # get trained lstm checkpoint
    ckpt_list = glob.glob(str(ckpt_save_dir) + "/*lstm*.ckpt")
    if not ckpt_list:
        print("ckpt file not generated")
        ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1]

    # export LSTM with AIR format
    export_lstm(ckpt_model)

    # analyse
    if config.enable_profiling and profiler is not None:
        profiler.analyse()

    # upload data to obs
    upload_data()
    print("============== ModelArts Postprocess finish ==============")


def export_lstm(ckpt_model):
    """
    covert ckpt to AIR and export lstm model
    :param ckpt_model: trained checkpoint
    :return: no return
    """
    print("start frozen model to AIR.")
    global embedding_table
    if embedding_table is None:
        print('embedding_table is None, re-generate')
        embedding_table = np.loadtxt(os.path.join(config.preprocess_path, "weight.txt")).astype(np.float32)

        if config.device_target == 'Ascend':
            pad_num = int(np.ceil(config.embed_size / 16) * 16 - config.embed_size)
            if pad_num > 0:
                embedding_table = np.pad(embedding_table, [(0, 0), (0, pad_num)], 'constant')
            config.embed_size = int(np.ceil(config.embed_size / 16) * 16)

    net = SentimentNet(vocab_size=embedding_table.shape[0],
                       embed_size=config.embed_size,
                       num_hiddens=config.num_hiddens,
                       num_layers=config.num_layers,
                       bidirectional=config.bidirectional,
                       num_classes=config.num_classes,
                       weight=Tensor(embedding_table),
                       batch_size=config.batch_size)

    frozen_to_air_args = {"ckpt_file": ckpt_model,
                          "batch_size": config.batch_size,
                          "input_arr": Tensor(
                              np.random.uniform(0.0, 1e5, size=[config.batch_size, 500]).astype(np.int32)),
                          "file_name": config.file_name,
                          "file_format": config.file_format}

    # convert model to AIR format
    frozen_to_air(net, frozen_to_air_args)
    print("Frozen model to AIR finish.")


def train_lstm():
    """
    train lstm
    :return:  no return
    """
    # print train work info
    print(config)
    print('device id:', get_device_id())
    print('device num:', get_device_num())
    print('rank id:', get_rank_id())
    print('job id:', get_job_id())

    # set context
    device_target = config.device_target
    _enable_graph_kernel = config.enable_graph_kernel and device_target == "GPU"
    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        enable_graph_kernel=_enable_graph_kernel,
        graph_kernel_flags="--enable_cluster_ops=MatMul",
        device_target=config.device_target)

    # enable parallel train
    device_num = config.device_num
    rank = 0
    if device_num > 1 or config.distribute:
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        if device_target == "Ascend":
            context.set_context(device_id=get_device_id())
            init()
            rank = get_rank()

        elif device_target == "GPU":
            init()
    else:
        context.set_context(device_id=get_device_id())

    # dataset preprocess
    if config.preprocess == 'true':
        import shutil
        if os.path.exists(config.preprocess_path):
            shutil.rmtree(config.preprocess_path)
        print("============== Starting Data Pre-processing ==============")
        convert_to_mindrecord(config.embed_size, config.aclImdb_path, config.preprocess_path, config.glove_path)
        print("==============    Data Pre-processing End   ==============")

    # prepare train dataset
    print('prepare train dataset with preprocessed data in {}, batch size: {}.'.
          format(config.preprocess_path, config.batch_size))
    ds_train = lstm_create_dataset(config.preprocess_path, config.batch_size, 1, device_num=device_num, rank=rank)
    if ds_train.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")
    print('train dataset prepare finished.')

    # init embedding_table
    print('init embedding table from {}.'.format(os.path.join(config.preprocess_path, "weight.txt")))
    global embedding_table
    embedding_table = np.loadtxt(os.path.join(config.preprocess_path, "weight.txt")).astype(np.float32)
    # DynamicRNN in this network on Ascend platform only support the condition that the shape of input_size
    # and hidden_size is multiples of 16, this problem will be solved later.
    if config.device_target == 'Ascend':
        pad_num = int(np.ceil(config.embed_size / 16) * 16 - config.embed_size)
        if pad_num > 0:
            embedding_table = np.pad(embedding_table, [(0, 0), (0, pad_num)], 'constant')
        config.embed_size = int(np.ceil(config.embed_size / 16) * 16)
    print('init embedding table finished.')

    # set loss function
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # init learning rate
    if config.dynamic_lr:
        print('generate dynamic learning rate start.')
        lr = Tensor(get_lr(global_step=0,
                           lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                           warmup_epochs=config.warmup_epochs,
                           total_epochs=config.num_epochs,
                           steps_per_epoch=ds_train.get_dataset_size(),
                           lr_adjust_epoch=config.lr_adjust_epoch))
        print('generate dynamic learning rate finished.')
    else:
        lr = config.learning_rate

    # init LSTM network
    network = SentimentNet(vocab_size=embedding_table.shape[0],
                           embed_size=config.embed_size,
                           num_hiddens=config.num_hiddens,
                           num_layers=config.num_layers,
                           bidirectional=config.bidirectional,
                           num_classes=config.num_classes,
                           weight=Tensor(embedding_table),
                           batch_size=config.batch_size)

    # load pre-trained model parameter
    if config.pre_trained:
        print('load pre-trained checkpoint from {}.'.format(config.pre_trained))
        load_param_into_net(network, load_checkpoint(config.pre_trained))
        print('load pre-trained checkpoint finished.')

    # init optimizer
    opt = nn.Momentum(network.trainable_params(), lr, config.momentum)

    # wrap LSTM network
    model = Model(network, loss, opt, {'acc': Accuracy()})

    global ckpt_save_dir
    if device_num > 1:
        ckpt_save_dir = os.path.join(config.ckpt_path + "_" + str(get_rank()))
    else:
        ckpt_save_dir = config.ckpt_path

    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="lstm", directory=ckpt_save_dir, config=config_ck)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    loss_cb = LossMonitor()

    print("============== Starting Training ==============")
    if config.device_target == "CPU":
        model.train(config.num_epochs, ds_train, callbacks=[time_cb, ckpoint_cb, loss_cb], dataset_sink_mode=False)
    else:
        model.train(config.num_epochs, ds_train, callbacks=[time_cb, ckpoint_cb, loss_cb])
    print("============== Training Success ==============")


if __name__ == "__main__":
    modelarts_preprocess()
    train_lstm()
    modelarts_postprocess()
