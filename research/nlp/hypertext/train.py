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
"""train file"""
import argparse
import os
from mindspore import load_checkpoint, load_param_into_net, context, Model
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.communication import management as MultiDevice
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank

from src.config import Config
from src.dataset import build_dataset, build_dataloader
from src.hypertext_train import HModelWithLoss, HModelTrainOneStepCell, EvalCallBack
from src.radam_optimizer import RiemannianAdam

parser = argparse.ArgumentParser(description='HyperText Text Classification')
parser.add_argument('--model', type=str, default='HyperText',
                    help='HyperText')
parser.add_argument('--modelPath', default='./output/save.ckpt', type=str, help='save model path')
parser.add_argument('--num_epochs', default=2, type=int, help='num_epochs')
parser.add_argument('--datasetdir', default='./data/iflytek_public', type=str,
                    help='dataset dir iflytek_public tnews_public')
parser.add_argument('--outputdir', default='./output', type=str, help='output dir')
parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
parser.add_argument('--datasetType', default='iflytek', type=str, help='iflytek/tnews')
parser.add_argument('--device', default='GPU', type=str, help='device GPU Ascend')
parser.add_argument("--run_distribute", type=str, default=False, help="run_distribute")
args = parser.parse_args()

config = Config(args.datasetdir, args.outputdir, args.device)

context.set_context(mode=context.GRAPH_MODE, device_target=config.device)
config.num_epochs = int(args.num_epochs)
config.batch_size = int(args.batch_size)
config.outputdir = args.outputdir
if not os.path.exists(config.outputdir):
    os.mkdir(config.outputdir)
if args.datasetType == 'tnews':
    config.useTnews()
else:
    config.useIflyek()
print('start process data ..........')
vocab, train_dataset, dev_dataset, test_dataset = build_dataset(config, use_word=True, min_freq=int(config.min_freq))
config.n_vocab = len(vocab)


def build_train(dataset, eval_data, lr, save_path=None, run_distribute=False):
    """build train"""
    net_with_loss = HModelWithLoss(config)
    net_with_loss.init_parameters_data()
    if save_path is not None:
        parameter_dict = load_checkpoint(save_path)
        load_param_into_net(net_with_loss, parameter_dict)
    if dataset is None:
        raise ValueError("pre-process dataset must be provided")
    optimizer = RiemannianAdam(learning_rate=lr,
                               params=filter(lambda x: x.requires_grad, net_with_loss.get_parameters()))
    net_with_grads = HModelTrainOneStepCell(net_with_loss, optimizer=optimizer)
    net_with_grads.set_train()
    model = Model(net_with_grads)
    print("Prepare to Training....")
    epoch_size = dataset.get_repeat_count()
    print("Epoch size ", epoch_size)
    eval_cb = EvalCallBack(net_with_loss.hmodel, eval_data, config.eval_step,
                           config.outputdir + '/' + 'hypertext_' + config.datasetType + '.ckpt')
    callbacks = [LossMonitor(10), eval_cb, TimeMonitor(50)]
    if run_distribute:
        print(f" | Rank {MultiDevice.get_rank()} Call model train.")
    model.train(epoch=config.num_epochs, train_dataset=dataset, callbacks=callbacks, dataset_sink_mode=False)


def set_parallel_env():
    """set parallel env"""
    context.reset_auto_parallel_context()
    MultiDevice.init()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                      device_num=MultiDevice.get_group_size(),
                                      gradients_mean=True)


def train_single(train_data, dev_data, lr):
    """train single"""
    print("Starting training on single device.")
    data_iter = build_dataloader(train_data, config.batch_size, config.max_length)
    dev_iter = build_dataloader(dev_data, config.batch_size, config.max_length)
    build_train(data_iter, dev_iter, lr, save_path=None, run_distribute=False)


def train_paralle(train_data, dev_data, lr):
    """train paralle"""
    set_parallel_env()
    print("Starting training on multiple devices.")
    data_iter = build_dataloader(train_data, config.batch_size, config.max_length,
                                 rank_size=MultiDevice.get_group_size(),
                                 rank_id=MultiDevice.get_rank(),
                                 shuffle=False)
    dev_iter = build_dataloader(dev_data, config.batch_size, config.max_length,
                                rank_size=MultiDevice.get_group_size(),
                                rank_id=MultiDevice.get_rank(),
                                shuffle=False)
    build_train(data_iter, dev_iter, lr, save_path=None, run_distribute=True)


def run_train(train_data, dev_data, lr, run_distribute):
    """run train"""
    if config.device == "GPU":
        init("nccl")
        config.rank_id = get_rank()
    if run_distribute:
        train_paralle(train_data, dev_data, lr)
    else:
        train_single(train_data, dev_data, lr)


run_train(train_dataset, dev_dataset, config.learning_rate, args.run_distribute)
