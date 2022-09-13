#Copyright 2022 Huawei Technologies Co., Ltd
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
"""train and export file"""
import argparse
import os
import numpy as np

from mindspore import load_checkpoint, load_param_into_net, context, Model, Tensor
from mindspore.communication import management as MultiDevice
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
from mindspore.nn import Cell
from mindspore.ops import ArgMaxWithValue
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.train.serialization import export

from src.config import Config
from src.dataset import build_dataset, build_dataloader
from src.hypertext import HModel
from src.hypertext_train import HModelWithLoss, HModelTrainOneStepCell, EvalCallBack
from src.radam_optimizer import RiemannianAdam

parser = argparse.ArgumentParser(description='HyperText Text Classification')
parser.add_argument('--data_url', type=str, help='dataset dir iflytek_public tnews_public')
parser.add_argument('--train_url', type=str, help='output dir')
parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
parser.add_argument('--datasetType', type=str, help='iflytek/tnews')
parser.add_argument('--device', default='Ascend', type=str, help='device GPU Ascend')
parser.add_argument('--num_epochs', default=2, type=int, help='num_epochs')
parser.add_argument("--run_distribute", type=str, default=False, help="run_distribute")
args = parser.parse_args()

if args.datasetType == "tnews":
    args.data_url = os.path.join(args.data_url, "tnews_public")
elif args.datasetType == "iflytek":
    args.data_url = os.path.join(args.data_url, "iflytek_public")
else:
    print("Unsupported dataset type....")
    exit()

config = Config(args.data_url, args.train_url, args.device)

if config.device == 'GPU':
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
elif config.device == 'Ascend':
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
config.num_epochs = int(args.num_epochs)
config.batch_size = int(args.batch_size)
config.outputdir = args.train_url
if not os.path.exists(config.outputdir):
    os.mkdir(config.outputdir)
if args.datasetType == 'tnews':
    config.useTnews()
else:
    config.useIflyek()
print('start process data ..........')
vocab, train_dataset, dev_dataset, test_dataset = build_dataset(config, use_word=True, min_freq=int(config.min_freq))
config.n_vocab = len(vocab)


class HyperTextTextInferExportCell(Cell):
    """
    HyperText network infer.
    """

    def __init__(self, network):
        """init fun"""
        super(HyperTextTextInferExportCell, self).__init__(auto_prefix=False)
        self.network = network
        self.argmax = ArgMaxWithValue(axis=1, keep_dims=True)

    def construct(self, x1, x2):
        """construct hypertexttext infer cell"""
        predicted_idx = self.network(x1, x2)
        predicted_idx = self.argmax(predicted_idx)
        return predicted_idx


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


def run_export():
    hmodel = HModel(config)
    file_name = 'hypertext_' + config.datasetType
    param_dict = load_checkpoint(os.path.join(args.train_url, file_name + '.ckpt'))
    load_param_into_net(hmodel, param_dict)
    ht_infer = HyperTextTextInferExportCell(hmodel)
    x1 = Tensor(np.ones((1, config.max_length)).astype(np.int32))
    x2 = Tensor(np.ones((1, config.max_length)).astype(np.int32))
    export(ht_infer, x1, x2, file_name=(args.train_url + file_name), file_format='AIR')


run_train(train_dataset, dev_dataset, config.learning_rate, args.run_distribute)
run_export()
