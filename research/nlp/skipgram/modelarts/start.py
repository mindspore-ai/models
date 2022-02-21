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
"""
get word2vec embeddings by running trian.py.
python train.py --device_target=[DEVICE_TARGET]
"""

import argparse
import ast
import os
import glob
import moxing
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import export
from mindspore import context
from mindspore.common import set_seed
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.model import Model
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from src.dataset import DataController
from src.lr_scheduler import poly_decay_lr
from src.skipgram import SkipGram

parser = argparse.ArgumentParser(description='Train SkipGram')
parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                    help='device target, support Ascend and GPU.')
parser.add_argument('--device_id', type=int, default=0, help='device id of GPU or Ascend.')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='run distribute.')
parser.add_argument('--pre_trained', type=str, default=None, help='the pretrained checkpoint file path.')
parser.add_argument('--train_data_dir', type=str, default=None, help='the directory of train data.')
parser.add_argument('--data_url', type=str, default=None, help='obs path of dataset')
parser.add_argument('--train_url', type=str, default=None, help='obs path of output')
parser.add_argument('--data_epoch', type=int, default=10, help='data_epoch.')
parser.add_argument("--file_name", type=str, default="skipgram", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='AIR',
                    help='file format')
parser.add_argument('--lr', type=int, default=1e-3, help='initial learning rate.')
parser.add_argument('--end_lr', type=int, default=1e-4, help='end learning rate.')
parser.add_argument('--train_epoch', type=int, default=1, help='training epoch')
parser.add_argument('--power', type=int, default=1, help='decay rate of learning rate.')
parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
parser.add_argument('--dataset_sink_mode', type=ast.literal_eval, default=False, help='bool.')
parser.add_argument('--emb_size', type=int, default=288, help='embedding size.')
parser.add_argument('--min_count', type=int, default=5, help='min_count')
parser.add_argument('--window_size', type=int, default=5, help='window size of center word.')
parser.add_argument('--neg_sample_num', type=int, default=5,
                    help='number of negative words in negative sampling.')
parser.add_argument('--save_checkpoint_steps', type=int, default=int(5e5),
                    help='step interval between two checkpoints.')
parser.add_argument('--keep_checkpoint_max', type=int, default=15,
                    help='maximal number of checkpoint files.')
parser.add_argument('--temp_dir', type=str, default='/cache/data/temp',
                    help='save files generated during code execution')
parser.add_argument('--ckpt_dir', type=str, default='/cache/data/temp/ckpts',
                    help='directory that save checkpoint files')
parser.add_argument('--ms_dir', type=str, default='/cache/data/temp/ms_dir',
                    help='directory that saves mindrecord data')
parser.add_argument('--w2v_emb_save_dir', type=str, default='/cache/data/temp/w2v_emb',
                    help='directory that saves word2vec embeddings')
parser.add_argument('--eval_data_dir', type=str, default='/cache/data/eval_data',
                    help='directory of evaluating data')

args, unparsed = parser.parse_known_args()
set_seed(1)

if __name__ == '__main__':
    train_data_dir = "/cache/data"
    moxing.file.copy_parallel(src_url=args.data_url, dst_url=train_data_dir)

    print("Set Context...")
    rank_size = int(os.getenv('RANK_SIZE')) if args.run_distribute else 1
    rank_id = int(os.getenv('RANK_ID')) if args.run_distribute else 0
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args.device_target,
                        device_id=args.device_id,
                        save_graphs=False)
    if args.run_distribute:
        init()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=rank_size,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    print('Done.')

    print("Get Mindrecord...")
    ms_dir = args.ms_dir
    print("train_data_dir: ", os.listdir(train_data_dir))
    print("ms_dir: ", os.listdir(ms_dir))
    data_controller = DataController(train_data_dir, ms_dir, args.min_count, args.window_size,
                                     args.neg_sample_num, args.data_epoch, args.batch_size,
                                     rank_size, rank_id)
    dataset = data_controller.get_mindrecord_dataset(col_list=['c_words', 'p_words', 'n_words'])
    print('Done.')

    print("Configure Training Parameters...")
    config_ck = CheckpointConfig(save_checkpoint_steps=args.save_checkpoint_steps,
                                 keep_checkpoint_max=args.keep_checkpoint_max)
    ckpoint = ModelCheckpoint(prefix="w2v", directory=args.ckpt_dir, config=config_ck)
    loss_monitor = LossMonitor(1000)
    time_monitor = TimeMonitor()
    total_step = dataset.get_dataset_size() * args.train_epoch
    print('Total Step:', total_step)
    decay_step = min(total_step, int(2.4e6) // rank_size)
    lrs = Tensor(poly_decay_lr(args.lr, args.end_lr, decay_step, total_step, args.power,
                               update_decay_step=False))

    callbacks = [loss_monitor, time_monitor]
    if rank_id == 0:
        callbacks = [loss_monitor, time_monitor, ckpoint]

    net = SkipGram(data_controller.get_vocabs_size(), args.emb_size)
    if args.pre_trained:
        load_param_into_net(net, load_checkpoint(args.pre_trained))
    optim = nn.Adam(net.trainable_params(), learning_rate=lrs)
    train_net = nn.TrainOneStepCell(network=net, optimizer=optim)
    model = Model(train_net)
    print('Done.')

    print("Train Model...")
    model.train(epoch=args.train_epoch, train_dataset=dataset,
                callbacks=callbacks, dataset_sink_mode=args.dataset_sink_mode)
    print('Done.')

    print("Save Word2Vec Embedding...")
    w2v_emb_save_dir = args.w2v_emb_save_dir
    net.save_w2v_emb(w2v_emb_save_dir, data_controller.id2word)
    print('Done.')

    print("End.")

    # freezing model
    print("Start export")
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.device_target == "Ascend":
        context.set_context(device_id=args.device_id)

    # export model
    load_checkpoint_path = os.path.join(args.ckpt_dir)
    ckpt_models = list(
        sorted(glob.glob(os.path.join(load_checkpoint_path, '*.ckpt')),
               key=os.path.getctime))[-1]
    print('ckpt_models = ', ckpt_models)

    param_dict = load_checkpoint(ckpt_models)
    vocab_size = param_dict['c_emb.embedding_table'].shape[0]
    emb_size = param_dict['c_emb.embedding_table'].shape[1]
    net = SkipGram(vocab_size, emb_size)
    load_param_into_net(net, param_dict)
    center_words = Tensor(np.ones(1, np.int32))
    pos_words = Tensor(np.ones(1, np.int32))
    neg_words = Tensor(np.ones([1, args.neg_sample_num], np.int32))
    export(net, center_words, pos_words, neg_words,
           file_name=args.ckpt_dir + '/skipgram',
           file_format=args.file_format)

    print("End export")
    moxing.file.copy_parallel(src_url=args.temp_dir, dst_url=args.train_url)
