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
train model
"""
import os
import argparse
import ast
import numpy as np
import moxing
import mindspore.common.dtype as mstype
from mindspore import Tensor, context, load_checkpoint, export
from mindspore.nn.optim import Adam
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.context import ParallelMode
from mindspore.communication.management import init
from mindspore import set_seed

from src.dataset import create_dataset
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id
from src.model import RetrievalWithLoss
from src.bert import BertConfig
from src.lr_schedule import Noam
from src.model import RetrievalWithSoftmax


set_seed(0)

parser = argparse.ArgumentParser(description='Duconv conversion')
parser.add_argument('--data_url', type=str, default=None, help='Location of Data')
parser.add_argument('--train_url', type=str, default='', help='Location of training outputs')
parser.add_argument('--task_name', type=str, default="match_kn_gene", choices=['match', 'match_kn', 'match_kn_gene'],
                    help='choose of task name')
parser.add_argument('--max_seq_length', type=int, default=512, help='the max sequence length')
parser.add_argument('--vocab_size', type=int, default=14373, help='vocab size')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--epoch', type=int, default=30, help='epoch')
parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                    help='device target, support Ascend and GPU.')
parser.add_argument('--device_id', type=int, default=0, help='device id')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='run distribute.')
parser.add_argument('--train_data_shuffle', type=ast.literal_eval, default=True, help='train data shuffle.')
parser.add_argument('--warmup_proportion', type=float, default=0.1, help='warm up proportion.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate.')
parser.add_argument('--save_checkpoint_steps', type=int, default=400, help='save checkpoint steps.')
parser.add_argument('--dataset_sink_mode', type=ast.literal_eval, default=True, help='dataset sink mode.')
parser.add_argument('--rank_save_ckpt_flag', type=ast.literal_eval, default=True, help='rank save ckpt flag.')
parser.add_argument('--save_checkpoint_path', type=str, default="/cache/checkpoint", help='save checkpoint path.')

args, unknown = parser.parse_known_args()

def save_ckpt_to_air(save_ckpt_path, path):
    '''save_ckpt_to_air:convert ckpt to air'''
    use_kn = bool("kn" in args.task_name)
    bertconfig = BertConfig(seq_length=args.max_seq_length, vocab_size=args.vocab_size)
    net = RetrievalWithSoftmax(bertconfig, use_kn)
    load_checkpoint(path, net=net)
    net.set_train(False)
    context_id = Tensor(np.zeros([args.batch_size, args.max_seq_length]), mstype.int32)
    context_segment_id = Tensor(np.zeros([args.batch_size, args.max_seq_length]), mstype.int32)
    context_pos_id = Tensor(np.zeros([args.batch_size, args.max_seq_length]), mstype.int32)
    kn_id = Tensor(np.zeros([args.batch_size, args.max_seq_length]), mstype.int32)
    kn_seq_length = Tensor(np.zeros([args.batch_size, 1]), mstype.int32)
    input_data = [context_id, context_segment_id, context_pos_id, kn_id, kn_seq_length]
    export(net.network, *input_data, file_name=save_ckpt_path +'Duconv', file_format="AIR")

@moxing_wrapper()
def run_duconv():
    '''duconv'''
    config = BertConfig(seq_length=args.max_seq_length, vocab_size=args.vocab_size)
    epoch = args.epoch
    # set context and device init
    if args.device_target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args.device_id)
        if args.run_distribute:
            device_num = get_device_num()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
    else:
        raise Exception("Target error,  only Ascend is supported.")

    use_kn = bool("kn" in args.task_name)
    # define dataset
    if args.run_distribute:
        device_num = get_device_num()
        dataset = create_dataset(args.batch_size, args.data_url,
                                 device_num=device_num, rank=args.device_id,
                                 do_shuffle=(args.train_data_shuffle), use_knowledge=use_kn)
    else:
        dataset_path = os.path.join(args.data_url, "train.mindrecord")
        dataset = create_dataset(args.batch_size, data_file_path=dataset_path,
                                 do_shuffle=(args.train_data_shuffle), use_knowledge=use_kn)
    steps_per_epoch = dataset.get_dataset_size()

    max_train_steps = args.epoch * steps_per_epoch
    warmup_steps = int(max_train_steps * args.warmup_proportion)
    # define net
    network = RetrievalWithLoss(config, use_kn)
    # define rate
    lr_schedule = Noam(config.hidden_size, warmup_steps, args.learning_rate)
    # define optimizer
    optimizer = Adam(network.trainable_params(), lr_schedule)
    # define model
    model = Model(network=network, optimizer=optimizer, amp_level="O2")

    data_size = args.save_checkpoint_steps if args.dataset_sink_mode else 100
    time_cb = TimeMonitor(data_size)
    loss_cb = LossMonitor(data_size)
    # define callbacks
    callbacks = [time_cb, loss_cb]
    # save model
    if get_rank_id() == 0:
        if args.rank_save_ckpt_flag:
            ckpt_cfg = CheckpointConfig(save_checkpoint_steps=args.save_checkpoint_steps,\
                keep_checkpoint_max=50)
        save_ckpt_path = os.path.join(args.save_checkpoint_path, 'ckpt_' + str(args.device_id) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_cfg, directory=save_ckpt_path,\
        prefix=args.task_name + '_rank_' + str(get_device_id()))
        callbacks.append(ckpt_cb)
    if args.dataset_sink_mode:
        epoch = 1
    path = os.path.join(save_ckpt_path, args.task_name + '_rank_' +
                        str(get_device_id())+'-'+str(epoch)+'_'+str(args.save_checkpoint_steps)+'.ckpt')
    # training
    print("============== Starting Training ==============")
    model.train(epoch, dataset, callbacks, dataset_sink_mode=args.dataset_sink_mode, sink_size=data_size)
    print("============== End Training ==============")
    # save ckptfile as air model
    save_ckpt_to_air(save_ckpt_path, path)
    moxing.file.copy_parallel(save_ckpt_path, args.train_url)
if __name__ == "__main__":
    run_duconv()
