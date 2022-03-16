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
"""Train"""
import os
from time import time

import numpy as np

from mindspore import nn, Tensor
from mindspore import context
from mindspore import set_seed
from mindspore import Model
from mindspore.common import dtype as mstype
from mindspore import save_checkpoint
from mindspore.train.callback import Callback
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.callback import TimeMonitor, LossMonitor, SummaryCollector
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.profiler import Profiler

from src.atae_for_train import NetWithLoss
from src.model import AttentionLstm
from src.load_dataset import load_dataset
from src.config import config


class StopAtStep(Callback):
    def __init__(self, start_step, stop_step):
        super(StopAtStep, self).__init__()
        self.start_step = start_step
        self.stop_step = stop_step
        self.profiler = Profiler(output_path='./summary_dir', start_profile=False)
    def step_begin(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.start_step:
            self.profiler.start()
    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.stop_step:
            self.profiler.stop()
    def end(self, run_context):
        self.profiler.analyse()


class SaveCallback(Callback):
    """
    define savecallback, save best model while training.
    """
    def __init__(self, eval_net, dataset, save_file_path, rank):
        super(SaveCallback, self).__init__()
        self.net = eval_net
        self.eval_dataset = dataset
        self.save_path = save_file_path
        self.acc = 0.5
        self.rank = rank

    def step_end(self, run_context):
        """
        eval and save ckpt while training
        """
        cb_params = run_context.original_args()

        correct = 0
        count = 0
        self.net.is_train = False
        self.net.set_train(False)
        for batch in self.eval_dataset.create_dict_iterator():
            content = batch['content']
            sen_len = batch['sen_len']
            aspect = batch['aspect']
            solution = batch['solution']

            pred = self.net(content, sen_len, aspect)

            polarity_pred = np.argmax(pred.asnumpy(), axis=1)
            polarity_label = np.argmax(solution.asnumpy(), axis=1)

            correct += (polarity_pred == polarity_label).sum()
            count += len(polarity_label)

        self.net.is_train = True
        self.net.set_train(True)

        res = correct / count
        if res > self.acc:
            self.acc = res
            file_name = self.save_path + '_max' + ".ckpt"
            save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)
            print(f"Rank {self.rank}: save the maximum accuracy checkpoint,the accuracy is {self.acc}")


if __name__ == '__main__':
    config.train_url = os.path.join(config.data_url, 'train')

    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device,
                        save_graphs=False)

    set_seed(config.rseed)

    data_menu = config.data_url

    if config.is_modelarts:
        import moxing as mox
        mox.file.copy_parallel(src_url=config.data_url, dst_url='/cache/dataset_menu')
        data_menu = '/cache/dataset_menu/'

    train_dataset = data_menu + '/train.mindrecord'
    eval_dataset = data_menu + '/test.mindrecord'
    word_path = data_menu + '/weight.npz'

    if config.parallel:
        # Parallel mode
        init()
        context.reset_auto_parallel_context()
        config.rank_id = get_rank()
        config.group_size = get_group_size()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True, parameter_broadcast=True,
                                          device_num=config.group_size)

    dataset_train = load_dataset(input_files=train_dataset,
                                 batch_size=config.batch_size, rank_size=config.group_size, rank_id=config.rank_id)
    dataset_val = load_dataset(input_files=eval_dataset,
                               batch_size=config.batch_size)

    epoch_size = config.epoch
    steps_per_epoch = dataset_train.get_dataset_size()
    total_steps = steps_per_epoch * epoch_size

    r = np.load(word_path)
    word_vector = r['weight']
    weight = Tensor(word_vector, mstype.float32)

    net = AttentionLstm(config, weight, is_train=True)
    model_with_loss = NetWithLoss(net, batch_size=1)

    word_vector_params = {}
    word_vector_params['lr'] = config.lr_word_vector
    other_params = {}
    other_params['params'] = []
    other_params['lr'] = config.lr
    for param in net.trainable_params():
        if param.name.endswith('embedding_word.embedding_table'):
            word_vector_params['params'] = [param]
        else:
            other_params['params'].append(param)
    params_list = [word_vector_params, other_params]

    if config.optimizer == 'Momentum':
        optimizer = nn.Momentum(params=params_list,
                                learning_rate=config.lr,  # mindspore issue - Momentum requires define lr here
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)
    elif config.optimizer == 'Adagrad':
        optimizer = nn.Adagrad(params=params_list,
                               weight_decay=config.weight_decay)

    train_net = nn.TrainOneStepCell(model_with_loss, optimizer)

    model = Model(train_net)

    if config.parallel:
        ckpoint_dir = "ckpt_rank" + str(get_rank())
        summary_dir = "./summary_dir" + str(get_rank())+'/'
    else:
        ckpoint_dir = "ckpt"
        summary_dir = "./summary_dir/"

    time_cb = TimeMonitor(data_size=steps_per_epoch)
    loss_cb = LossMonitor()
    summary_collector = SummaryCollector(summary_dir=summary_dir)
    cb = [time_cb, loss_cb, summary_collector]
    if not config.parallel or get_rank() == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=steps_per_epoch,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix="atae-lstm", directory='./train/', config=config_ck)
        cb.append(ckpoint_cb)

    if config.is_modelarts:
        os.makedirs('/cache/train_output/')
        ckpoint_dir = '/cache/train_output/atae-lstm'

    save_cb = SaveCallback(net, dataset_val, ckpoint_dir, config.rank_id)
    cb.append(save_cb)

    if config.profile and not config.dataset_sink_mode:
        profiler_cb = StopAtStep(start_step=10, stop_step=20)
        cb.append(profiler_cb)

    print("start train")
    start_time = time()
    model.train(epoch_size, dataset_train, callbacks=cb, dataset_sink_mode=config.dataset_sink_mode)
    end_time = time()
    print("train success!")
    time_str = f"{int(end_time - start_time) // 60} min {int(end_time - start_time) % 60} sec"
    print("Total time taken: " + time_str)

    if config.is_modelarts:
        mox.file.copy_parallel(src_url='/cache/train_output/', dst_url=config.train_url)
