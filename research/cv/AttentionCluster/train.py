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
"""Train  Attention Cluster"""
import os
from pprint import pprint
from time import time

import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.context as context
import mindspore.common as common
import mindspore.train.callback as callback
from mindspore.train.model import Model
from mindspore.train.callback import Callback
from mindspore.train.callback import SummaryCollector
from mindspore.profiler import Profiler
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.communication.management import get_rank, get_group_size

from src.models.attention_cluster import AttentionCluster
from src.datasets.mnist_feature import MNISTFeature
from src.utils.config import config as cfg
from src.utils.callback import SaveCallback


class StopAtStep(Callback):
    def __init__(self, start_step, stop_step):
        super(StopAtStep, self).__init__()
        self.start_step = start_step
        self.stop_step = stop_step
        self.profiler = Profiler(output_path=cfg.summary_dir, start_profile=False)
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


if __name__ == '__main__':
    summary_dir = cfg.summary_dir

    # Training settings
    if not os.path.exists(cfg.result_dir):
        os.makedirs(cfg.result_dir)

    common.set_seed(cfg.seed)

    if cfg.device not in ["GPU", "Ascend"]:
        raise NotImplementedError("Training only supported for Ascend and GPU.")

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device, save_graphs=False)

    if cfg.distributed:
        if cfg.device == "Ascend":
            init()
            rank = get_rank()
            group_size = get_group_size()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=group_size,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              parameter_broadcast=True)
        elif cfg.device == "GPU":
            init('nccl')
            rank = get_rank()
            group_size = get_group_size()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=group_size,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True,
                                              parameter_broadcast=True)
    else:
        context.set_context(device_id=cfg.device_id)
        rank = 0

    if rank == 0:
        print("===> Configuration:")
        pprint(cfg)

    #define net
    fdim = [50]
    natt = [cfg.natt]
    nclass = 1024
    net = AttentionCluster(fdims=fdim, natts=natt, nclass=nclass, fc=cfg.fc)

    # define loss
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)

    # define optimizer
    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    # loss_scale_manager = loss_manager.DynamicLossScaleManager(4, 3000)

    # create dataset
    train_dataset_generator = MNISTFeature(root=cfg.data_dir, train=True, transform=None)
    print(f"Train dataset len: {len(train_dataset_generator)}")
    eval_dataset_generator = MNISTFeature(root=cfg.data_dir, train=False, transform=None)
    print(f"Eval dataset len: {len(eval_dataset_generator)}")
    eval_dataset = ds.GeneratorDataset(eval_dataset_generator, ["feature", "target"], shuffle=False)
    eval_dataset = eval_dataset.batch(cfg.batch_size, drop_remainder=False)

    if cfg.distributed:
        train_dataset = ds.GeneratorDataset(train_dataset_generator, ["feature", "target"], shuffle=True,
                                            num_shards=get_group_size(), shard_id=get_rank())
    else:
        train_dataset = ds.GeneratorDataset(train_dataset_generator, ["feature", "target"], shuffle=True)
    train_dataset = train_dataset.batch(cfg.batch_size, drop_remainder=True)

    # define model
    model = Model(network=net, loss_fn=loss, optimizer=optimizer, #amp_level='O0', loss_scale_manager=loss_scale_manager,
                  metrics={'top_1_accuracy': nn.Top1CategoricalAccuracy(),
                           'top_5_accuracy': nn.Top5CategoricalAccuracy()}
                  )

    # define callback
    summary_cb = SummaryCollector(summary_dir+'/thread_num'+str(rank))
    step_size = train_dataset.get_dataset_size()
    time_cb = callback.TimeMonitor(data_size=step_size)
    loss_cb = callback.LossMonitor()
    cb = [time_cb, loss_cb, summary_cb]
    if rank == 0:
        save_ckpt_cb = SaveCallback(model, eval_dataset, cfg)
        cb.append(save_ckpt_cb)

    if cfg.profile:
        sink_mode = False
        profiler_cb = StopAtStep(start_step=10, stop_step=20)
        callback.append(profiler_cb)
    else:
        sink_mode = True

    # train
    print("===> Started training...")
    start_time = time()
    model.train(epoch=cfg.epochs, train_dataset=train_dataset, callbacks=cb, dataset_sink_mode=sink_mode)
    total_time = time() - start_time
    print(f"===> Done! Time taken: {total_time:.3f} sec")
