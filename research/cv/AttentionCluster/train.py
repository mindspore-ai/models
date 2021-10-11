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
"""Train  Attention Cluster"""
import os
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.context as context
import mindspore.common as common
import mindspore.train.callback as callback
import mindspore.train.loss_scale_manager as loss_manager
from mindspore.train.model import Model
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.communication.management import get_rank, get_group_size
from src.models.attention_cluster import AttentionCluster
from src.datasets.mnist_feature import MNISTFeature
from src.utils.config import parse_opts
from src.utils.callback import SaveCallback

if __name__ == '__main__':
    # Training settings
    args = parse_opts()
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # init context
    common.set_seed(args.seed)
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    if args.distributed:
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=device_id)
        init()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        rank = get_rank()
    else:
        context.set_context(device_id=args.device_id)
        rank = 0
    #define net
    fdim = [50]
    natt = [args.natt]
    nclass = 1024
    net = AttentionCluster(fdims=fdim, natts=natt, nclass=nclass, fc=args.fc)

    # define loss
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)

    # define optimizer
    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=args.lr, weight_decay=args.weight_decay)
    loss_scale_manager = loss_manager.DynamicLossScaleManager(4, 3000)

    # create dataset
    train_dataset_generator = MNISTFeature(root=args.data_dir, train=True, transform=None)

    eval_dataset_generator = MNISTFeature(root=args.data_dir, train=False, transform=None)
    eval_dataset = ds.GeneratorDataset(eval_dataset_generator, ["feature", "target"], shuffle=False)
    eval_dataset = eval_dataset.batch(args.batch_size, drop_remainder=True)

    if args.distributed:
        train_dataset = ds.GeneratorDataset(train_dataset_generator, ["feature", "target"], shuffle=True,
                                            num_shards=get_group_size(), shard_id=get_rank())
    else:
        train_dataset = ds.GeneratorDataset(train_dataset_generator, ["feature", "target"], shuffle=True)
    train_dataset = train_dataset.batch(args.batch_size, drop_remainder=True)

    # define model
    model = Model(network=net, loss_fn=loss, optimizer=optimizer, amp_level='O0', loss_scale_manager=loss_scale_manager,
                  metrics={'top_1_accuracy': nn.Top1CategoricalAccuracy(),
                           'top_5_accuracy': nn.Top5CategoricalAccuracy()}
                  )

    # define callback
    step_size = train_dataset.get_dataset_size()
    time_cb = callback.TimeMonitor(data_size=step_size)
    loss_cb = callback.LossMonitor()
    cb = [time_cb, loss_cb]
    if rank == 0:
        save_ckpt_cb = SaveCallback(model, eval_dataset, args)
        cb.append(save_ckpt_cb)

    # train
    model.train(epoch=args.epochs, train_dataset=train_dataset, callbacks=cb, dataset_sink_mode=False)
