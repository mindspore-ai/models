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

import os
import time

from mindspore import context, nn
from mindspore import save_checkpoint
from mindspore.communication import management as D
from mindspore.communication.management import get_group_size, get_rank
from src.models.mtcnn import RNetWithLoss, RNetTrainOneStepCell
from src.dataset import create_train_dataset
from src.utils import MultiEpochsDecayLR

def train_rnet(args):
    print("The argument is: ", args)
    context.set_context(device_target=args.device_target, mode=context.GRAPH_MODE)
    device_id = 0
    device_num = 1
    if args.distribute:
        D.init()
        device_id = get_rank()
        device_num = get_group_size()

        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)

    else:
        context.set_context(device_id=int(os.getenv('DEVICE_ID', '0')))

    # Create train dataset
    ds_train = create_train_dataset(args.mindrecord_file, args.batch_size,
                                    device_num, device_id, num_workers=args.num_workers)
    steps_per_epoch = ds_train.get_dataset_size()
    network = RNetWithLoss()

    network.set_train(True)

    # decay lr
    if args.distribute:
        lr_scheduler = MultiEpochsDecayLR(args.lr, [7, 15, 20], steps_per_epoch)
    else:
        lr_scheduler = MultiEpochsDecayLR(args.lr, [6, 14, 20], steps_per_epoch)

    # optimizer
    optimizer = nn.Adam(params=network.trainable_params(), learning_rate=lr_scheduler, weight_decay=1e-4)

    # train net
    train_net = RNetTrainOneStepCell(network, optimizer)
    train_net.set_train(True)

    print("Start training RNet")
    for epoch in range(1, args.end_epoch+1):
        step = 0
        time_list = []
        for d in ds_train.create_tuple_iterator():
            start_time = time.time()
            loss = train_net(*d)
            step += 1
            print(f'epoch: {epoch} step: {step}, loss is {loss}')
            per_time = time.time() - start_time
            time_list.append(per_time)
        print('per step time: ', '%.2f' % (sum(time_list) / len(time_list) * 1000), "(ms/step)")

    if args.distribute and device_id == 0:
        save_checkpoint(network, os.path.join(args.ckpt_path, 'rnet_distribute.ckpt'))
    elif not args.distribute:
        save_checkpoint(network, os.path.join(args.ckpt_path, 'rnet_standalone.ckpt'))
