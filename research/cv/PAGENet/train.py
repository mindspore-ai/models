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
import time
import argparse
import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore.communication.management import init, get_rank, get_group_size
from config import MODE, device_target, train_size, train_img_path, train_edge_path, train_gt_path, batch_size, EPOCH, \
    LR, WD
from src.pagenet import MindsporeModel
from src.train_loss import total_loss, WithLossCell
from src.mind_dataloader_final import get_train_loader


def main(train_mode='single'):
    context.set_context(mode=MODE,
                        device_target=device_target,
                        reserve_class_name_in_scope=False)

    if train_mode == 'single':
        # env set

        # dataset
        train_loader = get_train_loader(train_img_path, train_gt_path, train_edge_path, batchsize=batch_size,
                                        trainsize=train_size)
        train_data_size = train_loader.get_dataset_size()
        # epoch
        epoch = EPOCH
    else:
        init()
        rank_id = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num, gradients_mean=True,
                                          parallel_mode=context.ParallelMode.DATA_PARALLEL)

        # dataset
        train_loader = get_train_loader(train_img_path, train_gt_path, train_edge_path, device_num=device_num,
                                        rank_id=rank_id, num_parallel_workers=8, batchsize=batch_size,
                                        trainsize=train_size)

        train_data_size = train_loader.get_dataset_size()
        # epoch
        epoch = EPOCH * 2

    # setup train_parameters

    model = MindsporeModel()
    # loss function
    loss_fn = total_loss()

    # learning_rate and optimizer

    optimizer = nn.Adam(model.trainable_params(), learning_rate=LR, weight_decay=WD)

    # train model
    net_with_loss = WithLossCell(model, loss_fn)
    train_network = nn.TrainOneStepCell(net_with_loss, optimizer)
    train_network.set_train()

    data_iterator = train_loader.create_tuple_iterator(num_epochs=epoch)

    start = time.time()
    for i in range(epoch):
        total_train_step = 0
        for imgs, gts, edges in data_iterator:

            loss = train_network(imgs, gts, edges)

            total_train_step = total_train_step + 1

            if total_train_step % 10 == 0:
                print("epoch： {}，  step： {}/{}， loss: {}".format(i, total_train_step, train_data_size, loss))

    if train_mode == 'single':
        mindspore.save_checkpoint(train_network, "PAGENET" + '.ckpt')
        print("PAGENET.ckpt" + " have saved!")

    else:
        mindspore.save_checkpoint(train_network, "PAGENET" + str(get_rank()) + '.ckpt')
        print("PAGENET" + str(get_rank()) + '.ckpt' + " have saved!")
    end = time.time()
    total = end - start
    print("total time is {}h".format(total / 3600))
    print("step time is {}s".format(total / (train_data_size * epoch)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('-m', '--train_mode', type=str)
    args = parser.parse_args()
    main(args.train_mode)
