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
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.communication.management import init, get_rank, get_group_size
from src.pagenet import MindsporeModel
from src.train_loss import TotalLoss, WithLossCell
from src.mind_dataloader_final import get_train_loader
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.mytrainonestep import CustomTrainOneStepCell

ms.set_seed(1)

@moxing_wrapper()
def main():
    context.set_context(mode=config.MODE,
                        device_target=config.device_target,
                        reserve_class_name_in_scope=False)

    if config.train_mode == 'single':
        # dataset
        train_loader = get_train_loader(config.train_img_path, config.train_gt_path, config.train_edge_path,
                                        batchsize=config.batch_size, trainsize=config.train_size)
        train_data_size = train_loader.get_dataset_size()
        # epoch
        epoch = config.EPOCH
    else:
        init()
        rank_id = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num, gradients_mean=True,
                                          parallel_mode=context.ParallelMode.DATA_PARALLEL)

        # dataset
        if config.need_loss_scale:
            config.batch_size = int(config.batch_size // device_num)
        train_loader = get_train_loader(config.train_img_path, config.train_gt_path, config.train_edge_path,
                                        device_num=device_num, rank_id=rank_id, num_parallel_workers=8,
                                        batchsize=config.batch_size, trainsize=config.train_size)

        train_data_size = train_loader.get_dataset_size()
        # epoch
        epoch = config.EPOCH * 2
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path, exist_ok=True)
    # setup train_parameters

    model = MindsporeModel(config)
    # loss function
    loss_fn = TotalLoss()

    # learning_rate and optimizer
    lr = config.LR

    if config.optimizer == "adam":
        optimizer = nn.Adam(model.trainable_params(), learning_rate=lr, weight_decay=config.WD)
    elif config.optimizer == "adamw":
        optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=lr, weight_decay=config.WD)
    else:
        raise ValueError(f"Not support {config.optimizer}, support ['adam', 'adamw']")

    # train model
    net_with_loss = WithLossCell(model, loss_fn)
    if config.need_loss_scale:
        loss_scale = 2048
        train_network = CustomTrainOneStepCell(net_with_loss, optimizer, loss_scale)
    else:
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
                print("epoch: {}，LR: {},  step： {}/{}， loss: {}".format(i, lr, total_train_step, train_data_size, loss))

    if config.train_mode == 'single':
        ms.save_checkpoint(train_network, os.path.join(config.output_path, "PAGENET.ckpt"))
        print("PAGENET.ckpt" + " have saved!")

    else:
        ms.save_checkpoint(train_network, os.path.join(config.output_path, f"PAGENET{get_rank()}.ckpt"))
        print(f"PAGENET{get_rank()}.ckpt have saved!")
    end = time.time()
    total = end - start
    print("total time is {}h".format(total / 3600))
    print("step time is {}s".format(total / (train_data_size * epoch)))


if __name__ == "__main__":
    main()
