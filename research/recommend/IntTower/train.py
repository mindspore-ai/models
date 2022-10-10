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


import model_config as cfg
import mindspore as ms
from mindspore import nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from model import IntTower
from util import AvgMeter, valid_epoch, setup_seed, train_epoch
from get_dataset import process_struct_data
from get_dataset import construct_dataset


def forward_fn(inputs, targets):
    logits = network(inputs)
    Loss = loss_fn(logits, targets.reshape(-1, 1))
    n_logit = logits.asnumpy()
    n_target = targets.reshape(-1, 1).asnumpy()
    Auc = roc_auc_score(n_target.astype(int), n_logit)
    return Loss, Auc


if __name__ == '__main__':
    print("1")

    ms.set_context(mode=ms.PYNATIVE_MODE)
    epoch = cfg.epoch
    batch_size = cfg.batch_size
    seed = cfg.seed
    lr = cfg.lr

    setup_seed(seed)

    data_path = './data/movielens.txt'
    train_dataset_generator, valid_dataset_generator, _ = process_struct_data(data_path)
    train_dataset = construct_dataset(train_dataset_generator, batch_size)
    valid_dataset = construct_dataset(valid_dataset_generator, batch_size)

    network = IntTower()
    loss_fn = nn.BCELoss(reduction='mean')
    net_opt = nn.Adam(network.trainable_params(), learning_rate=lr)

    best_loss = float('inf')
    best_auc = 0
    for i in range(epoch):
        loss_meter = AvgMeter()
        auc_meter = AvgMeter()
        tqdm_object = tqdm(train_dataset.create_dict_iterator(), total=len(train_dataset_generator) // batch_size)
        print("epoch %d :" % (i))
        train_loss = 0
        count = 0
        for batch in tqdm_object:
            loss, auc = train_epoch(forward_fn,
                                    batch["data"], batch["label"], net_opt)
            count = len(batch["label"])
            loss_meter.update(loss, count)
            auc_meter.update(auc, count)
            tqdm_object.set_postfix(train_loss=loss_meter.avg, train_auc=auc_meter.avg)

        valid_loss, valid_auc = valid_epoch(valid_dataset, network, loss_fn, valid_dataset_generator, batch_size)
        if valid_auc.avg > best_auc:
            best_auc = valid_auc.avg
            ms.save_checkpoint(network, "./IntTower.ckpt")
