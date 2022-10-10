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

import random
import numpy as np
import mindspore as ms
from mindspore.ops import value_and_grad, set_seed
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


class MyDataset:

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data.iloc[index], ms.Tensor(self.label.iloc[index], dtype=ms.float32)

    def __len__(self):
        return len(self.label)


def test_epoch(test_dataset, model, loss_fn, test_dataset_generator, batch_size):
    loss_meter = AvgMeter()
    auc_meter = AvgMeter()
    tqdm_object = tqdm(test_dataset.create_dict_iterator(), total=len(test_dataset_generator) // batch_size)
    for batch in tqdm_object:
        logits = model(batch["data"])
        loss = loss_fn(logits, batch["label"].reshape(-1, 1))
        n_logit = logits.asnumpy()
        n_target = batch["label"].reshape(-1, 1).asnumpy()
        auc = roc_auc_score(n_target.astype(int), n_logit)
        count = len(batch["label"])
        loss_meter.update(loss, count)
        auc_meter.update(auc, count)
        tqdm_object.set_postfix(test_loss=loss_meter.avg, test_auc=auc_meter.avg)
    return loss_meter, auc_meter


def valid_epoch(valid_dataset, model, loss_fn, valid_dataset_generator, batch_size):
    loss_meter = AvgMeter()
    auc_meter = AvgMeter()
    tqdm_object = tqdm(valid_dataset.create_dict_iterator(), total=len(valid_dataset_generator) // batch_size)
    for batch in tqdm_object:
        logits = model(batch["data"])
        loss = loss_fn(logits, batch["label"].reshape(-1, 1))
        n_logit = logits.asnumpy()
        n_target = batch["label"].reshape(-1, 1).asnumpy()
        auc = roc_auc_score(n_target.astype(int), n_logit)
        count = len(batch["label"])
        loss_meter.update(loss, count)
        auc_meter.update(auc, count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg, valid_auc=auc_meter.avg)
    return loss_meter, auc_meter


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)


def forward_fn(network, inputs, targets, loss_fn):
    logits = network(inputs)
    loss = loss_fn(logits, targets.reshape(-1, 1))
    n_logit = logits.asnumpy()
    n_target = targets.reshape(-1, 1).asnumpy()
    auc = roc_auc_score(n_target.astype(int), n_logit)
    return loss, auc


def train_epoch(forward, inputs, targets, net_opt):
    grad_fn = value_and_grad(forward, None, net_opt.parameters)
    (loss, auc), grads = grad_fn(inputs, targets)
    net_opt(grads)
    return loss, auc
