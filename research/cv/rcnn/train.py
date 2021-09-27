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
"""
Train RCNN and get checkpoint files.
"""
import argparse
import numpy as np
import mindspore
from mindspore import load_param_into_net, load_checkpoint, Tensor, nn

from src.common.mindspore_utils import MSUtils
from src.common.multi_margin_loss import MultiMarginLoss
from src.common.trainer import Trainer
from src.dataset import FinetuneAndSVMDataset, RegressionDataset, FinetuneAndSVMDataset_test
from src.generator_lr import get_lr
from src.model import AlexNetCombine, BBoxNet
from src.paths import Model


def finetune():
    """
    finetune
    """
    train_dataset = FinetuneAndSVMDataset(True, ['train', 'val'])
    train_dataloader = MSUtils.prepare_dataloader(train_dataset, ['samplers', 'labels'])
    validate_dataset = FinetuneAndSVMDataset_test(True, ['test'])
    validate_dataloader = MSUtils.prepare_dataloader(validate_dataset, ['samplers', 'labels'])

    model = AlexNetCombine(21, phase='train')
    load_param_into_net(model.backbone, load_checkpoint(Model.pretrained_alexnet))

    criteria = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    lr = Tensor(get_lr(lr=0.013, epoch_size=2, steps_per_epoch=train_dataset.__len__()))
    optimizer = nn.Momentum(model.get_parameters(), learning_rate=lr, momentum=0.9, weight_decay=1e-4)

    trainer = Trainer('finetune', train_dataloader, ['samplers'], 'labels', model, criteria, optimizer,
                      validation_dataloader=validate_dataloader)
    for _ in range(2):
        trainer.train()
        trainer.validate(calculate_accuracy=True, save_best=True)


def svm():
    """
    svm
    """
    train_dataset = FinetuneAndSVMDataset(False, ['train', 'val'])
    train_dataloader = MSUtils.prepare_dataloader(train_dataset, ['samplers', 'labels'])
    validate_dataset = FinetuneAndSVMDataset_test(False, ['test'])
    validate_dataloader = MSUtils.prepare_dataloader(validate_dataset, ['samplers', 'labels'])

    model = AlexNetCombine(21, phase='train')
    load_param_into_net(model, load_checkpoint(Model.finetune))

    weight_list = np.array(
        [15.9, 13.6, 9.9, 17.4, 14.0, 20.9, 3.8, 12.9, 6.0, 17.6, 23.3, 9.5, 13.2, 14.4, 1.0, 9.7, 18.5, 19.4,
         16.6, 14.5, 4.5])
    criteria = MultiMarginLoss(21, Tensor(weight_list, dtype=mindspore.float32))

    lr = Tensor(get_lr(lr=1e-3, epoch_size=30, steps_per_epoch=train_dataset.__len__()))
    optimizer = nn.Momentum(model.get_parameters(), learning_rate=lr, momentum=0.9, weight_decay=1e-4)

    trainer = Trainer('svm', train_dataloader, ['samplers'], 'labels', model, criteria, optimizer,
                      validation_dataloader=validate_dataloader)
    for _ in range(30):
        trainer.train()
        trainer.validate(calculate_accuracy=True, save_best=True)


def regression():
    """
    regression
    """
    train_dataset = RegressionDataset(['train', 'val'])
    train_dataloader = MSUtils.prepare_dataloader(train_dataset, ["crop_img", "trans", "cls_onehot"], batch_size=4096,
                                                  is_shuffle=True)
    validate_dataset = RegressionDataset(['test'])
    validate_dataloader = MSUtils.prepare_dataloader(validate_dataset, ["crop_img", "trans", "cls_onehot"],
                                                     batch_size=4096)
    model = BBoxNet()
    load_param_into_net(model.backbone, load_checkpoint(Model.finetune))

    criteria = nn.MSELoss()
    optimizer = nn.Adam(model.get_parameters(), learning_rate=1e-4)

    trainer = Trainer('regression', train_dataloader, ['crop_img', 'cls_onehot'], 'trans', model, criteria, optimizer,
                      validation_dataloader=validate_dataloader)
    for _ in range(30):
        trainer.train()
        trainer.validate(train_reg=True)


def eval_finetune():
    """
    eval_finetune
    """
    validate_dataset = FinetuneAndSVMDataset_test(True, ['test'])
    validate_dataloader = MSUtils.prepare_dataloader(validate_dataset, ['samplers', 'labels'])

    model = AlexNetCombine(21, phase='test')
    criteria = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    load_param_into_net(model, load_checkpoint(Model.finetune))

    trainer = Trainer('finetune', None, ['samplers'], 'labels', model, criteria, None,
                      validation_dataloader=validate_dataloader)
    trainer.validate(calculate_accuracy=True, debug=False)


def eval_svm():
    """
    eval_svm
    """
    validate_dataset = FinetuneAndSVMDataset_test(False, ['test'])
    validate_dataloader = MSUtils.prepare_dataloader(validate_dataset, ['samplers', 'labels'])

    weight_list = np.array(
        [15.9, 13.6, 9.9, 17.4, 14.0, 20.9, 3.8, 12.9, 6.0, 17.6, 23.3, 9.5, 13.2, 14.4, 1.0, 9.7, 18.5, 19.4,
         16.6, 14.5, 4.5])

    model = AlexNetCombine(21, phase='test')
    criteria = MultiMarginLoss(21, Tensor(weight_list, dtype=mindspore.float32))
    load_param_into_net(model, load_checkpoint(Model.svm))

    trainer = Trainer('svm', None, ['samplers'], 'labels', model, criteria, None,
                      validation_dataloader=validate_dataloader)
    trainer.validate(calculate_accuracy=True, debug=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=int, default=1)
    parser.add_argument('-s', '--step', type=int, default=0)
    args = parser.parse_args()

    MSUtils.initialize(device="Ascend", device_id=args.device)
    steps = [finetune, svm, regression, eval_finetune, eval_svm]
    steps[args.step]()
