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
the trainer of finetune,svm and bbox regression.
"""
import os
import time

import numpy as np
from mindspore import nn

from src.common.logger import Logger
from src.paths import Model
from src.common.mindspore_utils import my_save_checkpoint as save_checkpoint


class MyWithLossCell(nn.Cell):
    """
    MyWithLossCell
    """

    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, x, y, label):
        """

        :param x: input x
        :param y: input y
        :param label: label
        :return: loss
        """
        out = self._backbone(x, y)
        return self._loss_fn(out, label)

    @property
    def backbone_network(self):
        """

        :return: backbone network
        """
        return self._backbone


class Trainer:
    """
    Trainer
    """

    def __init__(self, name, dataloader, input_names: list, label_name: str, model, criteria, optimizer,
                 lr_scheduler=None,
                 validation_dataloader=None):
        self.name = name
        self.dataloader = dataloader
        self.validation_dataloader = validation_dataloader
        self.model = model
        self.criteria = criteria
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logger = Logger('trainer_' + name)
        self.best_acc = -1.0
        self.best_loss = 100000
        self.epoch_total_times = []
        self.epoch_total_steps = []

        if len(input_names) == 1:
            self.loss_net = nn.WithLossCell(self.model, self.criteria)
            self.input_num = 1
            self.x1_name = input_names[0]
        elif len(input_names) == 2:
            self.loss_net = MyWithLossCell(self.model, self.criteria)
            self.input_num = 2
            self.x1_name = input_names[0]
            self.x2_name = input_names[1]
        else:
            self.logger.critical('The length of "input_names" should be 1 or 2.')

        self.y_name = label_name
        if self.optimizer is not None:
            self.net = nn.TrainOneStepCell(self.loss_net, self.optimizer)
        self.epoch = 0

    def train(self):
        """
        train
        """
        self.model.set_train(True)
        self.epoch += 1
        batch_loss = 0.
        step = 0
        epoch_total_time = 0
        for data in self.dataloader:
            time_step_start = time.time()
            x1 = data[self.x1_name]
            y = data[self.y_name]
            if self.input_num == 2:
                x2 = data[self.x2_name]
                loss = self.net(x1, x2, y)
            else:
                loss = self.net(x1, y)
            batch_loss += loss.asnumpy()
            step += 1
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            step_time = (time.time() - time_step_start) * 1000
            epoch_total_time += step_time
            self.logger.debug(f'[train] epoch {self.epoch}, step: {step}, time: {step_time:.0f} ms/step, loss: {loss}')
        self.logger.info(f'[train] epoch {self.epoch}, loss: {batch_loss / step}')
        epoch_average_time = epoch_total_time / step
        self.epoch_total_times.append(epoch_total_time)
        self.epoch_total_steps.append(step)
        self.logger.info(f'[epoch time] total {epoch_total_time:.0f} ms, average {epoch_average_time:.0f} ms/step')
        self.logger.info(f'[total time] total {np.sum(self.epoch_total_times):.0f} ms, '
                         f'average {np.sum(self.epoch_total_times) / np.sum(self.epoch_total_steps):.0f} ms/step')

        save_checkpoint(self.model, os.path.join(Model.save_path, self.name + "_latest.ckpt"))
        save_checkpoint(self.model, os.path.join(Model.save_path, self.name + "_epoch_%d.ckpt" % self.epoch))

    def validate(self, calculate_accuracy=False, debug=False, save_best=False, train_reg=False):
        """
        :param calculate_accuracy: calculate accuracy or not
        :param debug: debug or not
        :param save_best: save best or not
        :param train_reg: train_reg
        :return: batch_loss
        """
        if self.validation_dataloader is None:
            self.logger.warning('The validation dataset is empty, skipping!')
            return None

        self.model.set_train(False)
        batch_loss = 0.
        step = 0
        total_correct_num = 0
        total_num = 0
        for data in self.validation_dataloader:
            x1 = data[self.x1_name]
            y = data[self.y_name]
            if self.input_num == 2:
                x2 = data[self.x2_name]
                y_pred = self.model(x1, x2)
                loss = self.criteria(y_pred, y)
            else:
                y_pred = self.model(x1)
                loss = self.criteria(y_pred, y)
            batch_loss += loss.asnumpy()
            step += 1
            if calculate_accuracy:
                predict = y_pred.asnumpy().argmax(axis=1)
                if debug:
                    self.logger.debug("predict: %s" % predict)
                    self.logger.debug("true: %s" % y)
                right = (predict == y)
                correct_num = right.sum()
                total_correct_num += correct_num
                total_num += right.size
                self.logger.debug('[valid] epoch %d, step: %d, loss: %s, acc: %.4f%%' % (
                    self.epoch, step, loss, 100 * correct_num / right.size))
            else:
                self.logger.debug('[valid] epoch %d, step: %d, loss: %s' % (self.epoch, step, loss))
        if calculate_accuracy:
            epoch_acc = 100 * total_correct_num / total_num
            self.logger.info('[valid] epoch %d, loss: %s, acc: %.4f%%' % (
                self.epoch, batch_loss / step, epoch_acc))
            if save_best and epoch_acc > self.best_acc:
                self.best_acc = epoch_acc
                save_checkpoint(self.model, os.path.join(Model.save_path, self.name + "_best.ckpt"))
        else:
            self.logger.info('[valid] epoch %d, loss: %s' % (self.epoch, batch_loss / step))

        if train_reg:
            if batch_loss < self.best_loss:
                self.best_loss = batch_loss
                save_checkpoint(self.model, os.path.join(Model.save_path, self.name + "_best.ckpt"))
        return batch_loss
