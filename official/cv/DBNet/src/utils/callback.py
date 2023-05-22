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
"""Monitor the result of DBNet."""
import os
import time
import numpy as np

import mindspore as ms
from mindspore.train.callback import Callback

from src.datasets.load import create_dataset
from src.modules.model import get_dbnet
from .metric import AverageMeter
from .eval_utils import WithEval


class ResumeCallback(Callback):
    def __init__(self, start_epoch_num=0):
        super(ResumeCallback, self).__init__()
        self.start_epoch_num = start_epoch_num

    def on_train_epoch_begin(self, run_context):
        run_context.original_args().cur_epoch_num += self.start_epoch_num


class DBNetMonitor(Callback):
    """
    Monitor the result of DBNet.
    If the loss is NAN or INF, it will terminate training.
    Note:
        If per_print_times is 0, do not print loss.
    Args:
        config(class): configuration class.
        train_net(nn.Cell): Train network.
        per_print_times (int): How many steps to print once loss. During sink mode, it will print loss in the
                               nearest step. Default: 1.
    Raises:
        ValueError: If per_print_times is not an integer or less than zero.
    """

    def __init__(self, config, train_net, lr, per_print_times=1, cur_steps=0):
        super(DBNetMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("The argument 'per_print_times' must be int and >= 0, "
                             "but got {}".format(per_print_times))
        self._per_print_times = max(per_print_times, 1)
        self._last_print_time = 0
        self.config = config
        self.batch_size = config.train.batch_size
        self.lr = lr
        self.loss_avg = AverageMeter()
        self.rank_id = config.rank_id
        self.device_num = config.device_num
        self.run_eval = config.run_eval
        self.eval_interval = config.eval_interval
        self.save_ckpt_dir = config.save_ckpt_dir
        if self.run_eval:
            config.backbone.pretrained = False
            eval_net = get_dbnet(config.net, config, isTrain=False)
            self.eval_net = WithEval(eval_net, config)
            val_dataset, _ = create_dataset(config, False)
            self.val_dataset = val_dataset.create_dict_iterator(output_numpy=True)
            self.max_f = 0.0
            self.early_stop = config.early_stop
            self.stop_value = config.stop_value.__dict__
        self.train_net = train_net
        self.epoch_start_time = time.time()
        self.step_start_time = time.time()
        self.train_start = time.time()
        self.cur_steps = cur_steps
        self.all_fps = []
        self.epoch_num = config.train.total_epochs - config.train.start_epoch_num

    def load_parameter(self):
        param_dict = dict()
        for name, param in self.train_net.parameters_and_names():
            param_dict[name] = param
        for name, param in self.eval_net.model.parameters_and_names():
            if name in param_dict:
                param.set_data(param_dict[name])
            else:
                print(f"parameter {name} not in train_net")

    def handle_loss(self, net_outputs):
        """Handle loss"""
        if isinstance(net_outputs, (tuple, list)):
            if isinstance(net_outputs[0], ms.Tensor) and isinstance(net_outputs[0].asnumpy(), np.ndarray):
                loss = net_outputs[0].asnumpy()
                if bool(net_outputs[1].asnumpy()):
                    self.config.logger.info('=====================overflow=====================')
        elif isinstance(net_outputs, ms.Tensor) and isinstance(net_outputs.asnumpy(), np.ndarray):
            loss = float(np.mean(net_outputs.asumpy()))
        return loss

    def on_train_begin(self, run_context):
        self.config.logger.info('train start')
        self.train_start = time.time()

    def on_train_epoch_begin(self, run_context):
        """
        Called before each epoch beginning.
        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.epoch_start_time = time.time()
        self.step_start_time = time.time()

    def on_train_step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        loss = self.handle_loss(cb_params.net_outputs)
        cur_epoch = cb_params.cur_epoch_num
        data_sink_mode = cb_params.get('dataset_sink_mode', False)
        if not data_sink_mode:
            cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

            if cur_step_in_epoch == 1:
                self.loss_avg = AverageMeter()
            self.loss_avg.update(loss)

            if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
                raise ValueError(
                    "epoch: {} step: {}. Invalid loss, terminating training.".format(cur_epoch, cur_step_in_epoch))

            if cur_step_in_epoch % self._per_print_times == 0:
                per_step_time = (time.time() - self.step_start_time) * 1000 / self._per_print_times
                fps = self.batch_size * 1000 * self.device_num / per_step_time
                loss_log = "epoch: [%s/%s] step: [%s/%s], loss: %.6f, lr: %.6f, per step time: %.3f ms, " \
                           "fps: %.2f img/s" % (
                               cur_epoch, self.config.train.total_epochs, cur_step_in_epoch, cb_params.batch_num,
                               np.mean(self.loss_avg.avg), self.lr[self.cur_steps], per_step_time, fps)
                self.config.logger.info(loss_log)
                self.step_start_time = time.time()
        self.cur_steps += 1

    def on_train_epoch_end(self, run_context):
        """
        Called after each training epoch end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        cur_epoch = cb_params.cur_epoch_num
        epoch_time = (time.time() - self.epoch_start_time)
        per_step_time = epoch_time * 1000 / cb_params.batch_num
        fps = 1000 * self.batch_size * self.device_num / per_step_time
        self.all_fps.append(fps)
        loss_log = "epoch: [%s/%s], loss: %.6f, epoch time: %.3f s, per step time: %.3f ms, fps: %.2f img/s" % (
            cur_epoch, self.config.train.total_epochs, loss[0].asnumpy(), epoch_time, per_step_time, fps)
        self.config.logger.info(loss_log)
        if self.run_eval and (cur_epoch - self.config.eval_start_epoch) % self.eval_interval == 0 and \
                cur_epoch >= self.config.eval_start_epoch:
            self.load_parameter()
            self.eval_net.model.set_train(False)
            metrics, fps = self.eval_net.eval(self.val_dataset, show_imgs=self.config.eval.show_images)

            cur_f = metrics.get('fmeasure', None)
            cur_f = cur_f.avg if cur_f else 0.0
            self.config.logger.info('current epoch is: %s \n FPS: %s \n Recall: %s \n Precision: %s \n Fmeasure: %s' % (
                cur_epoch, fps, metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg))
            if cur_f >= self.max_f and self.rank_id == 0:
                self.config.logger.info('update best ckpt at epoch: %s, best fmeasure is: %s' % (cur_epoch, cur_f))
                if ms.context.get_context("enable_ge"):
                    from mindspore.train.callback import _set_cur_net
                    _set_cur_net(cb_params.train_network)
                    cb_params.train_network.exec_checkpoint_graph()
                ms.save_checkpoint(self.eval_net.model,
                                   os.path.join(self.save_ckpt_dir, f"best_rank{self.config.rank_id}.ckpt"))
                self.max_f = cur_f
            if self.early_stop and isinstance(self.stop_value, dict) and self.stop_value:
                stop = True
                for key in self.stop_value.keys():
                    if metrics[key].avg < self.stop_value[key]:
                        stop = False
                if stop:
                    self.config.logger.info(f"early stop! update best ckpt at epoch: {cur_epoch}, "
                                            f"best recall: {metrics['recall'].avg}, "
                                            f"precision: {metrics['precision'].avg}, "
                                            f"fmeasure: {metrics['fmeasure'].avg}")
                    ms.save_checkpoint(self.eval_net.model,
                                       os.path.join(self.save_ckpt_dir, f"best_rank{self.config.rank_id}.ckpt"))
                    run_context.request_stop()


    def on_train_end(self, run_context):
        cb_params = run_context.original_args()
        if self.rank_id == 0 and self.run_eval:
            self.config.logger.info('best fmeasure is: %s' % self.max_f)
        self.config.logger.info(f'end train, avg fps (except first epoch) is {np.mean(np.array(self.all_fps[1:]))}')
        samples = self.batch_size * self.device_num * cb_params.batch_num
        all_cost = time.time() - self.train_start
        self.config.logger.info(f'training total cost {all_cost} s, samples {samples}, '
                                f'avg fps is {samples * self.epoch_num / all_cost} FPS')
