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
""" eval_callback """
from pathlib import Path
import stat
import os
import numpy as np
from mindspore.nn.metrics.metric import Metric
from mindspore.train.callback import Callback, LossMonitor
from mindspore.train.summary import SummaryRecord
from mindspore import Tensor
from mindspore import log as logger
from mindspore import save_checkpoint
from src.utils.metrics import SoftmaxCrossEntropyLoss



class EvalCallBack(Callback):
    """Precision verification using callback function."""

    def __init__(self, models, eval_dataset, eval_per_epochs, epochs_per_eval, summary_record, ckpt_dir='./'):
        super(EvalCallBack, self).__init__()
        self.models = models
        self.eval_dataset = eval_dataset
        self.eval_per_epochs = eval_per_epochs
        self.epochs_per_eval = epochs_per_eval
        self._summary_record = summary_record
        self.val_loss = 1e7
        self.best_ckpt_path = ''
        self.cpkt_dir = ckpt_dir


    def remove_ckpoint_file(self, file_name):
        """Remove the specified checkpoint file from this checkpoint manager and also from the directory."""
        try:
            os.chmod(file_name, stat.S_IWRITE)
            os.remove(file_name)
        except OSError:
            logger.warning("OSError, failed to remove the older ckpt file %s.", file_name)
        except ValueError:
            logger.warning("ValueError, failed to remove the older ckpt file %s.", file_name)

    def epoch_end(self, run_context):
        """ evaluate during training """
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epochs == 0:
            val_loss = self.models.eval(self.eval_dataset, dataset_sink_mode=False)
            self.epochs_per_eval["epoch"].append(cur_epoch)
            self.epochs_per_eval["val_loss"].append(val_loss)
            self._summary_record.add_value('scalar', 'loss_eval', Tensor(val_loss['val_loss']))
            self._summary_record.record(cb_param.cur_step_num)
            print(val_loss)

            cur_loss = val_loss['val_loss']
            if cur_loss < self.val_loss:
                self.val_loss = cur_loss
                if os.path.exists(self.best_ckpt_path):
                    self.remove_ckpoint_file(self.best_ckpt_path)
                self.best_ckpt_path = self.cpkt_dir \
                                        + "cpnet_best" \
                                        + "_epoch-" + str(cur_epoch) \
                                        + "_val-loss-" + str(cur_loss).replace('.', '_') \
                                        + ".ckpt"

                try:
                    path = Path(self.best_ckpt_path)
                    if not path.exists():
                        path.parent.mkdir(parents=True)
                except OSError:
                    logger.warning("OSError, failed to create dir to store ckpt file %s.", self.best_ckpt_path)
                f = open(self.best_ckpt_path, "x")
                f.close()
                save_checkpoint(cb_param.train_network, self.best_ckpt_path)
                print("update best checkpoint at: {}".format(self.best_ckpt_path), flush=True)
                print("Save the minimum validation loss checkpoint,the validation loss is", cur_loss)


        if cur_epoch > 90:
            self.eval_per_epochs = 1

        if cur_epoch % 20 == 0:
            ckpt_path = self.cpkt_dir \
                                    + "cpnet_" \
                                    + "_epoch-" + str(cur_epoch) \
                                    + ".ckpt"
            try:
                path = Path(ckpt_path)
                if not path.exists():
                    path.parent.mkdir(parents=True)
            except OSError:
                logger.warning("OSError, failed to create dir to store ckpt file %s.", ckpt_path)
            f = open(ckpt_path, "x")
            f.close()
            save_checkpoint(cb_param.train_network, ckpt_path)
            print("Save checkpoint at: {}".format(self.best_ckpt_path), flush=True)


    def get_dict(self):
        """ get eval dict"""
        return self.epochs_per_eval

class cpnet_metric(Metric):
    """ callback class """
    def __init__(self, num_classes=150, ignore_label=255):
        super(cpnet_metric, self).__init__()
        self.loss = SoftmaxCrossEntropyLoss(num_classes, ignore_label)
        self.val_loss = 0
        self.count = 0
        self.clear()

    def clear(self):
        """ clear the init value """
        self.val_loss = 0
        self.count = 0

    def update(self, *inputs):
        """ update the calculate process """
        if len(inputs) != 2:
            raise ValueError('Expect inputs (y_pred, y), but got {}'.format(len(inputs)))
        predict, _, _ = inputs[0]
        the_loss = self.loss(predict, inputs[1])
        self.val_loss += the_loss
        self.count += 1

    def eval(self):
        """ return the result """
        return self.val_loss / float(self.count)

class CustomLossMonitor(LossMonitor):
    """Own Loss Monitor that uses specified Summary Record instance"""

    def __init__(self, summary_record: SummaryRecord, mode: str, frequency: int = 1):
        super(CustomLossMonitor, self).__init__()
        self._summary_record = summary_record
        self._mode = mode
        self._freq = frequency

    def epoch_begin(self, run_context):
        """Called before each epoch beginning."""
        if  self._mode != 'eval':
            super(CustomLossMonitor, self).epoch_begin(run_context)

    def epoch_end(self, run_context):
        """Called after each epoch finished."""
        if self._mode != 'eval':
            cb_params = run_context.original_args()

            if cb_params.cur_step_num % self._freq == 0:
                step_loss = cb_params.net_outputs

                if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
                    step_loss = step_loss[0]
                if isinstance(step_loss, Tensor):
                    step_loss = np.mean(step_loss.asnumpy())

                self._summary_record.add_value('scalar', 'loss_' + self._mode, Tensor(step_loss))
                self._summary_record.record(cb_params.cur_step_num)

            super(CustomLossMonitor, self).epoch_end(run_context)

    def step_begin(self, run_context):
        """Called before each step beginning."""
        if self._mode != 'eval':
            super(CustomLossMonitor, self).step_begin(run_context)

    def step_end(self, run_context):
        """Called after each step finished."""
        cb_params = run_context.original_args()

        if cb_params.cur_step_num % self._freq == 0:
            step_loss = cb_params.net_outputs

            if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
                step_loss = step_loss[0]
            if isinstance(step_loss, Tensor):
                step_loss = np.mean(step_loss.asnumpy())

            self._summary_record.add_value('scalar', 'loss_' + self._mode, Tensor(step_loss))
            self._summary_record.record(cb_params.cur_step_num)

        if self._mode != 'eval':
            super(CustomLossMonitor, self).step_end(run_context)

    def end(self, run_context):
        """Called once after network training."""
        if self._mode != 'eval':
            super(CustomLossMonitor, self).end(run_context)
