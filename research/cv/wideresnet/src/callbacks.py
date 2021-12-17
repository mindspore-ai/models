"""Callbacks for loss monitoring and checkpoints saving"""

import os
import stat
import time
import numpy as np
from mindspore.train.callback import LossMonitor, Callback
from mindspore.train.summary import SummaryRecord
from mindspore import Tensor
from mindspore import save_checkpoint
from mindspore import log as logger
from mindspore.train.serialization import load_param_into_net
from src.model_utils.config import config as cfg

class CustomLossMonitor(LossMonitor):
    """Own Loss Monitor that uses specified Summary Record instance"""

    def __init__(self, summary_record: SummaryRecord, mode: str, frequency: int = 1):
        super(CustomLossMonitor, self).__init__()
        self._summary_record = summary_record
        self._mode = mode
        if frequency > 1:
            self._freq = frequency

    def epoch_begin(self, run_context):
        """Called before each epoch beginning."""
        if self._mode != 'eval':
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

class PredictionsCallback(Callback):

    """Own Loss Monitor that uses specified Summary Record instance"""

    def __init__(self, summary_record: SummaryRecord, summary_freq: int = 1):
        super(PredictionsCallback, self).__init__()
        self._summary_record = summary_record
        self._freq = summary_freq

    def step_end(self, run_context):
        """Called after each step finished."""
        cb_params = run_context.original_args()
        step = cb_params.cur_step_num
        if step % self._freq == 0:
            self._summary_record.record(step)


class EvalCallback(Callback):
    """
    save best ckpt
    """
    def __init__(self, eval_model, model, eval_dataset, ckpt_dir_path,
                 modelart, summary_record=None, eval_freq=1, start_epoch=1):
        super(EvalCallback, self).__init__()
        self.eval_model = eval_model
        self.model = model
        self.eval_dataset = eval_dataset
        self.cpkt_dir = ckpt_dir_path
        self.acc = cfg.acc_lower_bound
        self.cur_acc = 0.0
        self.modelart = modelart
        self.metrics_name = 'top_1_accuracy'
        self.eval_start_epoch = start_epoch
        self.interval = eval_freq
        self.best_ckpt_path = 'best.ckpt'
        self._summary_record = summary_record
        self.step_wise = False
        self.step_interval = 30

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
        """
        epoch end
        """
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        if cur_epoch >= self.eval_start_epoch and (cur_epoch - self.eval_start_epoch) % self.interval == 0:
            load_param_into_net(self.eval_model.train_network, self.model.train_network.parameters_dict())

            eval_start = time.time()
            result = self.eval_model.eval(self.eval_dataset, dataset_sink_mode=False)
            eval_cost = time.time() - eval_start
            self.cur_acc = result['top_1_accuracy']
            print("epoch: {}, {}: {}, eval_cost:{:.2f}".format(cur_epoch, self.metrics_name, self.cur_acc, eval_cost),
                  flush=True)

            #++++++++Summary Record++++++++++
            if self._summary_record is not None:
                self._summary_record.add_value('scalar', self.metrics_name, Tensor(self.cur_acc))
                self._summary_record.add_value('scalar', "eval_cost", Tensor(eval_cost))
                self._summary_record.record(cb_params.cur_step_num)
            #++++++++++++++++++++++++++++++++

            if self.cur_acc > self.acc:
                self.acc = self.cur_acc
                self.step_wise = cfg.save_best_ckpt
                if self.modelart:
                    import moxing as mox
                    mox.file.copy_parallel(src_url=cfg.save_checkpoint_path, dst_url=self.cpkt_dir)
                print("Step-wise evaluation enabled. Accuracy reached: ", self.acc)

            if self.cur_acc > 0.93:
                self.interval = 1

    def step_end(self, run_context):
        """
        step end
        """

        cb_params = run_context.original_args()
        if (self.step_wise and cb_params.cur_step_num % self.step_interval == 0):
            load_param_into_net(self.eval_model.train_network, self.model.train_network.parameters_dict())
            result = self.eval_model.eval(self.eval_dataset, dataset_sink_mode=False)
            self.cur_acc = result['top_1_accuracy']
            print("cur_acc is", self.cur_acc)

            if result['top_1_accuracy'] > self.acc:
                self.acc = result['top_1_accuracy']
                if os.path.exists(self.best_ckpt_path):
                    self.remove_ckpoint_file(self.best_ckpt_path)
                self.best_ckpt_path = self.cpkt_dir \
                                        + "WideResNet_best" \
                                        + "_epoch-" + str(cb_params.cur_epoch_num) \
                                        + "_acc-" + str(self.acc) \
                                        + ".ckpt"
                save_checkpoint(cb_params.train_network, self.best_ckpt_path)
                print("update best checkpoint at: {}".format(self.best_ckpt_path), flush=True)
                if self.modelart:
                    import moxing as mox
                    mox.file.copy_parallel(src_url=cfg.save_checkpoint_path, dst_url=self.cpkt_path)
                print("Save the maximum accuracy checkpoint,the accuracy is", self.acc)

    def end(self, run_context):
        ''' end of training'''
        cb_params = run_context.original_args()
        print("Training ended!")
        print(f"Epoch: {cb_params.cur_epoch_num} Max top 1 acc. : {self.acc}")
