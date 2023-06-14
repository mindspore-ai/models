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

"""The callback function of train and eval phase"""
import os
import time
from datetime import datetime
import numpy as np

from mindspore import Tensor, save_checkpoint
from mindspore.train.callback import Callback


class TrainLog:
    """Create model save directory and log save file"""

    def __init__(self, cfg):
        self.cfg = cfg

        self.save_dir = self.cfg["saveCkpt"]
        self.safe_makedirs(self.save_dir)
        self.logFile_one = self.create_logfile(self.save_dir, "latest")
        self.logFile_best = self.create_logfile(self.save_dir, "best")

        if cfg["train_phase"] == "pre_train_t_net":
            self.save_pre_train_t_net = os.path.join(self.save_dir, "pre_train_t_net")
            self.safe_makedirs(self.save_pre_train_t_net)
            self.logFile_one_t_net = self.create_logfile(self.save_pre_train_t_net, "latest")
            self.logFile_best_t_net = self.create_logfile(self.save_pre_train_t_net, "best")
        elif cfg["train_phase"] == "pre_train_m_net":
            self.save_pre_train_m_net = os.path.join(self.save_dir, "pre_train_m_net")
            self.safe_makedirs(self.save_pre_train_m_net)
            self.logFile_one_m_net = self.create_logfile(self.save_pre_train_m_net, "latest")
            self.logFile_best_m_net = self.create_logfile(self.save_pre_train_m_net, "best")
        else:
            self.save_end_to_end = os.path.join(self.save_dir, "end_to_end")
            self.safe_makedirs(self.save_end_to_end)
            self.logFile_one_end = self.create_logfile(self.save_end_to_end, "latest")
            self.logFile_best_end = self.create_logfile(self.save_end_to_end, "best")

    @staticmethod
    def create_logfile(path_dir, suffix_name):
        file_name = os.path.join(path_dir, "log_{}.txt".format(suffix_name))
        return open(file_name, "a") if os.path.exists(file_name) else open(file_name, "w")

    @staticmethod
    def safe_makedirs(path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

    def close_log_file(self):
        self.logFile_one.close()
        self.logFile_best.close()
        if self.cfg["train_phase"] == "pre_train_t_net":
            self.logFile_one_t_net.close()
            self.logFile_best_t_net.close()
        elif self.cfg["train_phase"] == "pre_train_m_net":
            self.logFile_one_m_net.close()
            self.logFile_best_m_net.close()
        else:
            self.logFile_one_end.close()
            self.logFile_best_end.close()

    def _save_model(self, model, epoch, suffix_name="latest"):
        """
        Save model

        suffix_name:
            latest type：name retain [epoch]
            best type：name fixed，do not retain epoch
        """
        if self.cfg["train_phase"] == "pre_train_t_net":
            if suffix_name == "latest":
                file_path = os.path.join(self.save_pre_train_t_net, "semantic_hm_{}_{}.ckpt".format(suffix_name, epoch))
            else:
                file_path = os.path.join(self.save_pre_train_t_net, "semantic_hm_{}.ckpt".format(suffix_name))
        elif self.cfg["train_phase"] == "pre_train_m_net":
            if suffix_name == "latest":
                file_path = os.path.join(self.save_pre_train_m_net, "semantic_hm_{}_{}.ckpt".format(suffix_name, epoch))
            else:
                file_path = os.path.join(self.save_pre_train_m_net, "semantic_hm_{}.ckpt".format(suffix_name))
        else:
            if suffix_name == "latest":
                file_path = os.path.join(self.save_end_to_end, "semantic_hm_{}_{}.ckpt".format(suffix_name, epoch))
            else:
                file_path = os.path.join(self.save_end_to_end, "semantic_hm_{}.ckpt".format(suffix_name))

        if self.cfg["keep_checkpoint_max"] != "0":
            save_checkpoint(model, file_path)

    def save_model(self, model, epoch, mode="latest"):
        if mode == "latest":
            self._save_model(model, epoch, suffix_name="latest")
        else:
            self._save_model(model, epoch, suffix_name="best")

    def save_log(self, log, mode="latest"):
        if mode == "latest":
            self.logFile_one.write(log + "\n")
            if self.cfg["train_phase"] == "pre_train_t_net":
                self.logFile_one_t_net.write(log + "\n")
            elif self.cfg["train_phase"] == "pre_train_m_net":
                self.logFile_one_m_net.write(log + "\n")
            else:
                self.logFile_one_end.write(log + "\n")
        else:
            self.logFile_best.write(log + "\n")
            if self.cfg["train_phase"] == "pre_train_t_net":
                self.logFile_best_t_net.write(log + "\n")
            elif self.cfg["train_phase"] == "pre_train_m_net":
                self.logFile_best_m_net.write(log + "\n")
            else:
                self.logFile_best_end.write(log + "\n")

    def clear_redundant_ckpt(self):
        """
        Clear redundant files
        """
        if self.cfg["keep_checkpoint_max"] == "all" or self.cfg["keep_checkpoint_max"] == "0":
            return

        path_dir = os.path.join(self.cfg["saveCkpt"], self.cfg["train_phase"])
        if not os.path.exists(path_dir):
            return

        list_file = os.listdir(path_dir)
        for i in range(len(list_file) - 1, -1, -1):
            if "semantic_hm_latest" not in list_file[i]:
                del list_file[i]
        list_file = sorted(list_file, key=lambda x: int(x.split("/")[-1].split("_")[-1].split(".")[0]))

        len_file = len(list_file)
        nums = int(self.cfg["keep_checkpoint_max"])
        if len_file > nums:
            for i in range(len_file - nums):
                os.remove(os.path.join(path_dir, list_file[i]))


class TrainCallBack(Callback):
    """Train callback function class for 0 card"""

    def __init__(self, cfg, network, model, eval_callback, eval_dataset, cur_epoch, per_print_times=1):
        self.cfg = cfg
        self.trainlog = TrainLog(cfg)

        self.network = network
        self.model = model
        self.eval_callback = eval_callback
        self.eval_dataset = eval_dataset

        self.eval_per_epoch = cfg["save_epoch"]
        self.epoch_per_eval = {"epoch": [], "sad": []}

        self.cur_epoch = cur_epoch
        self._per_print_times = per_print_times

        self.stage = cfg["train_phase"]
        self.best_sad = None

        self.start_time = time.time()
        self.time_step_begin = time.time()

        self._samples_num_t = 0  # loss nums
        self._loss_t = 0

    def step_begin(self, run_context):
        self.time_step_begin = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        self._samples_num_t += 1
        self._loss_t += loss

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError(
                "epoch: {} step: {}. Invalid loss, terminating training.".format(
                    cb_params.cur_epoch_num + self.cur_epoch, cur_step_in_epoch
                )
            )
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print(
                "train epoch: %s step: %s, loss: %s, speed: %s"
                % (
                    cb_params.cur_epoch_num + self.cur_epoch,
                    cur_step_in_epoch,
                    loss,
                    time.time() - self.time_step_begin,
                ),
                flush=True,
            )

    def epoch_begin(self, run_context):
        self.start_time = time.time()
        self._samples_num_t = 0
        self._loss_t = 0

    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        loss = self._loss_t / (self._samples_num_t + 1e-6)

        if cur_epoch % self.eval_per_epoch == 0:
            current_time = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
            log = "{} {} [{}/{}]\tstage: {}\tspeed: {:.5f}s\tlr: {:.5f}\tloss: {:.5f}\t".format(
                current_time,
                "train",
                cur_epoch + self.cur_epoch,
                self.cfg["nEpochs"],
                self.cfg["train_phase"],
                time.time() - self.start_time,
                float(self.cfg["lr"]),
                loss,
            )
            print(log)
            self.trainlog.save_log(log, mode="latest")
            self.trainlog.save_model(self.network, cur_epoch + self.cur_epoch, mode="latest")
            self.trainlog.clear_redundant_ckpt()

            # eval
            if self.cfg["train_phase"] != "pre_train_t_net":
                t0 = time.time()
                ret = self.model.eval(self.eval_dataset, callbacks=self.eval_callback, dataset_sink_mode=False)
                current_time = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
                log = "{} {} [{}/{}]\tstage: {}\tspeed: {:.5f}s\tlr: {:.5f}\tloss: {:.5f}\tsad: {:.5f}\t".format(
                    current_time,
                    "eval",
                    cur_epoch + self.cur_epoch,
                    self.cfg["nEpochs"],
                    self.cfg["train_phase"],
                    time.time() - t0,
                    float(self.cfg["lr"]),
                    ret["sad"][1],
                    np.float(ret["sad"][0]),
                )
                print(log)
                self.trainlog.save_log(log, mode="latest")

                if self.best_sad is None:
                    self.best_sad = ret["sad"][0]
                if ret["sad"][0] <= self.best_sad:
                    self.best_sad = ret["sad"][0]
                    self.trainlog.save_log(log, mode="best")
                    self.trainlog.save_model(self.network, cur_epoch + self.cur_epoch, mode="best")

                self.epoch_per_eval["epoch"].append(cur_epoch + self.cur_epoch)
                self.epoch_per_eval["sad"].append(ret["sad"][0])

    def end(self, run_context):
        self.trainlog.close_log_file()
        if self.stage == "end_to_end":
            list_sad = self.epoch_per_eval["sad"]
            print(
                "metrics sad: num: {}\tsum_sad: {}\tave_sad: {}".format(
                    len(list_sad), np.sum(list_sad), np.mean(list_sad)
                )
            )


class LossMonitorSub(Callback):
    """Callback function class for not 0 card"""

    def __init__(self, cur_epoch=0, per_print_times=1):
        super(LossMonitorSub, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("The argument 'per_print_times' must be int and >= 0, but got {}".format(per_print_times))
        self.cur_epoch = cur_epoch
        self._per_print_times = per_print_times

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError(
                "epoch: {} step: {}. Invalid loss, terminating training.".format(
                    cb_params.cur_epoch_num + self.cur_epoch, cur_step_in_epoch
                )
            )
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print(
                "epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num + self.cur_epoch, cur_step_in_epoch, loss),
                flush=True,
            )


class EvalCallBack(Callback):
    """Evaluation callback function class"""

    def __init__(self, per_print_times=1):
        self._per_print_times = per_print_times

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[2], Tensor) and isinstance(loss[2].asnumpy(), np.ndarray):
                loss = loss[2]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print("eval step: %s, loss is %s" % (cur_step_in_epoch, loss), flush=True)
