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
"""Reporter class, modified from cyclegan in modelzoo."""

import logging
import os
import time
from datetime import datetime
from mindspore.train.serialization import save_checkpoint
from src.tools import save_image
from src.dataset import _get_rank_info

class Reporter(logging.Logger):
    """
    This class includes several functions that can save images/checkpoints and print/save logging information.
    """

    def __init__(self, output_path, stage, start_epochs=0, dataset_size=50000, batch_size=128, save_imgs=True):
        super(Reporter, self).__init__("cgan")
        self.output_path = output_path
        self.log_dir = os.path.join(self.output_path, 'log')
        self.imgs_dir = os.path.join(self.output_path, "imgs")
        self.imgs_dir_random = os.path.join(self.imgs_dir, "random_results")
        self.imgs_dir_fixed = os.path.join(self.imgs_dir, "fixed_results")
        self.ckpts_dir = os.path.join(self.output_path, "ckpt")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        if not os.path.exists(self.imgs_dir):
            os.makedirs(self.imgs_dir, exist_ok=True)
        if not os.path.exists(self.imgs_dir_random):
            os.makedirs(self.imgs_dir_random, exist_ok=True)
        if not os.path.exists(self.imgs_dir_fixed):
            os.makedirs(self.imgs_dir_fixed, exist_ok=True)
        if not os.path.exists(self.ckpts_dir):
            os.makedirs(self.ckpts_dir, exist_ok=True)

        rank_size, self.rank_id = _get_rank_info()
        if rank_size > 1:
            self.run_distribute = True
        else:
            self.run_distribute = False
        self.checkpoint_save_iters = 5
        self.save_imgs = save_imgs
        # console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        self.addHandler(console)
        # file handler
        log_name = stage + datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S') + '_rank_{}.log'.format(self.rank_id)
        self.log_fn = os.path.join(self.log_dir, log_name)
        fh = logging.FileHandler(self.log_fn)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.addHandler(fh)
        self.step = 0
        self.batch_size = batch_size
        self.epoch = start_epochs
        self.dataset_size = dataset_size
        self.print_iter = 100
        self.maximum_number_of_ckpt = 5
        self.G_loss = []
        self.D_loss = []
        self.ckpt_list = []
        self.epoch_times = []

    def info(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args, **kwargs)


    def epoch_start(self):
        self.step_start_time = time.time()
        self.epoch_start_time = time.time()
        self.step = 0
        self.epoch += 1
        self.G_loss = []
        self.D_loss = []

    def step_end(self, res_G, res_D):
        """print log when step end."""
        self.step += 1
        loss_D = float(res_D.asnumpy())
        loss_G = float(res_G.asnumpy())

        self.G_loss.append(loss_G)
        self.D_loss.append(loss_D)
        if self.step % self.print_iter == 0:
            step_cost = (time.time() - self.step_start_time) * 1000 / self.print_iter
            losses = "D_loss: {:.2f}, G_loss:{:.2f}".format(loss_D, loss_G)
            info = "epoch[{}] [{}/{}] step cost: {:.2f} ms, {}".format(
                self.epoch, self.step, self.dataset_size, step_cost, losses)
            if self.run_distribute:
                info = "Rank[{}] , {}".format(self.rank_id, info)
            self.info(info)
            self.step_start_time = time.time()

    def epoch_end(self, net):
        """print log and save cgeckpoints when epoch end."""
        epoch_cost = (time.time() - self.epoch_start_time) * 1000
        self.epoch_times.append(epoch_cost)
        pre_step_time = epoch_cost / self.dataset_size
        self.mean_loss_G = sum(self.G_loss) / len(self.G_loss)
        self.mean_loss_D = sum(self.D_loss) / len(self.D_loss)
        info = "epoch [{}] total cost: {:.2f} ms, pre step: {:.2f} ms, D_loss: {:.2f}, G_loss: {:.2f}".format(
            self.epoch, epoch_cost, pre_step_time, self.mean_loss_D, self.mean_loss_G)
        if self.run_distribute:
            info = "Rank[{}] {}".format(self.rank_id, info)
        self.info(info)
        if self.rank_id == 0:
            if self.epoch % self.checkpoint_save_iters == 0:
                g_name = os.path.join(self.ckpts_dir, f"generator{self.epoch}.ckpt")
                d_name = os.path.join(self.ckpts_dir, f"discriminator{self.epoch}.ckpt")
                save_checkpoint(net.G_with_loss.generator, g_name)
                save_checkpoint(net.G_with_loss.discriminator, d_name)
                self.ckpt_list.append(self.epoch)
                if len(self.ckpt_list) > self.maximum_number_of_ckpt:
                    del_epoch = self.ckpt_list[0]
                    os.remove(os.path.join(self.ckpts_dir, f"generator{del_epoch}.ckpt"))
                    os.remove(os.path.join(self.ckpts_dir, f"discriminator{del_epoch}.ckpt"))
                    self.ckpt_list.remove(del_epoch)

    def end_train(self):
        _len = len(self.epoch_times)
        _sum = sum(self.epoch_times)
        epoch_times = _sum / _len
        info = 'total {} epochs, cost {:.2f} ms, pre epoch cost {:.2f}'.format(_len, _sum, epoch_times)
        if self.run_distribute:
            info = "Rank[{}] {}".format(self.rank_id, info)
        self.info(info)
        self.info('==========end train ===============')

    def visualizer(self, img, fixed_img):  #
        if self.save_imgs and self.rank_id == 0 and self.step % self.print_iter == 0:
            save_image(img, os.path.join(self.imgs_dir_random, f"{self.epoch}_{self.step}_img.jpg"), self.batch_size)
            save_image(fixed_img, os.path.join(self.imgs_dir_fixed, f"{self.epoch}_{self.step}_img.jpg"),
                       self.batch_size)

    def visualizer_eval(self, img, step):
        save_image(img, os.path.join(self.imgs_eval_random, f"{step}_img.jpg"), self.batch_size)

    def start_predict(self):
        """start_predict"""
        self.predict_start_time = time.time()
        self.imgs_eval = os.path.join(self.output_path, "eval")
        self.imgs_eval_random = os.path.join(self.imgs_eval, "random_results")
        self.imgs_eval_fixed = os.path.join(self.imgs_eval, "fixed_results")
        if not os.path.exists(self.imgs_eval):
            os.makedirs(self.imgs_eval)
        if not os.path.exists(self.imgs_eval_random):
            os.makedirs(self.imgs_eval_random, exist_ok=True)
        if not os.path.exists(self.imgs_eval_fixed):
            os.makedirs(self.imgs_eval_fixed, exist_ok=True)
        info = 'saved in ' + self.imgs_eval
        self.info(info)

    def end_predict(self, step):
        cost = (time.time() - self.predict_start_time) * 1000
        pre_step_cost = cost / self.dataset_size
        self.info('total {} imgs saved, cost {:.2f} ms, pre img cost {:.2f}'.format(self.batch_size * (step + 1), cost,
                                                                                    pre_step_cost))
        self.info('==========end generate ===============')
