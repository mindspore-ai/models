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
"""Reporter class"""
import logging
import os
import time
from datetime import datetime

from mindspore.train.serialization import save_checkpoint

from src.util import make_joined_image
from src.util import save_image

_S_IWRITE = 128


class Reporter(logging.Logger):
    """Save images/checkpoints and print/save logging information"""

    def __init__(self, config):
        super().__init__('HiFaceGAN')
        self.log_dir = os.path.join(config.outputs_dir, 'log')
        self.imgs_dir = os.path.join(config.outputs_dir, 'imgs')
        self.ckpts_dir = os.path.join(config.outputs_dir, 'ckpt')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        if not os.path.exists(self.imgs_dir):
            os.makedirs(self.imgs_dir, exist_ok=True)
        if not os.path.exists(self.ckpts_dir):
            os.makedirs(self.ckpts_dir, exist_ok=True)
        self.rank = config.rank
        self.save_checkpoint_epochs = config.save_checkpoint_epochs
        # console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        self.addHandler(console)
        # file handler
        log_name = datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S') + '_rank_{}.log'.format(self.rank)
        self.log_fn = os.path.join(self.log_dir, log_name)
        fh = logging.FileHandler(self.log_fn)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.addHandler(fh)
        self.save_args(config)
        self.step = 0
        self.epoch = 0
        self.dataset_size = config.dataset_size
        self.device_num = config.group_size
        self.print_iter = config.print_iter
        self.G_loss = []
        self.D_loss = []
        self._checkpoints_list = []
        self.keep_checkpoint_max = config.keep_checkpoint_max

    def info(self, msg, *config, **kwargs):
        """Print info message"""
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, config, **kwargs)

    def save_args(self, config):
        """Save args"""
        self.info('Args:')
        args_dict = vars(config)
        for key in args_dict.keys():
            self.info('--> %s: %s', key, args_dict[key])
        self.info('')

    def remove_ckpoint_file(self, file_name):
        """Remove the specified checkpoint file from this checkpoint manager and also from the directory."""
        try:
            os.chmod(file_name, _S_IWRITE)
            os.remove(file_name)
            self._checkpoints_list.remove(file_name)
        except OSError:
            self.warning("OSError, failed to remove the older ckpt file %s.", file_name)
        except ValueError:
            self.warning("ValueError, failed to remove the older ckpt file %s.", file_name)

    def remove_oldest_ckpoint_file(self):
        """Remove the oldest checkpoint file from this checkpoint manager and also from the directory."""
        ckpoint_files = sorted(self._checkpoints_list, key=os.path.getmtime)
        self.remove_ckpoint_file(ckpoint_files[0])

    def epoch_start(self):
        """Function to run at the start of the epoch"""
        self.step_start_time = time.time()
        self.epoch_start_time = time.time()
        self.step = 0
        self.epoch += 1
        self.G_loss = []
        self.D_loss = []

    def step_end(self, res_G, res_D):
        """Function to run at the end of the epoch"""
        self.step += 1
        res_G = [float(loss.asnumpy()) for loss in res_G]
        res_D = [float(loss.asnumpy()) for loss in res_D]

        self.G_loss.append(res_G[0])
        self.D_loss.append(res_D[0])

        losses_str = 'G_loss: {:.2f}, D_loss: {:.2f}, G_vgg_loss: {:.2f}, G_gan_loss: {:.2f}, ' \
                     'G_gan_feat_loss: {:.2f}, D_fake_loss: {:.2f}, D_real_loss: {:.2f}'

        if self.step % self.print_iter == 0:
            step_cost = (time.time() - self.step_start_time) * 1000 / self.print_iter
            losses = losses_str.format(res_G[0], res_D[0], res_G[1], res_G[2], res_G[3], res_D[1], res_D[2])
            self.info('Epoch[{}] [{}/{}] step cost: {:.2f} ms, {}'.format(
                self.epoch, self.step, self.dataset_size, step_cost, losses))
            self.step_start_time = time.time()

    def epoch_end(self, net):
        """Print log and save checkpoints when epoch end"""
        epoch_cost = (time.time() - self.epoch_start_time) * 1000
        per_step_time = epoch_cost / self.dataset_size
        mean_loss_G = sum(self.G_loss) / self.dataset_size
        mean_loss_D = sum(self.D_loss) / self.dataset_size
        self.info('Epoch [{}] total cost: {:.2f} ms, per step: {:.2f} ms, G_loss: {:.2f}, D_loss: {:.2f}'.format(
            self.epoch, epoch_cost, per_step_time, mean_loss_G, mean_loss_D))

        if self.epoch % self.save_checkpoint_epochs == 0:
            ckpt_path = os.path.join(self.ckpts_dir, f'generator_{self.epoch}.ckpt')
            save_checkpoint(net.generator_with_loss_cell.generator, ckpt_path)
            self._checkpoints_list.append(ckpt_path)

            if len(self._checkpoints_list) > self.keep_checkpoint_max:
                self.remove_oldest_ckpoint_file()

    def visualizer(self, lq, hq, generated):
        """Save images"""
        if self.step % self.dataset_size == 0:
            joined_image = make_joined_image(lq[0], generated[0], hq[0])
            save_image(joined_image[:, :, ::-1], os.path.join(self.imgs_dir, f'{self.epoch}.png'))

    def start_predict(self):
        """Function to run at the start of the evaluation"""
        self.predict_start_time = time.time()
        self.info('========== start predict ===============')

    def end_predict(self):
        """Function to run at the end of the evaluation"""
        cost = (time.time() - self.predict_start_time) / 60
        self.info('total cost {:.2f} min'.format(cost))
        self.info('========== end predict ===============\n')
