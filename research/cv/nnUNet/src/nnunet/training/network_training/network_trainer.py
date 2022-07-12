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

"""network trainer"""

import sys
from time import time, sleep
from collections import OrderedDict
from abc import abstractmethod
from datetime import datetime
from typing import Tuple
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import mindspore
from mindspore import save_checkpoint as sc
from mindspore import load_param_into_net, nn
from batchgenerators.utilities.file_and_folder_operations import join, \
    isfile, save_pickle, load_pickle, maybe_mkdir_p, \
    os
from src.nnunet.network_architecture.neural_network import SegmentationNetwork
from sklearn.model_selection import KFold

from tqdm import trange




class NetworkTrainer():
    def __init__(self, deterministic=True, fp16=False):
        """
        A generic class that can train almost any neural network (RNNs excluded). It provides basic functionality such
        as the training loop, tracking of training and validation losses (and the target metric if you implement it)
        Training can be terminated early if the validation loss (or the target metric if implemented) do not improve
        anymore. This is based on a moving average (MA) of the loss/metric instead of the raw values to get more smooth
        results.

        """
        self.fp16 = fp16
        self.amp_grad_scaler = None
        print("deterministic out")
        if deterministic:
            print("deterministic in")
            np.random.seed(12345)
            mindspore.common.set_seed(12345)
            self.seed_train = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.seed_val = [0, 0, 0, 0, 0, 0]
        else:
            pass

        # SET THESE IN self.initialize()
        self.network: Tuple[SegmentationNetwork, nn.DataParallel] = None
        self.optimizer = None
        self.lr_scheduler = None
        self.tr_gen = self.val_gen = None
        self.was_initialized = False

        # SET THESE IN INIT
        self.output_folder = None
        self.fold = None
        self.loss = None
        self.dataset_directory = None

        # SET THESE IN LOAD_DATASET OR DO_SPLIT
        self.dataset = None
        self.dataset_tr = self.dataset_val = None

        # THESE DO NOT NECESSARILY NEED TO BE MODIFIED
        self.patience = 10
        self.val_eval_criterion_alpha = 0.9  # alpha * old + (1-alpha) * new

        self.train_loss_MA_alpha = 0.93
        self.train_loss_MA_eps = 5e-4

        self.max_num_epochs = 200
        self.num_batches_per_epoch = 1
        self.num_val_batches_per_epoch = 50
        self.also_val_in_tr_mode = False
        self.lr_threshold = 1e-6

        # LEAVE THESE ALONE
        self.val_eval_criterion_MA = None
        self.train_loss_MA = None
        self.best_val_eval_criterion_MA = None
        self.best_MA_tr_loss_for_patience = None
        self.best_epoch_based_on_MA_tr_loss = None
        self.all_tr_losses = []
        self.all_val_losses = []
        self.all_val_losses_tr_mode = []
        self.all_val_eval_metrics = []  # does not have to be used
        self.epoch = 0
        self.log_file = None
        self.deterministic = deterministic

        self.use_progress_bar = False
        if 'nnunet_use_progress_bar' in os.environ.keys():
            self.use_progress_bar = bool(int(os.environ['nnunet_use_progress_bar']))

        # Settings for saving checkpoints
        self.save_every = 50
        self.save_latest_only = True
        self.save_intermediate_checkpoints = True
        self.save_best_checkpoint = True
        self.save_final_checkpoint = True

    @abstractmethod
    def initialize(self, training=True):
        """initialize function"""
        return None

    @abstractmethod
    def load_dataset(self):
        """load dataset"""
        return None

    def do_split(self):
        """
        This is a suggestion for if your dataset is a dictionary (my personal standard)

        """
        splits_file = join(self.dataset_directory, "splits_final.pkl")
        if not isfile(splits_file):
            self.print_to_log_file("Creating new split...")
            splits = []
            all_keys_sorted = np.sort(list(self.dataset.keys()))
            kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
            for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                train_keys = np.array(all_keys_sorted)[train_idx]
                test_keys = np.array(all_keys_sorted)[test_idx]
                splits.append(OrderedDict())
                splits[-1]['train'] = train_keys
                splits[-1]['val'] = test_keys
            save_pickle(splits, splits_file)

        splits = load_pickle(splits_file)

        if self.fold == "all":
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            tr_keys = splits[self.fold]['train']
            val_keys = splits[self.fold]['val']

        tr_keys.sort()
        val_keys.sort()

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]

        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def plot_progress(self):
        """
        Should probably by improved
        """
        try:
            font = {'weight': 'normal',
                    'size': 18}

            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            x_values = list(range(self.epoch + 1))

            ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label="loss_tr")

            ax.plot(x_values, self.all_val_losses, color='r', ls='-', label="loss_val, train=False")

            if self.all_val_losses_tr_mode:
                ax.plot(x_values, self.all_val_losses_tr_mode, color='g', ls='-', label="loss_val, train=True")
            if len(self.all_val_eval_metrics) == len(x_values):
                ax2.plot(x_values, self.all_val_eval_metrics, color='g', ls='--', label="evaluation metric")

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("evaluation metric")
            ax.legend()
            ax2.legend(loc=9)

            fig.savefig(join(self.output_folder, "progress.png"))
            plt.close()
        except IOError:
            self.print_to_log_file("failed to plot: ", sys.exc_info())

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        """print information to log file"""
        timestamp = time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ("%s:" % dt_object, *args)

        if self.log_file is None:
            maybe_mkdir_p(self.output_folder)
            timestamp = datetime.now()
            self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                 (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                  timestamp.second))
            with open(self.log_file, 'w') as f:
                f.write("Starting... \n")
        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                sleep(0.5)
                ctr += 1
        if also_print_to_console:
            print(*args)

    def save_checkpoint(self, fname, save_optimizer=True):
        """save checkpoint"""
        start_time = time()
        sc(self.network, fname)
        self.print_to_log_file("done, saving took %.2f seconds" % (time() - start_time))

    def load_best_checkpoint(self, train=True):
        """load best checkpoint"""
        if self.fold is None:
            raise RuntimeError("Cannot load best checkpoint if self.fold is None")
        if isfile(join(self.output_folder, "model_best.ckpt")):
            self.load_checkpoint(join(self.output_folder, "model_best.ckpt"), train=train)
        else:
            self.print_to_log_file("WARNING! model_best.ckpt does not exist! Cannot load best checkpoint. Falling "
                                   "back to load_latest_checkpoint")
            self.load_latest_checkpoint(train)

    def load_latest_checkpoint(self, train=True):
        """load latest checkpoint"""
        if isfile(join(self.output_folder, "model_final_checkpoint.ckpt")):
            return self.load_checkpoint(join(self.output_folder, "model_final_checkpoint.ckpt"), train=train)
        if isfile(join(self.output_folder, "model_latest.ckpt")):
            return self.load_checkpoint(join(self.output_folder, "model_latest.ckpt"), train=train)
        if isfile(join(self.output_folder, "model_best.ckpt")):
            return self.load_best_checkpoint(train)
        raise RuntimeError("No checkpoint found")

    def load_final_checkpoint(self, train=False):
        """load final checkpoint"""
        filename = join(self.output_folder, "model_final_checkpoint.ckpt")
        if not isfile(filename):
            raise RuntimeError("Final checkpoint not found. Expected: %s. Please finish the training first." % filename)
        return self.load_checkpoint(filename, train=train)

    def load_checkpoint(self, fname, train=True):
        """load checkpoint"""
        self.print_to_log_file("loading checkpoint", fname, "train=", train)
        if not self.was_initialized:
            self.initialize(train)

        saved_model = mindspore.load_checkpoint(fname)
        self.load_checkpoint_ram(saved_model, train)

    @abstractmethod
    def initialize_network(self):
        """initialize network"""
        return None

    @abstractmethod
    def initialize_optimizer_and_scheduler(self):
        """initialize optimizer and scheduler"""
        return None

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram

        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()

        for k, value in checkpoint.items():
            key = k
            new_state_dict[key] = value

        load_param_into_net(self.network, new_state_dict)
        if train:
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and \
                    checkpoint['lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

        # load best loss (if present)
        if 'best_stuff' in checkpoint.keys():
            self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA = \
                checkpoint[
                    'best_stuff']

        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                                   "due to an old bug and should only appear when you are loading old models. New "
                                   "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]

    def _maybe_init_amp(self):
        """init amp"""
        return None

    def plot_network_architecture(self):
        """plot network architecture"""
        return None

    def run_training(self):
        """run training"""
        maybe_mkdir_p(self.output_folder)
        self.plot_network_architecture()

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.set_train(mode=True)

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for _ in tbar:
                        tbar.set_description("Epoch {}/{}".format(self.epoch + 1, self.max_num_epochs))
                        l = self.run_iteration(self.tr_gen, True, True)
                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
            else:
                for _ in range(self.num_batches_per_epoch):
                    l = self.run_iteration(self.tr_gen, True, True)
                    train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            # validation with train=False
            val_losses = []
            for _ in range(self.num_val_batches_per_epoch):
                l = self.run_iteration(self.val_gen, False, True)
                val_losses.append(l)
            self.all_val_losses.append(np.mean(val_losses))
            self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

            # validate
            if self.also_val_in_tr_mode:
                self.network.set_train(mode=True)
                # validation with train=True
                val_losses = []
                for _ in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False)
                    val_losses.append(l)
                self.all_val_losses_tr_mode.append(np.mean(val_losses))
                self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.
        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.ckpt"))
        if isfile(join(self.output_folder, "model_latest.ckpt")):
            os.remove(join(self.output_folder, "model_latest.ckpt"))
        if isfile(join(self.output_folder, "model_latest.ckpt.pkl")):
            os.remove(join(self.output_folder, "model_latest.ckpt.pkl"))

    def maybe_update_lr(self):
        """maybe update learning rate"""
        self.lr_scheduler.step(self.epoch + 1)
        self.print_to_log_file("lr is now (scheduler) %s" % str(self.optimizer.param_groups[0]['lr']))

    def maybe_save_checkpoint(self):
        """
        Saves a checkpoint every save_ever epochs.

        """
        if self.save_intermediate_checkpoints and (self.epoch % self.save_every == (self.save_every - 1)):
            self.print_to_log_file("saving scheduled checkpoint file...")
            if not self.save_latest_only:
                self.save_checkpoint(join(self.output_folder, "model_ep_%03.0d.ckpt" % (self.epoch + 1)))
            self.save_checkpoint(join(self.output_folder, "model_latest.ckpt"))
            self.print_to_log_file("done")

    def update_eval_criterion_MA(self):
        """
        If self.all_val_eval_metrics is unused (len=0) then we fall back to using -self.all_val_losses
        for the MA to determine early stopping
        (not a minimization, but a maximization of a metric and therefore the - in the latter case)

        """
        if self.val_eval_criterion_MA is None:
            if not self.all_val_eval_metrics:
                self.val_eval_criterion_MA = - self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.all_val_eval_metrics[-1]
        else:
            if not self.all_val_eval_metrics:

                # We here use alpha * old - (1 - alpha) * new because new in this case is the vlaidation loss and lower
                # is better, so we need to negate it.
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA - (
                    1 - self.val_eval_criterion_alpha) * \
                    self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA + (
                    1 - self.val_eval_criterion_alpha) * \
                    self.all_val_eval_metrics[-1]

    def manage_patience(self):
        """manage patience"""
        # update patience
        continue_training = True
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_on_MA_tr_loss were not yet initialized,
            # initialize them
            if self.best_MA_tr_loss_for_patience is None:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA

            if self.best_epoch_based_on_MA_tr_loss is None:
                self.best_epoch_based_on_MA_tr_loss = self.epoch

            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA

            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                if self.save_best_checkpoint: self.save_checkpoint(join(self.output_folder, "model_best.ckpt"))

            if self.train_loss_MA + self.train_loss_MA_eps < self.best_MA_tr_loss_for_patience:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                self.best_epoch_based_on_MA_tr_loss = self.epoch
            else:
                pass

            if self.epoch - self.best_epoch_based_on_MA_tr_loss > self.patience:
                pass

            else:
                pass

        return continue_training

    def on_epoch_end(self):
        """epoch end function"""
        self.finish_online_evaluation()

        # metrics
        self.plot_progress()
        self.maybe_update_lr()
        self.maybe_save_checkpoint()
        self.update_eval_criterion_MA()
        continue_training = self.manage_patience()
        return continue_training

    def update_train_loss_MA(self):
        """update train loss"""
        if self.train_loss_MA is None:
            self.train_loss_MA = self.all_tr_losses[-1]
        else:
            self.train_loss_MA = self.train_loss_MA_alpha * self.train_loss_MA + (1 - self.train_loss_MA_alpha) * \
                                 self.all_tr_losses[-1]

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """run iteration"""
        return 0

    def run_online_evaluation(self, *args, **kwargs):
        """run online evaluation"""
        return None

    def finish_online_evaluation(self):
        """finish online evaluation"""
        return None

    @abstractmethod
    def validate(self, *args, **kwargs):
        """validate"""
        return None

    def find_lr(self, num_iters=1000, init_value=1e-6, final_value=10., beta=0.98):
        """
        stolen and adapted from here: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html

        """
        import math
        self._maybe_init_amp()
        mult = (final_value / init_value) ** (1 / num_iters)
        lr = init_value
        self.optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        losses = []
        log_lrs = []

        for batch_num in range(1, num_iters + 1):
            # +1 because this one here is not designed to have negative loss...
            loss = self.run_iteration(self.tr_gen, do_backprop=True, run_online_evaluation=False).data.item() + 1

            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss
            smoothed_loss = avg_loss / (1 - beta ** batch_num)

            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                break

            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss

            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))

            # Update the lr for the next step
            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr


        lrs = [10 ** i for i in log_lrs]
        _ = plt.figure()
        plt.xscale('log')
        plt.plot(lrs[10:-5], losses[10:-5])
        plt.savefig(join(self.output_folder, "lr_finder.png"))
        plt.close()
        return log_lrs, losses
