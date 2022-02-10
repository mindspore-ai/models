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

"""Train"""

import os
from collections import OrderedDict

from mindspore import set_seed
from mindspore import context
from mindspore import load_checkpoint, save_checkpoint, DatasetHelper
from mindspore.communication import init
from mindspore.context import ParallelMode
from mindspore.nn import Sigmoid
from mindspore.nn.optim import Adam

from model_utils.config import base_config
from src.dataset import create_dataset, save_img
from src.egnet import build_model, init_weights
from src.sal_edge_loss import SalEdgeLoss, WithLossCell
from src.train_forward_backward import TrainClear, TrainOptimize, TrainForwardBackward
if base_config.train_online:
    import moxing as mox
    mox.file.shift('os', 'mox')

def main(config):
    if config.train_online:
        local_data_url = os.path.join("/cache", config.train_path)
        mox.file.copy_parallel(config.online_train_path, local_data_url)
        config.train_path = local_data_url
        if not config.online_pretrained_path == "":
            pretrained_path = os.path.join("/cache", config.pretrained_url)
            mox.file.copy_parallel(config.online_pretrained_path, pretrained_path)
            if config.pre_trained == "":
                if config.base_model == "vgg":
                    config.vgg = os.path.join("/cache", config.vgg)
                    mox.file.copy_parallel(os.path.join(config.online_pretrained_path,
                                                        os.path.basename(config.vgg)), config.vgg)
                elif config.base_model == "resnet":
                    config.resnet = os.path.join("/cache", config.resnet)
                    mox.file.copy_parallel(os.path.join(config.online_pretrained_path,
                                                        os.path.basename(config.resnet)), config.resnet)
            else:
                config.pre_trained = os.path.join("/cache", config.pre_trained)
                mox.file.copy_parallel(os.path.join(config.online_pretrained_path,
                                                    os.path.basename(config.pre_trained)), config.pre_trained)
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target,
                        reserve_class_name_in_scope=False,
                        device_id=os.getenv('DEVICE_ID', '0'))
    if config.is_distributed:
        config.epoch = config.epoch * 6
        set_seed(1234)
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        init()
    train_dataset, _ = create_dataset(config.batch_size, num_thread=config.num_thread, train_path=config.train_path,
                                      is_distributed=config.is_distributed)
    run = config.train_save_name
    if not os.path.exists(config.save_fold):
        os.mkdir(config.save_fold)
    if not os.path.exists("%s/run-%s" % (config.save_fold, run)):
        os.mkdir("%s/run-%s" % (config.save_fold, run))
        os.mkdir("%s/run-%s/logs" % (config.save_fold, run))
        os.mkdir("%s/run-%s/models" % (config.save_fold, run))
    config.save_fold = "%s/run-%s" % (config.save_fold, run)
    train = Solver(train_dataset, config)
    train.train()


class Solver:
    def __init__(self, train_ds, config):
        self.train_ds = train_ds
        self.config = config
        self.network = build_model(config.base_model)
        init_weights(self.network)
        # Load pretrained model
        if self.config.pre_trained == "":
            if config.base_model == "vgg":
                if os.path.exists(self.config.vgg):
                    self.network.base.load_pretrained_model(self.config.vgg)
                    print("Load VGG pretrained model")
            elif config.base_model == "resnet":
                if os.path.exists(self.config.resnet):
                    self.network.base.load_pretrained_model(self.config.resnet)
                    print("Load ResNet pretrained model")
            else:
                raise ValueError("unknown base model")
        else:
            load_checkpoint(self.config.pre_trained, self.network)
            print("Load EGNet pretrained model")
        self.log_output = open("%s/logs/log.txt" % config.save_fold, "w")

        """some hyper params"""
        p = OrderedDict()
        if self.config.base_model == "vgg":  # Learning rate resnet:5e-5, vgg:2e-5(begin with 2e-8, warm up to 2e-5 in epoch 3)
            p["lr_bone"] = 2e-8
            if self.config.is_distributed:
                p["lr_bone"] = 2e-9
        elif self.config.base_model == "resnet":
            p["lr_bone"] = 5e-5
            if self.config.is_distributed:
                p["lr_bone"] = 5e-9
        else:
            raise ValueError("unknown base model")
        p["lr_branch"] = 0.025  # Learning rate
        p["wd"] = 0.0005  # Weight decay
        p["momentum"] = 0.90  # Momentum
        self.p = p
        self.lr_decay_epoch = [15, 24]
        if config.is_distributed:
            self.lr_decay_epoch = [15*6, 24*6]
        self.tmp_path = "tmp_see"

        self.lr_bone = p["lr_bone"]
        self.lr_branch = p["lr_branch"]
        self.optimizer = Adam(self.network.trainable_params(), learning_rate=self.lr_bone,
                              weight_decay=p["wd"], loss_scale=self.config.loss_scale)
        self.print_network()
        self.loss_fn = SalEdgeLoss(config.n_ave_grad, config.batch_size)
        params = self.optimizer.parameters
        self.grad_sum = params.clone(prefix="grad_sum", init="zeros")
        self.zeros = params.clone(prefix="zeros", init="zeros")
        self.train_forward_backward = self.build_train_forward_backward_network()
        self.train_optimize = self.build_train_optimize()
        self.train_clear = self.build_train_clear()
        self.sigmoid = Sigmoid()

    def build_train_forward_backward_network(self):
        """Build forward and backward network"""
        network = self.network
        network = WithLossCell(network, self.loss_fn)
        self.config.loss_scale = 1.0
        network = TrainForwardBackward(network, self.optimizer, self.grad_sum, self.config.loss_scale).set_train()
        return network

    def build_train_optimize(self):
        """Build optimizer network"""
        network = TrainOptimize(self.optimizer, self.grad_sum).set_train()
        return network

    def build_train_clear(self):
        """Build clear network"""
        network = TrainClear(self.grad_sum, self.zeros).set_train()
        return network

    def print_network(self):
        """
        print network architecture
        """
        name = "EGNet-" + self.config.base_model
        model = self.network
        num_params = 0
        i = 0
        for param in model.get_parameters():
            i += 1
            num_params += param.size
        print(name)
        print(model)
        print(f"The number of layers: {i}")
        print(f"The number of parameters: {num_params}")

    def train(self):
        """training phase"""
        ave_grad = 0
        iter_num = self.train_ds.get_dataset_size()
        dataset_helper = DatasetHelper(self.train_ds, dataset_sink_mode=False, epoch_num=self.config.epoch)
        if not os.path.exists(self.tmp_path):
            os.mkdir(self.tmp_path)
        for epoch in range(self.config.epoch):
            r_edge_loss, r_sal_loss, r_sum_loss = 0, 0, 0
            for i, data_batch in enumerate(dataset_helper):
                sal_image, sal_label, sal_edge = data_batch[0], data_batch[1], data_batch[2]
                if sal_image.shape[2:] != sal_label.shape[2:]:
                    print("Skip this batch")
                    continue
                self.train_forward_backward(sal_image, sal_label, sal_edge)
                r_edge_loss += self.loss_fn.edge_loss.asnumpy()
                r_sal_loss += self.loss_fn.sal_loss.asnumpy()
                r_sum_loss += self.loss_fn.total_loss.asnumpy()

                if (ave_grad + 1) % self.config.n_ave_grad == 0:
                    self.train_optimize()
                    self.train_clear()
                    ave_grad = 0
                else:
                    ave_grad += 1
                if (i + 1) % self.config.show_every == 0:
                    num_step = self.config.n_ave_grad * self.config.batch_size
                    log_str = "epoch: [%2d/%2d], iter: [%5d/%5d] || Edge : %10.4f || Sal : %10.4f || Sum : %10.4f" \
                              % (epoch + 1, self.config.epoch, i + 1, iter_num,
                                 r_edge_loss * num_step / self.config.show_every,
                                 r_sal_loss * num_step / self.config.show_every,
                                 r_sum_loss * num_step / self.config.show_every)
                    print(log_str)
                    print(f"Learning rate: {self.lr_bone}")
                    self.log_output.write(log_str + "\n")
                    self.log_output.write(f"Learning rate: {self.lr_bone}\n")
                    r_edge_loss, r_sal_loss, r_sum_loss = 0, 0, 0

                if (i + 1) % self.config.save_tmp == 0:
                    _, _, up_sal_final = self.network(sal_image)
                    sal = self.sigmoid((up_sal_final[-1])).asnumpy().squeeze()
                    sal_image = sal_image.asnumpy().squeeze().transpose((1, 2, 0))
                    sal_label = sal_label.asnumpy().squeeze()
                    save_img(sal, os.path.join(self.tmp_path, f"iter{i}-sal-0.jpg"))
                    save_img(sal_image, os.path.join(self.tmp_path, f"iter{i}-sal-data.jpg"))
                    save_img(sal_label, os.path.join(self.tmp_path, f"iter{i}-sal-target.jpg"))

            if (epoch + 1) % self.config.epoch_save == 0:
                if self.config.train_online:
                    save_checkpoint(self.network, "epoch_%d_%s_bone.ckpt" %
                                    (epoch + 1, self.config.base_model))
                    mox.file.copy_parallel("epoch_%d_%s_bone.ckpt" %
                                           (epoch + 1, self.config.base_model),
                                           os.path.join(self.config.train_url, "epoch_%d_%s_bone.ckpt" %
                                                        (epoch + 1, self.config.base_model)))
                else:
                    save_checkpoint(self.network, "%s/models/epoch_%d_%s_bone.ckpt" %
                                    (self.config.save_fold, epoch + 1, self.config.base_model))
            if self.config.base_model == "vgg" or self.config.is_distributed:
                if self.config.is_distributed:
                    lr_rise_epoch = [3, 6, 9, 12]
                else:
                    lr_rise_epoch = [1, 2, 3]
                if epoch in lr_rise_epoch:
                    self.lr_bone = self.lr_bone * 10
                    self.optimizer = Adam(filter(lambda p: p.requires_grad, self.network.get_parameters()),
                                          learning_rate=self.lr_bone, weight_decay=self.p["wd"])
                    self.train_optimize = self.build_train_optimize()
            if epoch in self.lr_decay_epoch:
                self.lr_bone = self.lr_bone * 0.1
                self.optimizer = Adam(filter(lambda p: p.requires_grad, self.network.get_parameters()),
                                      learning_rate=self.lr_bone, weight_decay=self.p["wd"])
                self.train_optimize = self.build_train_optimize()
        if self.config.train_online:
            save_checkpoint(self.network, "final_%s_bone.ckpt"% (self.config.base_model))
            mox.file.copy_parallel("final_%s_bone.ckpt"% (self.config.base_model),
                                   os.path.join(self.config.train_url, "final_%s_bone.ckpt"% (self.config.base_model)))
        else:
            save_checkpoint(self.network,
                            "%s/models/final_%s_bone.ckpt" % (self.config.save_fold, self.config.base_model))


if __name__ == "__main__":
    main(base_config)
    