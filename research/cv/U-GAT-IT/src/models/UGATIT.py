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
"""
pipline for U-GAT-IT
"""
import time
import math
import os
from glob import glob

import cv2
import numpy as np
import mindspore.ops as ops
from mindspore import nn
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.common import initializer as init
from mindspore.communication.management import get_rank

from .networks import ResnetGenerator, Discriminator, GWithLossCell, DWithLossCell
from .cell import TrainOneStepG, TrainOneStepD, Generator
from ..utils.tools import denorm, tensor2numpy, RGB2BGR, cam
from ..dataset.dataset import TrainDataLoader, TestDataLoader
from ..metrics.metrics import mean_kernel_inception_distance


class UGATIT:
    """pipline"""
    def __init__(self, args):
        self.light = args.light
        self.distributed = args.distributed
        self.mode = args.phase

        if self.light:
            self.model_name = 'UGATIT_light'
        else:
            self.model_name = 'UGATIT'

        self.modelart = args.enable_modelarts

        self.train_url = args.train_url
        self.output_path = args.output_path
        self.dataset = args.dataset
        self.data_path = args.data_path

        self.decay_flag = args.decay_flag
        self.epoch = args.epoch
        self.decay_epoch = args.decay_epoch

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.lr_policy = 'linear'

        self.loss_scale = args.loss_scale
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch
        self.use_global_norm = args.use_global_norm

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight
        self.weights = [self.adv_weight, self.cycle_weight, self.identity_weight, self.cam_weight]
        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.resume = args.resume

        """utils"""
        self.oneslike = ops.OnesLike()
        self.zeroslike = ops.ZerosLike()
        self.assign = ops.Assign()

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# epochs: ", self.epoch)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """build model"""
        self.train_nums = 1
        if self.mode == 'train':
            train_loader, test_loader, train_nums = TrainDataLoader(self.img_size,
                                                                    self.data_path,
                                                                    self.dataset,
                                                                    self.batch_size,
                                                                    self.distributed)
            self.train_loader = train_loader
            self.test_iterator = test_loader.create_dict_iterator()

            self.train_nums = train_nums
            print("Training dataset size = ", self.train_nums)
        elif self.mode == 'test':
            test_loader = TestDataLoader(self.img_size,
                                         self.data_path,
                                         self.dataset)
            self.test_iterator = test_loader.create_dict_iterator()
        else:
            raise RuntimeError("Invalid mode")
        print("Dataset load finished")

        self.genA2B = ResnetGenerator(input_nc=3,
                                      output_nc=3,
                                      ngf=self.ch,
                                      n_blocks=self.n_res,
                                      img_size=self.img_size,
                                      light=self.light)
        self.genB2A = ResnetGenerator(input_nc=3,
                                      output_nc=3,
                                      ngf=self.ch,
                                      n_blocks=self.n_res,
                                      img_size=self.img_size,
                                      light=self.light)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
        self.generator = Generator(self.genA2B, self.genB2A)

        self.init_weights(self.genA2B, 'KaimingUniform', math.sqrt(5))
        self.init_weights(self.genB2A, 'KaimingUniform', math.sqrt(5))
        self.init_weights(self.disGA, 'KaimingUniform', math.sqrt(5))
        self.init_weights(self.disGB, 'KaimingUniform', math.sqrt(5))
        self.init_weights(self.disLA, 'KaimingUniform', math.sqrt(5))
        self.init_weights(self.disLB, 'KaimingUniform', math.sqrt(5))
        self.start_epoch = 1
        if self.resume:
            model_list = glob(os.path.join(self.output_path, self.dataset, 'model', '*.ckpt'))
            if model_list:
                model_list.sort()
                self.start_epoch = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.output_path, self.dataset, 'model'), self.start_epoch)
                print(" [*]Epoch %d Load SUCCESS" % self.start_epoch)
        start_step = (self.start_epoch - 1) * self.train_nums
        self.learning_rate = self.get_lr()[start_step:]
        loss_scale = self.loss_scale

        self.D_loss_net = DWithLossCell(self.disGA,
                                        self.disLA,
                                        self.disGB,
                                        self.disLB,
                                        self.weights)
        self.G_loss_net = GWithLossCell(self.generator,
                                        self.disGA,
                                        self.disLA,
                                        self.disGB,
                                        self.disLB,
                                        self.weights)

        self.G_optim = nn.Adam(self.generator.trainable_params(),
                               learning_rate=self.learning_rate,
                               beta1=0.5,
                               beta2=0.999,
                               weight_decay=self.weight_decay)

        self.D_optim = nn.Adam(self.D_loss_net.trainable_params(),
                               learning_rate=self.learning_rate,
                               beta1=0.5,
                               beta2=0.999,
                               weight_decay=self.weight_decay)

        self.D_train_net = TrainOneStepD(self.D_loss_net, self.D_optim, loss_scale, self.use_global_norm)
        self.G_train_net = TrainOneStepG(self.G_loss_net, self.generator, self.G_optim,
                                         loss_scale, self.use_global_norm)

    def get_lr(self):
        """
        Learning rate generator.
        """
        if self.lr_policy == 'linear':
            lrs = [self.lr] * self.train_nums * self.decay_epoch
            for epoch in range(self.decay_epoch, self.epoch):
                lr_epoch = self.lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch)
                lrs += [lr_epoch] * self.train_nums
            return lrs
        return self.lr

    def init_weights(self, net, init_type='normal', init_gain=0.02):
        """init weights"""
        for _, cell in net.cells_and_names():
            if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose, nn.Dense)):
                if init_type == 'normal':
                    cell.weight.set_data(init.initializer(init.Normal(init_gain), cell.weight.shape))
                elif init_type == 'xavier':
                    cell.weight.set_data(init.initializer(init.XavierUniform(init_gain), cell.weight.shape))
                elif init_type == 'KaimingUniform':
                    cell.weight.set_data(init.initializer(init.HeUniform(init_gain), cell.weight.shape))
                elif init_type == 'constant':
                    cell.weight.set_data(init.initializer(0.0005, cell.weight.shape))
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            elif isinstance(cell, (nn.GroupNorm, nn.BatchNorm2d)):
                cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
                cell.beta.set_data(init.initializer('zeros', cell.beta.shape))

    def train(self):
        """train"""
        self.D_train_net.set_train()
        self.G_train_net.set_train()
        data_loader = self.train_loader.create_dict_iterator()
        # training loop
        print('training start !')
        for epoch in range(self.start_epoch, self.epoch + 1):
            i = 0
            for data in data_loader:
                i += 1
                start_time = time.time()
                real_A = data["image_A"]
                real_B = data["image_B"]

                # Update
                fake_A2B, fake_B2A, Generator_loss = self.G_train_net(real_A, real_B)
                Discriminator_loss = self.D_train_net(real_A, real_B, fake_A2B, fake_B2A)

                # clip parameter of AdaILN and ILN, applied after optimizer step
                for m in self.genA2B.cells_and_names():
                    if hasattr(m[1], 'rho'):
                        w = m[1].rho.data
                        w = ops.clip_by_value(w, 0, 1)
                        m[1].rho.data.set_data(w)
                for m in self.genB2A.cells_and_names():
                    if hasattr(m[1], 'rho'):
                        w = m[1].rho.data
                        w = ops.clip_by_value(w, 0, 1)
                        m[1].rho.data.set_data(w)

                print("epoch %d:[%5d/%5d] time per iter: %4.4f " % (epoch,
                                                                    i,
                                                                    self.train_nums,
                                                                    time.time() - start_time))
                print("d_loss:", Discriminator_loss)
                print("g_loss:", Generator_loss)

            if epoch % self.print_freq == 0:
                if self.distributed:
                    if get_rank() == 0:
                        self.print(epoch)
                        save_checkpoint(self.genA2B,
                                        os.path.join(self.output_path, self.dataset + '_genA2B_params_latest.ckpt'))
                        save_checkpoint(self.genB2A,
                                        os.path.join(self.output_path, self.dataset + '_genB2A_params_latest.ckpt'))
                        save_checkpoint(self.disGA,
                                        os.path.join(self.output_path, self.dataset + '_disGA_params_latest.ckpt'))
                        save_checkpoint(self.disGB,
                                        os.path.join(self.output_path, self.dataset + '_disGB_params_latest.ckpt'))
                        save_checkpoint(self.disLA,
                                        os.path.join(self.output_path, self.dataset + '_disLA_params_latest.ckpt'))
                        save_checkpoint(self.disLB,
                                        os.path.join(self.output_path, self.dataset + '_disLB_params_latest.ckpt'))

                else:
                    self.print(epoch)
                    save_checkpoint(self.genA2B,
                                    os.path.join(self.output_path, self.dataset + '_genA2B_params_latest.ckpt'))
                    save_checkpoint(self.genB2A,
                                    os.path.join(self.output_path, self.dataset + '_genB2A_params_latest.ckpt'))
                    save_checkpoint(self.disGA,
                                    os.path.join(self.output_path, self.dataset + '_disGA_params_latest.ckpt'))
                    save_checkpoint(self.disGB,
                                    os.path.join(self.output_path, self.dataset + '_disGB_params_latest.ckpt'))
                    save_checkpoint(self.disLA,
                                    os.path.join(self.output_path, self.dataset + '_disLA_params_latest.ckpt'))
                    save_checkpoint(self.disLB,
                                    os.path.join(self.output_path, self.dataset + '_disLB_params_latest.ckpt'))

            if epoch % self.save_freq == 0:
                if self.distributed:
                    if get_rank() == 0:
                        self.save(os.path.join(self.output_path, self.dataset, 'model'), epoch)
                else:
                    self.save(os.path.join(self.output_path, self.dataset, 'model'), epoch)

    def print(self, epoch):
        """save middle results"""
        test_sample_num = 5
        A2B = np.zeros((self.img_size * 7, 0, 3))
        B2A = np.zeros((self.img_size * 7, 0, 3))

        for _ in range(test_sample_num):
            data = next(self.test_iterator)

            real_A = data["image_A"]
            real_B = data["image_B"]

            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
            fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

            # Without copying real_A and real_B tensors before feeding them
            # into genB2A and genA2B does not work correctly with the GPU backend.
            fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A.copy())
            fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B.copy())

            A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                       cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                       RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                       cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                       RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                       cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                       RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

            B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                       cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                       RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                       cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                       RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                       cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                       RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

        cv2.imwrite(os.path.join(self.output_path, self.dataset, 'img', 'A2B_%07d.png' % epoch), A2B * 255.0)
        cv2.imwrite(os.path.join(self.output_path, self.dataset, 'img', 'B2A_%07d.png' % epoch), B2A * 255.0)

    def save(self, savedir, epoch):
        save_checkpoint(self.genA2B, os.path.join(savedir, self.dataset + '_genA2B_params_%07d.ckpt' % epoch))
        save_checkpoint(self.genB2A, os.path.join(savedir, self.dataset + '_genB2A_params_%07d.ckpt' % epoch))
        save_checkpoint(self.disGA, os.path.join(savedir, self.dataset + '_disGA_params_%07d.ckpt' % epoch))
        save_checkpoint(self.disGB, os.path.join(savedir, self.dataset + '_disGB_params_%07d.ckpt' % epoch))
        save_checkpoint(self.disLA, os.path.join(savedir, self.dataset + '_disLA_params_%07d.ckpt' % epoch))
        save_checkpoint(self.disLB, os.path.join(savedir, self.dataset + '_disLB_params_%07d.ckpt' % epoch))

    def load(self, loaddir, epoch):
        """load checkpoint"""
        genA2B_params = load_checkpoint(os.path.join(loaddir, self.dataset + '_genA2B_params_%07d.ckpt' % epoch))
        not_load = {}
        not_load['genA2B'] = load_param_into_net(self.genA2B, genA2B_params)
        if self.mode == 'train':
            genB2A_params = load_checkpoint(os.path.join(loaddir, self.dataset + '_genB2A_params_%07d.ckpt' % epoch))
            disGA_params = load_checkpoint(os.path.join(loaddir, self.dataset + '_disGA_params_%07d.ckpt' % epoch))
            disGB_params = load_checkpoint(os.path.join(loaddir, self.dataset + '_disGB_params_%07d.ckpt' % epoch))
            disLA_params = load_checkpoint(os.path.join(loaddir, self.dataset + '_disLA_params_%07d.ckpt' % epoch))
            disLB_params = load_checkpoint(os.path.join(loaddir, self.dataset + '_disLB_params_%07d.ckpt' % epoch))

            not_load['genB2A'] = load_param_into_net(self.genB2A, genB2A_params)
            not_load['disGA'] = load_param_into_net(self.disGA, disGA_params)
            not_load['disGB'] = load_param_into_net(self.disGB, disGB_params)
            not_load['disLA'] = load_param_into_net(self.disLA, disLA_params)
            not_load['disLB'] = load_param_into_net(self.disLB, disLB_params)
        print("these params are not loaded: ", not_load)

    def test(self, inception_ckpt_path=None):
        """test"""
        self.genA2B.set_train(True)
        output_path = os.path.join(self.output_path, self.dataset)
        model_list = glob(os.path.join(output_path, 'model', '*.ckpt'))
        if model_list:
            model_list.sort()
            start_epoch = int(model_list[-1].split('_')[-1].split('.')[0])
            self.load(os.path.join(output_path, 'model'), start_epoch)
            print(" [*] epoch %d Load SUCCESS" % start_epoch)
        else:
            print(" [*] Load FAILURE")
            return
        for n, data in enumerate(self.test_iterator):
            real_A = data['image_A']

            fake_A2B, _, _ = self.genA2B(real_A)
            A = RGB2BGR(tensor2numpy(denorm(real_A[0])))
            A2B = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))

            cv2.imwrite(os.path.join(output_path, 'test', 'A_%d.png' % (n + 1)), A * 255.0)
            cv2.imwrite(os.path.join(output_path, 'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)

        if inception_ckpt_path is not None:
            dataset_path = os.path.join(self.data_path, self.dataset)
            mean_kernel_inception_distance(output_path, dataset_path, inception_ckpt_path)
