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
""" train """

import sys
import random
import math
import time
import os.path
import numpy as np
import mindspore as ms
from mindspore import context
from src.options.train_options import TrainOptions
from src.models.netG import SPADEGenerator
from src.data import DatasetInit, ade20k_dataset
from src.util.lr_schedule import dynamic_lr
from src.util.adam import Adam
from src.models.netD import MultiscaleDiscriminator
from src.models.loss import GANLoss, VGGLoss
from src.models.cells import GenTrainOneStepCell, DisTrainOneStepCell, GenWithLossCell, DisWithLossCell

def main():
    opt = TrainOptions().parse()
    print(' '.join(sys.argv))
    random.seed(1)
    np.random.seed(1)
    ms.dataset.config.set_seed(1)
    ms.set_seed(1)
    if opt.distribute:
        ms.communication.management.init("nccl")
        device_id = ms.communication.management.get_rank()
        is_save = device_id
        device_num = ms.communication.management.get_group_size()
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ms.context.ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        context.set_context(
            mode=context.GRAPH_MODE,
            device_target="GPU", device_id=opt.id)
        is_save = 0
    # dataset init
    instance = ade20k_dataset.ADE20KDataset()
    instance.initialize(opt)
    dataset_init = DatasetInit(opt)
    if opt.distribute:
        dataset = dataset_init.create_dataset_distribute(instance, device_id, device_num)
    else:
        dataset = dataset_init.create_dataset_not_distribute(instance)
    batch_dataset_size = dataset.get_dataset_size()
    dataset_iterator = dataset.create_dict_iterator(output_numpy=True, num_epochs=opt.total_epoch - opt.now_epoch)
    # network init
    netG = SPADEGenerator(opt)
    netD = MultiscaleDiscriminator(opt)
    # load param into net
    if opt.now_epoch != 0:
        from mindspore import load_checkpoint, load_param_into_net
        epoch = math.floor(opt.now_epoch / 10) * 10
        g_checkpoints_path = './{0}/netG_epoch_{1}.ckpt'.format(opt.checkpoints_dir, epoch)
        d_checkpoints_path = './{0}/netD_epoch_{1}.ckpt'.format(opt.checkpoints_dir, epoch)

        g_param_dict = load_checkpoint(g_checkpoints_path)
        load_param_into_net(netG, g_param_dict)
        d_param_dict = load_checkpoint(d_checkpoints_path)
        load_param_into_net(netD, d_param_dict)
        print("continue train by ", epoch, "epoch")

    G_lr = ms.Tensor(dynamic_lr(opt, batch_dataset_size, opt.G_lr), ms.float32)
    D_lr = ms.Tensor(dynamic_lr(opt, batch_dataset_size, opt.D_lr), ms.float32)
    netG_params = list(filter(lambda x: 'param_free_norm' not in x.name, netG.trainable_params()))
    netD_params = list(filter(lambda x: 'bn' not in x.name, netD.trainable_params()))
    optimizer_G = Adam(netG_params, learning_rate=G_lr, beta1=opt.beta1, beta2=opt.beta2)
    optimizer_D = Adam(netD_params, learning_rate=D_lr, beta1=opt.beta1, beta2=opt.beta2)
    GLoss = GANLoss(opt.gan_mode, opt=opt)
    FLoss = ms.nn.L1Loss()
    VggLoss = VGGLoss(opt)
    VggLoss.set_train(False)
    netG_with_criterion = GenWithLossCell(opt, netG, netD, GLoss, FLoss, VggLoss)
    netD_with_criterion = DisWithLossCell(netG, netD, GLoss)
    netG_train = GenTrainOneStepCell(netG_with_criterion, optimizer_G)
    netD_train = DisTrainOneStepCell(netD_with_criterion, optimizer_D)
    netG_train.set_train()
    netD_train.set_train()
    netG_train.G.VggLoss.set_train(False)
    for epoch in range(opt.total_epoch - opt.now_epoch):
        for i, data in enumerate(dataset_iterator):
            input_semantics = ms.Tensor(data['input_semantics'])
            real_image = ms.Tensor(data['image'])
            start_time = time.time()
            loss_G = netG_train(input_semantics, real_image)
            loss_D = netD_train(input_semantics, real_image)
            end_time = time.time()
            print('[%d/%d][%d/%d]: Loss_D: %f Loss_G: %f  step_time: %f s'
                  % (epoch +1+ opt.now_epoch, opt.total_epoch, i + 1, batch_dataset_size,
                     loss_D.asnumpy(), loss_G.asnumpy(), end_time - start_time))
        if (epoch + 1 + opt.now_epoch) % 10 == 0 and epoch + opt.now_epoch < opt.total_epoch and is_save == 0:
            netG_ckpt_name = './{0}/netG_epoch_{1}.ckpt'.format(opt.checkpoints_dir, epoch+opt.now_epoch+1)
            netD_ckpt_name = './{0}/netD_epoch_{1}.ckpt'.format(opt.checkpoints_dir, epoch+opt.now_epoch+1)
            ms.train.serialization.save_checkpoint(netG, netG_ckpt_name)
            ms.train.serialization.save_checkpoint(netD, netD_ckpt_name)

    netG_ckpt_name = './{0}/netG_epoch_{1}.ckpt'.format(opt.checkpoints_dir, opt.total_epoch)
    netD_ckpt_name = './{0}/netD_epoch_{1}.ckpt'.format(opt.checkpoints_dir, opt.total_epoch)
    if not os.path.isfile(netG_ckpt_name):
        ms.train.serialization.save_checkpoint(netG, netG_ckpt_name)
    if not os.path.isfile(netD_ckpt_name):
        ms.train.serialization.save_checkpoint(netD, netD_ckpt_name)

if __name__ == '__main__':
    main()
