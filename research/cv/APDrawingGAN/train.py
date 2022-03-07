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
"""train"""

import os
import time
from mindspore import context, nn, Tensor
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from src.models.APDrawingGAN import APDrawingGAN
from src.models.APDrawingGAN_G import Generator
from src.models.APDrawingGAN_D import Discriminator
from src.models.APDrawingGAN_WithLossCellG import WithLossCellG
from src.models.APDrawingGAN_WithLossCellD import WithLossCellD
from src.option.options import Options
from src.data import create_dataset

set_seed(1)

opt = Options().get_settings()
context.set_context(mode=context.GRAPH_MODE, device_target=opt.device_target)


def train():
    """train"""
    if opt.device_target == 'Ascend':
        if opt.run_distribute:
            opt.device_num = int(os.getenv('RANK_SIZE', '1'))
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, device_num=get_group_size(),
                                              gradients_mean=True)
            opt.rank = get_rank()
            opt.group_size = get_group_size()
            opt.ckpt_dir = os.path.join(opt.ckpt_dir, 'rank{}'.format(opt.rank))
            all_dataset = create_dataset(opt)
            dataset = all_dataset.batch(opt.batch_size, drop_remainder=True)
        else:
            context.set_context(device_id=int(opt.device_id))
            all_dataset = create_dataset(opt)
            dataset = all_dataset.batch(opt.batch_size)
    if opt.device_target == 'GPU':
        if opt.run_distribute:
            init("nccl")
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, device_num=get_group_size(),
                                              gradients_mean=True)
            opt.rank = get_rank()
            opt.group_size = get_group_size()
            opt.ckpt_dir = os.path.join(opt.ckpt_dir, 'rank{}'.format(opt.rank))
            all_dataset = create_dataset(opt)
            dataset = all_dataset.batch(opt.batch_size, drop_remainder=True)
        else:
            context.set_context(device_id=int(opt.device_id))
            all_dataset = create_dataset(opt)
            dataset = all_dataset.batch(opt.batch_size)

    ############################### network   ################################
    netG = Generator(opt)
    netD = Discriminator(opt)

    ############################### optimizers ###############################
    optimizer_G = nn.Adam(netG.trainable_params(), opt.lr, opt.beta1)
    optimizer_D = nn.Adam(netD.trainable_params(), opt.lr, opt.beta1)

    ##############################  with loss  ################################

    net_G_with_criterion = WithLossCellG(net_D=netD, net_G=netG, opt=opt)
    net_D_with_criterion = WithLossCellD(net_D=netD, net_G=netG, opt=opt)

    ############################## train one step cell#########################

    myTrainOneStepCellForG = nn.TrainOneStepCell(net_G_with_criterion, optimizer_G)
    myTrainOneStepCellForD = nn.TrainOneStepCell(net_D_with_criterion, optimizer_D)

    APDrawingGAN_modle = APDrawingGAN(myTrainOneStepCellForD, myTrainOneStepCellForG)
    if opt.pretrain:
        param_dict = load_checkpoint(opt.auxiliary_dir)
        load_param_into_net(APDrawingGAN_modle, param_dict)
    APDrawingGAN_modle.set_train()

    if not os.path.exists(opt.ckpt_dir):
        os.makedirs(opt.ckpt_dir)

    ############################## training ###################################
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        start_epoch_time = time.time()
        for data in dataset.create_dict_iterator():
            input_data = {}
            for d, v in data.items():
                if d in ('A_paths', 'B_paths'):
                    input_data[d] = v
                else:
                    input_data[d] = Tensor(v)

            real_B = input_data['B']
            real_B_bg = input_data['bg_B']
            real_B_eyel = input_data['eyel_B']
            real_B_eyer = input_data['eyer_B']
            real_B_nose = input_data['nose_B']
            real_B_mouth = input_data['mouth_B']
            real_B_hair = input_data['hair_B']
            # latent_code
            real_A = input_data['A']
            real_A_bg = input_data['bg_A']
            real_A_eyel = input_data['eyel_A']
            real_A_eyer = input_data['eyer_A']
            real_A_nose = input_data['nose_A']
            real_A_mouth = input_data['mouth_A']
            real_A_hair = input_data['hair_A']

            center = input_data['center']
            center = center.asnumpy()
            mask = input_data['mask']  # mask for non-eyes,nose,mouth
            mask2 = input_data['mask2']  # mask for non-bg
            dt1gt = input_data['dt1gt']
            dt2gt = input_data['dt2gt']
            netG.set_pad(center[0])
            netD.set_index(center[0])

            Gout = myTrainOneStepCellForG(real_B, real_B_bg, real_B_eyel, real_B_eyer, real_B_nose, real_B_mouth,
                                          real_B_hair, real_A, real_A_bg, real_A_eyel, real_A_eyer, real_A_nose,
                                          real_A_mouth, real_A_hair, mask, mask2, dt1gt, dt2gt).view(-1)
            Gout = Gout.mean()
            Dout = myTrainOneStepCellForD(real_B, real_B_bg, real_B_eyel, real_B_eyer, real_B_nose, real_B_mouth,
                                          real_B_hair, real_A, real_A_bg, real_A_eyel, real_A_eyer, real_A_nose,
                                          real_A_mouth, real_A_hair, mask, mask2, dt1gt, dt2gt).view(-1)
            Dout = Dout.mean()

        if opt.rank == 0:
            print('epoch', epoch, 'DLoss', Dout, '  \tGLoss', Gout)
            print('epoch', epoch, 'use time:', time.time() - start_epoch_time, 's')
            print('performance', (time.time() - start_epoch_time) * 1000 / all_dataset.get_dataset_size(), 'ms/step')
        if (epoch + 1) % opt.save_epoch_freq == 0 and opt.rank == 0:
            save_checkpoint(APDrawingGAN_modle, os.path.join(opt.ckpt_dir, f"ADPrawingGANP_modle_{epoch + 1}.ckpt"))
            save_checkpoint(netG, os.path.join(opt.ckpt_dir, f"netG_{epoch + 1}.ckpt"))
            save_checkpoint(netD, os.path.join(opt.ckpt_dir, f"netD_{epoch + 1}.ckpt"))
    if opt.isModelarts:
        from src.utils.tools import modelarts_result2obs
        modelarts_result2obs(opt)


if __name__ == '__main__':
    train()
