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
"""train"""
import os
import time
import mindspore as ms
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common import initializer as init
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.context import ParallelMode
from mindspore import communication
from mindspore.communication.management import get_group_size, get_rank
from mindspore.train.serialization import load_checkpoint, load_param_into_net, save_checkpoint
from PIL import Image
import numpy as np
from src.model import create_D, create_G
from src.dataset import create_dataset
from src.cell import GenWithLossCell, DisWithLossCell
from src.args import get_args

if __name__ == '__main__':

    t_begin = time.time()
    args_opt = get_args()

    # init context
    target = args_opt.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=target)

    if args_opt.run_distribute:
        communication.management.init()
        device_num = get_group_size()
        print("group_size(device_num) is: ", device_num)
        rank_id = get_rank()
        print("rank_id is: ", rank_id)
        # set auto parallel context
        context.set_auto_parallel_context(device_num=device_num,
                                          gradients_mean=True,
                                          parallel_mode=ParallelMode.DATA_PARALLEL)
        if args_opt.device_target == 'Ascend':
            context.set_context(device_id=args_opt.device_id)
    else:
        context.set_context(device_id=args_opt.device_id)
        device_num = 1
        rank_id = 0
    if args_opt.experiment is None:
        args_opt.experiment = 'samples'
    os.system('rm -rf {0}'.format(args_opt.experiment))
    os.system('mkdir {0}'.format(args_opt.experiment))


    ms.set_seed(0)
    dataset = create_dataset(args_opt.dataroot, args_opt.dataset, args_opt.batchSize, args_opt.imageSize, 1,
                             args_opt.workers)

    def init_weight(net):
        for _, cell in net.cells_and_names():
            if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
                cell.weight.set_data(init.initializer(init.Normal(0.02), cell.weight.shape))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(init.initializer(Tensor(np.random.normal(1, 0.02, cell.gamma.shape), \
                mstype.float32), cell.gamma.shape))
                cell.beta.set_data(init.initializer('zeros', cell.beta.shape))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(init.Normal(0.02), cell.weight.shape))

    def save_image(img, img_path, IMAGE_SIZE):
        """save image"""
        mul = ops.Mul()
        add = ops.Add()
        if isinstance(img, Tensor):
            img = mul(img, 255 * 0.5)
            img = add(img, 255 * 0.5)

            img = img.asnumpy().astype(np.uint8).transpose((0, 2, 3, 1))

        elif not isinstance(img, np.ndarray):
            raise ValueError("img should be Tensor or numpy array, but get {}".format(type(img)))

        IMAGE_ROW = 8  # Row num
        IMAGE_COLUMN = 8  # Column num
        PADDING = 2  # Interval of small pictures
        to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE + PADDING * (IMAGE_COLUMN + 1),
                                     IMAGE_ROW * IMAGE_SIZE + PADDING * (IMAGE_ROW + 1)))  # create a new picture
        # cycle
        ii = 0
        for y in range(1, IMAGE_ROW + 1):
            for x in range(1, IMAGE_COLUMN + 1):
                from_image = Image.fromarray(img[ii])
                to_image.paste(from_image, ((x - 1) * IMAGE_SIZE + PADDING * x, (y - 1) * IMAGE_SIZE + PADDING * y))
                ii = ii + 1

        to_image.save(img_path)  # save


    # define net----------------------------------------------------------------------------------------------
    # Generator
    netG = create_G(args_opt.model_type, args_opt.imageSize, args_opt.nz, args_opt.nc, args_opt.ngf)

    init_weight(netG)

    if args_opt.netG != '':  # load checkpoint if needed
        load_param_into_net(netG, load_checkpoint(args_opt.netG))

    netD = create_D(args_opt.model_type, args_opt.dataset, args_opt.imageSize, args_opt.nc, args_opt.ngf)
    init_weight(netD)

    if args_opt.netD != '':
        load_param_into_net(netD, load_checkpoint(args_opt.netD))

    fixed_noise = Tensor(np.random.normal(0, 1, size=[args_opt.batchSize, args_opt.nz, 1, 1]), dtype=ms.float32)

    # # setup optimizer
    optimizerD = nn.Adam(
        netD.trainable_params(),
        learning_rate=args_opt.lrD,
        beta1=args_opt.beta1,
        beta2=args_opt.beta2,
        # weight_decay=0.001
    )
    optimizerG = nn.Adam(
        netG.trainable_params(),
        learning_rate=args_opt.lrG,
        beta1=args_opt.beta1,
        beta2=args_opt.beta2)

    netG_train = nn.TrainOneStepCell(GenWithLossCell(netG, netD), optimizerG)
    netD_train = nn.TrainOneStepCell(DisWithLossCell(netG, netD), optimizerD)

    netG_train.set_train()
    netD_train.set_train()

    gen_iterations = 0

    t0 = time.time()
    # Train
    for epoch in range(args_opt.niter):
        epoch_start = time.time()
        data_iter = dataset.create_dict_iterator()
        length = dataset.get_dataset_size()
        i = 0
        while i < length:
            ###########################
            # (1) Update D network
            ###########################
            for p in netD.trainable_params():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            # train the discriminator Diters times
            Diters = args_opt.Diters

            j = 0
            while j < Diters and i < length:
                j += 1

                data = data_iter.__next__()
                i += 1

                # train with real and fake
                real = data['image']
                noise = Tensor(np.random.normal(0, 1, size=[args_opt.batchSize, args_opt.nz, 1, 1]), dtype=ms.float32)

                loss_D = netD_train(real, noise)

            # ##########################
            # (2) Update G network
            # ##########################
            for p in netD.trainable_params():
                p.requires_grad = False  # to avoid computation

            noise = Tensor(np.random.normal(0, 1, size=[args_opt.batchSize, args_opt.nz, 1, 1]), dtype=ms.float32)

            loss_G = netG_train(noise)
            gen_iterations += 1

            t1 = time.time()
            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f'
                  % (epoch, args_opt.niter, i, length, gen_iterations,
                     loss_D.asnumpy(), loss_G.asnumpy()))
            print('step_cost: %.4f seconds' % (float(t1 - t0)))
            t0 = t1

        fake = netG(fixed_noise)
        save_image(fake, '{0}/fake_samples_{1}.png'.format(args_opt.experiment, epoch), args_opt.imageSize)
        epoch_end = time.time()
        print('epoch_cost: %.4f seconds' % (epoch_end - epoch_start))
        if rank_id == 0 and (epoch+1) % args_opt.save_iterations == 0:
            save_checkpoint(netD, '{0}/debug_netD_epoch_{1}.ckpt'.format(args_opt.experiment, epoch+1))
            save_checkpoint(netG, '{0}/debug_netG_epoch_{1}.ckpt'.format(args_opt.experiment, epoch+1))

    t_end = time.time()
    print('total_cost: %.4f seconds' % (float(t_end - t_begin)))
    print("Train success!")
