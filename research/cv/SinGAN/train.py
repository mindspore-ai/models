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
"""Train SinGAN"""
import os
import ast
import time
import numpy as np
import matplotlib.pyplot as plt
from mindspore import load_checkpoint, load_param_into_net, nn, Tensor, context
from mindspore.ops import Sqrt
import src.functions as functions
from src.model import get_model
from src.imresize import imresize
from src.config import get_arguments
from src.manipulate import SinGAN_generate
from src.loss import GenLoss, DisLoss
from src.cell import TrainOneStepCellGen, TrainOneStepCellDis
def preLauch():
    """parse the console argument"""
    parser = get_arguments()

    # Train Device.
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument("--distribute", type=ast.literal_eval, default=False, help="Run distribute, default is false.")
    parser.add_argument('--device_target', type=str, default='Ascend')
    parser.add_argument('--device_num', type=int, default=1, help='device num, default is 1.')
    parser.add_argument('--device_id', type=int, default=0, help='device id of Ascend (Default: 0)')

    # Directories.
    parser.add_argument('--input_dir', type=str, default='data', help='input image dir')
    parser.add_argument('--input_name', type=str, default='thunder.jpg', help='input image name')
    parser.add_argument('--n_gen', type=int, default=50, help='number of images to generate at last stage')
    parser.add_argument('--out', type=str, default='Train_Output', help='output folder')
    opt = parser.parse_args()
    functions.post_config(opt)

    context.set_context(save_graphs=False, device_id=opt.device_id, \
                            device_target=opt.device_target, mode=context.GRAPH_MODE)
    return opt


def train(opt, Gs, Zs, reals, NoiseAmp):
    """training"""
    real_ = functions.read_image(opt)
    in_s = 0
    scale_num = 0
    real = imresize(real_, opt.scale1, opt)
    reals = functions.creat_reals_pyramid(real, reals, opt)
    while scale_num < opt.stop_scale + 1:
        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_, scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
            pass

        plt.imsave('%s/real_scale.png' % (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)
        G_curr, D_curr = get_model(scale_num, opt)
        if scale_num % 4 != 0:
            load_param_into_net(G_curr, load_checkpoint('%s/%d/netG.ckpt' % (opt.out_, scale_num-1)))
            load_param_into_net(D_curr, load_checkpoint('%s/%d/netD.ckpt' % (opt.out_, scale_num-1)))

        z_curr, in_s, G_curr = train_single_scale(D_curr, G_curr, reals, Gs, Zs, in_s, NoiseAmp, opt)

        G_curr = functions.reset_grads(G_curr, False)
        G_curr.set_train(False)
        D_curr = functions.reset_grads(D_curr, False)
        D_curr.set_train(False)

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        scale_num += 1
        del D_curr, G_curr

def train_single_scale(netD, netG, reals, Gs, Zs, in_s, NoiseAmp, opt):
    """training for single scale"""
    real = Tensor(reals[len(Gs)])
    opt.nzx = real.shape[2]
    opt.nzy = real.shape[3]

    fixed_noise = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy])
    z_opt = Tensor(np.zeros_like((fixed_noise), dtype=np.float32))

    # Define network with loss
    D_loss_cell = DisLoss(opt, netG, netD)
    G_loss_cell = GenLoss(opt, netG, netD)

    # Define optimizer
    optimizerD = nn.Adam(netD.trainable_params(), learning_rate=opt.lr_d, beta1=opt.beta1, beta2=0.999)
    optimizerG = nn.Adam(netG.trainable_params(), learning_rate=opt.lr_g, beta1=opt.beta1, beta2=0.999)

    # Define One step train
    D_trainOneStep = TrainOneStepCellDis(D_loss_cell, optimizerD, clip=10)
    G_trainOneStep = TrainOneStepCellGen(G_loss_cell, optimizerG, clip=10)

    # Train one step
    D_trainOneStep.set_train(True)
    G_trainOneStep.set_train(True)

    for epoch in range(opt.niter):
        start = time.time()
        if Gs == []:
            z_opt = functions.generate_noise([1, opt.nzx, opt.nzy])
            z_opt = Tensor(np.broadcast_to(z_opt, (1, opt.nc_z, opt.nzx, opt.nzy)))
            noise_ = functions.generate_noise([1, opt.nzx, opt.nzy])
            noise_ = Tensor(np.broadcast_to(noise_, (1, opt.nc_z, opt.nzx, opt.nzy)))
        else:
            noise_ = Tensor(functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy]))
        # Update D network
        functions.reset_grads(netD, True)
        functions.reset_grads(netG, False)
        netG.set_train(False)
        netD.set_train(True)
        for j in range(opt.Dsteps):
            if (j == 0) & (epoch == 0):
                if Gs == []:
                    prev = Tensor(np.zeros((1, opt.nc_z, opt.nzx, opt.nzy), dtype=np.float32))
                    in_s = prev
                    z_prev = Tensor(np.zeros((1, opt.nc_z, opt.nzx, opt.nzy), dtype=np.float32))
                    opt.noise_amp = 1
                else:
                    prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', opt)
                    z_prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rec', opt)
                    criterion = nn.MSELoss()
                    RMSE = Sqrt()(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init * RMSE
            else:
                prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', opt)

            if Gs == []:
                noise = noise_
            else:
                noise = opt.noise_amp * noise_ + prev
            d_loss, _, _, _ = D_trainOneStep(real, noise, prev)

        # Update G network
        functions.reset_grads(netD, False)
        functions.reset_grads(netG, True)
        netG.set_train(True)
        netD.set_train(False)
        for j in range(opt.Gsteps):
            Z_opt = opt.noise_amp * z_opt + z_prev
            g_loss, _, _, x_fake, _ = G_trainOneStep(real, Z_opt, z_prev, noise, prev)

        if epoch % 500 == 0 or epoch == (opt.niter-1):
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))
            plt.imsave('%s/fake_sample.png' %  (opt.outf), \
                functions.convert_image_np(x_fake.asnumpy()), vmin=0, vmax=1)
            plt.imsave('%s/G(z_opt).png'    % (opt.outf), \
                functions.convert_image_np(netG(Z_opt, z_prev).asnumpy()), vmin=0, vmax=1)
    end = time.time()
    pref = (end - start) * 1000 / opt.niter / max(opt.Gsteps, opt.Dsteps)
    print("scale_num {}, epoch {}, {:.3f} ms per step, d_loss is {:.4f}, g_loss is {:.4f}".format(len(Gs), \
                                epoch, pref, d_loss.asnumpy(), g_loss.asnumpy()))
    functions.save_networks(netG, netD, z_opt, opt)
    return z_opt, in_s, netG

def draw_concat(Gs, Zs, reals, NoiseAmp, in_s, mode, opt):
    """get image at previous scale"""
    G_z = in_s
    if Gs:
        if mode == 'rand':
            count = 0
            for G, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Zs, reals, reals[1:], NoiseAmp):
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2], Z_opt.shape[3]])
                    z = Tensor(np.broadcast_to(z, (1, 3, z.shape[2], z.shape[3])))
                else:
                    z = Tensor(functions.generate_noise([opt.nc_z, Z_opt.shape[2], Z_opt.shape[3]]))
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                z_in = noise_amp * z + G_z
                G_z = G(z_in, G_z)
                G_z = Tensor(imresize(G_z.asnumpy(), 1/opt.scale_factor, opt))
                G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Zs, reals, reals[1:], NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                z_in = noise_amp * Z_opt + G_z
                G_z = G(z_in, G_z)
                G_z = Tensor(imresize(G_z.asnumpy(), 1/opt.scale_factor, opt))
                G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                count += 1
    return G_z

def main():
    """main_train"""
    opt = preLauch()
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    try:
        os.makedirs(dir2save)
    except OSError:
        pass
    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)
    start_train = time.time()
    train(opt, Gs, Zs, reals, NoiseAmp)
    end_train = time.time()
    pref_train = (end_train - start_train) / 60
    print("=============training success after {:.1f} mins=========".format(pref_train))
    SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)
    print("============generate fake images success================")

if __name__ == '__main__':
    main()
