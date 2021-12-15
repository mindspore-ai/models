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
# ============================================================================s
"""Generate Fake Images for SinGAN"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mindspore import Tensor
import src.functions as functions
from src.imresize import imresize

def SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, n=0):
    """generate produced images"""
    in_s = Tensor(np.zeros_like((reals[0]), dtype=np.float32))
    images_cur = []
    for G, Z_opt, noise_amp in zip(Gs, Zs, NoiseAmp):
        G.set_train(False)
        nzx = (Z_opt.shape[2])
        nzy = (Z_opt.shape[3])
        images_prev = images_cur
        images_cur = []

        for i in range(opt.n_gen):
            if n == 0:
                z_curr = functions.generate_noise([1, nzx, nzy])
                z_curr = Tensor(np.broadcast_to(z_curr, (1, 3, z_curr.shape[2], z_curr.shape[3])))
            else:
                z_curr = functions.generate_noise([opt.nc_z, nzx, nzy])
                z_curr = Tensor(z_curr)

            if images_prev == []:
                I_prev = in_s
            else:
                I_prev = images_prev[i]
                I_prev = Tensor(imresize(I_prev.asnumpy(), 1/opt.scale_factor, opt))
                I_prev = I_prev[:, :, 0:round(reals[n].shape[2]), 0:round(reals[n].shape[3])]
                I_prev = I_prev[:, :, 0:z_curr.shape[2], 0:z_curr.shape[3]]

            z_in = noise_amp * (z_curr) + I_prev
            I_curr = G(z_in, I_prev)

            if n == len(Gs)-1:
                dir2save = '%s/RandomSamples/%s' % (opt.out, opt.input_name[:-4])
                try:
                    os.makedirs(dir2save)
                except OSError:
                    pass
                plt.imsave('%s/%d.png' % (dir2save, i), functions.convert_image_np(I_curr.asnumpy()), vmin=0, vmax=1)
            images_cur.append(I_curr)
        n += 1

    return I_curr
