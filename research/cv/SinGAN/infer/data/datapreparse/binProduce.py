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
import argparse
import numpy as np
import src.functions as functions
from src.imresize import imresize


def preLauch():
    """ Args Setting """

    parser = argparse.ArgumentParser(description="SinGAN inference")
    parser.add_argument('--input_dir', type=str, default='../input/', help='input image dir')
    parser.add_argument('--input_name', type=str, default='thunder.jpg', help='input image name')
    parser.add_argument('--nc_im', type=int, default=3, help='image # channels')
    parser.add_argument('--min_size', type=int, default=25, help='image minimal size at the coarser scale')
    parser.add_argument('--scale_factor_init', type=float, default=0.75, help='pyramid scale factor')
    parser.add_argument('--max_size', type=int, default=250, help='image maximal size at the finer scale')

    opt = parser.parse_args()
    return opt
def produce():

    opt = preLauch()
    reals = []
    noise_amp = np.load('../../sdk/config/noise_amp7.npy')
    z_curr = np.load('../../sdk/config/z_curr7.npy', allow_pickle=True)
    nzx = (z_curr.shape[2])
    nzy = (z_curr.shape[3])
    z_curr = functions.generate_noise([3, nzx, nzy])
    z_curr = z_curr.reshape(1, 3, 169, 250)
    I_curr = np.load('../../sdk/config/I_curr6.npy', allow_pickle=True)

    images_cur = []
    images_cur.append(I_curr)
    real = functions.read_image(opt)

    functions.adjust_scales2image(real, opt)
    real_ = functions.read_image(opt)
    real = imresize(real_, opt.scale1, opt)
    reals = functions.creat_reals_pyramid(real, reals, opt)
    I_prev = imresize(I_curr.reshape(1, 3, 129, 191), 1/opt.scale_factor, opt)
    I_prev = I_prev[:, :, 0:round(reals[7].shape[2]), 0:round(reals[7].shape[3])]
    I_prev = I_prev[:, :, 0:z_curr.shape[2], 0:z_curr.shape[3]]
    z_in = noise_amp* (z_curr) + I_prev
    writer = open("../../mxbase/inferpath/z_in.bin", "wb")
    filedata = np.ascontiguousarray(z_in)
    writer.write(filedata)
    writer.close()

    writer = open("../../mxbase/inferpath/I_prev.bin", "wb")
    filedata2 = np.ascontiguousarray(I_prev)
    writer.write(filedata2)

if __name__ == "__main__":
    args = preLauch()
    produce()
    