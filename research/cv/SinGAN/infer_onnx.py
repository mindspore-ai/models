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
"""Infer SinGAN"""
import os

import numpy as np
import onnxruntime
from matplotlib import pyplot as plt
from mindspore import Tensor, context
import src.functions as functions
from src.imresize import imresize
from src.config import get_arguments


def preLauch():
    """parse the console argument"""
    parser = get_arguments()

    # Infer Device.
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--device_target', type=str, default='GPU')
    parser.add_argument('--device_id', type=int, default=1, help='device id of GPU (Default: 0)')

    # Directories.
    parser.add_argument('--input_dir', type=str, default='data', help='input image dir')
    parser.add_argument('--input_name', type=str, default='thunder.jpg', help='input image name')
    parser.add_argument('--onnx_dir', type=str, default='export_Output', help='onnx dir')
    parser.add_argument('--model_dir', type=str, default='TrainedModels', help='input model dir')
    parser.add_argument('--n_gen', type=int, default=50, help='number of images to generate at last stage')
    parser.add_argument('--infer_output', type=str, default='infer_Output', help='output dir')

    opt = parser.parse_args()
    functions.post_config(opt)

    context.set_context(save_graphs=False, device_id=opt.device_id, \
                            device_target=opt.device_target, mode=context.GRAPH_MODE)
    return opt

def main():
    """main_infer"""
    opt = preLauch()
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)
    scale_num = 0
    while scale_num < opt.stop_scale + 1:
        opt.outf = '%s/%d' % (opt.model_dir, scale_num)

        model = onnxruntime.InferenceSession('%s/%d/SinGAN.onnx' % (opt.onnx_dir, scale_num))
        Gs.append(model)

        z_curr = Tensor(np.load('%s/z_curr.npy' % (opt.outf)))
        Zs.append(z_curr)

        noise_amp = Tensor(np.load('%s/noise_amp.npy' % (opt.outf)))
        NoiseAmp.append(noise_amp)

        scale_num += 1
    real_ = functions.read_image(opt)
    real = imresize(real_, opt.scale1, opt)
    reals = functions.creat_reals_pyramid(real, reals, opt)
    SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)
    print("===========infer success================")

def SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, n=0):
    """generate produced images"""
    input_name = ['x', 'y']
    in_s = Tensor(np.zeros_like((reals[0]), dtype=np.float32))
    images_cur = []
    for G, Z_opt, noise_amp in zip(Gs, Zs, NoiseAmp):
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

            I_curr = G.run(None, {input_name[0]: z_in.asnumpy(), input_name[1]: I_prev.asnumpy()})
            I_curr = Tensor(I_curr)
            if n == len(Gs)-1:
                dir2save = '%s/%s' % (opt.infer_output, opt.input_name[:-4])
                try:
                    os.makedirs(dir2save)
                except OSError:
                    pass
                plt.imsave('%s/%d.png' % (dir2save, i), functions.convert_image_np(I_curr[0].asnumpy()), vmin=0, vmax=1)
            images_cur.append(I_curr[0])
        n += 1

    return I_curr

if __name__ == '__main__':
    main()
