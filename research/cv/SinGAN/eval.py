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
"""Eval SinGAN"""
import numpy as np
from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import src.functions as functions
from src.model import get_model
from src.imresize import imresize
from src.config import get_arguments
from src.manipulate import SinGAN_generate

def preLauch():
    """parse the console argument"""
    parser = get_arguments()

    # Eval Device.
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--device_target', type=str, default='Ascend')
    parser.add_argument('--device_id', type=int, default=1, help='device id of Ascend (Default: 0)')

    # Directories.
    parser.add_argument('--input_dir', type=str, default='data', help='input image dir')
    parser.add_argument('--input_name', type=str, default='thunder.jpg', help='input image name')
    parser.add_argument('--n_gen', type=int, default=50, help='number of images to generate at last stage')
    parser.add_argument('--out', type=str, default='eval_Output', help='output dir')
    opt = parser.parse_args()
    functions.post_config(opt)

    context.set_context(save_graphs=False, device_id=opt.device_id, \
                            device_target=opt.device_target, mode=context.GRAPH_MODE)
    return opt

def main():
    """main_eval"""
    opt = preLauch()
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)
    scale_num = 0
    while scale_num < opt.stop_scale + 1:
        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_, scale_num)
        G_curr, _ = get_model(scale_num, opt)
        load_param_into_net(G_curr, load_checkpoint('%s/%d/netG.ckpt' % (opt.out_, scale_num)))
        Gs.append(G_curr)

        z_curr = Tensor(np.load('%s/z_curr.npy' % (opt.outf)))
        Zs.append(z_curr)

        noise_amp = Tensor(np.load('%s/noise_amp.npy' % (opt.outf)))
        NoiseAmp.append(noise_amp)

        scale_num += 1
    real_ = functions.read_image(opt)
    real = imresize(real_, opt.scale1, opt)
    reals = functions.creat_reals_pyramid(real, reals, opt)
    SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)
    print("===========eval success================")

if __name__ == '__main__':
    main()
