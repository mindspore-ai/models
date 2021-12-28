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
"""Functions for SinGAN"""
import math
import random
import numpy as np
from src.imresize import imresize, np2mind, denorm
from skimage import io as img
from mindspore import set_seed, save_checkpoint, Tensor

def generate_noise(size):
    """generate_noise"""
    noise = np.random.randn(1, size[0], size[1], size[2]).astype(np.float32)
    return noise

def convert_image_np(inp):
    """convert_image_np"""
    inp = denorm(inp)
    inp = inp[-1, :, :, :]
    inp = inp.transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    return inp

def read_image(opt):
    """read_image"""
    x = img.imread('%s/%s' % (opt.input_dir, opt.input_name))
    x = np2mind(x, opt)
    return x

def reset_grads(model, require_grad):
    """reset_grads"""
    for p in model.parameters_dict().values():
        p.requires_grad = require_grad
    return model


def save_networks(netG, netD, z_opt, opt):
    """save_networks"""
    save_checkpoint(netG, '%s/netG.ckpt' % (opt.outf))
    save_checkpoint(netD, '%s/netD.ckpt' % (opt.outf))
    np.save('%s/z_curr.npy' % (opt.outf), z_opt.asnumpy().astype(np.float32))
    np.save('%s/noise_amp.npy' % (opt.outf), Tensor(opt.noise_amp).asnumpy().astype(np.float32))

def adjust_scales2image(real_, opt):
    """adjust_scales2image"""
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size \
        / (min(real_.shape[2], real_.shape[3])), 1), opt.scale_factor_init))) + 1
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) \
        / max([real_.shape[2], real_.shape[3]]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]), 1)
    real = imresize(real_, opt.scale1, opt)
    opt.scale_factor = math.pow(opt.min_size / (min(real.shape[2], real.shape[3])), 1 / (opt.stop_scale))
    return real

def creat_reals_pyramid(real, reals, opt):
    """creat_reals_pyramid"""
    for i in range(0, opt.stop_scale+1, 1):
        scale = math.pow(opt.scale_factor, opt.stop_scale - i)
        curr_real = imresize(real, scale, opt)
        reals.append(curr_real)
    return reals

def generate_dir2save(opt):
    """generate_dir2save"""
    dir2save = None
    if opt.mode == 'train':
        dir2save = 'TrainedModels/%s/scale_factor=%f,alpha=%d' % (opt.input_name[:-4], opt.scale_factor_init, opt.alpha)
    else:
        dir2save = 'TrainedModels/%s/scale_factor=%f,alpha=%d' % (opt.input_name[:-4], opt.scale_factor_init, opt.alpha)
    return dir2save


def post_config(opt):
    """post_config"""
    # init fixed parameters
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    opt.out_ = 'TrainedModels/%s/scale_factor=%f/' % (opt.input_name[:-4], opt.scale_factor)

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(0, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    set_seed(opt.manualSeed)
    return opt
