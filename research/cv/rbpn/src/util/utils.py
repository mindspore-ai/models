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


"""utils """

import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common import initializer as init
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio



def gram_matrix(img):
    """gram_matrix
    Args:
        img(numpy): one image
    Outputs:
        numpy
    """
    a, b, c, d = img.shape
    reshape = ops.Reshape()
    features = reshape(img, (a * b, c * d))

    matmul = ops.MatMul()
    transpose = ops.Transpose()
    div = ops.Div()
    perm = (1, 0)
    G = matmul(features, transpose(features, perm))
    return div(G, (a * b * c * d))


def save_img(img, img_name, save_dir):
    """ save image"""
    if isinstance(img, Tensor):
        img = img.asnumpy()
    save_fn = os.path.join(save_dir, img_name + ".png")
    img = img.squeeze().clip(0, 1).transpose(1, 2, 0)
    cv2.imwrite(save_fn, cv2.cvtColor(img * 255, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])


def init_weights(net, init_type='normal', init_gain=0.1):
    """
       Initialize network weights.
       Parameters:
           net (Cell): Network to be initialized
           init_type (str): The name of an initialization method: normal | xavier.
           init_gain (float): Gain factor for normal and xavier.
       """
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
            if init_type == 'normal':
                cell.weight.set_data(
                    init.initializer(init.Normal(init_gain), cell.weight.shape))
            elif init_type == 'xavier':
                cell.weight.set_data(
                    init.initializer(init.XavierUniform(init_gain), cell.weight.shape))
            elif init_type == 'uniform':
                cell.weight.set_data(
                    init.initializer(init.Uniform(init_gain), cell.weight.shape))
            elif init_type == 'constant':
                cell.weight.set_data(init.initializer(0.001, cell.weight.shape))
            elif init_type == 'KaimingUniform':
                cell.weight.set_data(init.initializer(init.HeUniform(init_gain), cell.weight.shape))
            elif init_type == 'truncatednormal':
                cell.weight.set_data(init.initializer(init.TruncatedNormal(init_gain), cell.weight.shape))
            elif init_type == 'KaimingNormal':
                cell.weight.set_data(init.initializer(init.HeNormal(init_gain), cell.weight.shape))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if cell.bias is not None:
                zeros = ops.Zeros()
                cell.bias.set_data(zeros(cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.BatchNorm2d):
            cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
            cell.beta.set_data(init.initializer('zeros', cell.beta.shape))


def _getYchannel(img):
    """return Y channel data"""
    img = img.transpose([1, 2, 0])
    img = rgb2ycbcr(img)
    img = img[:, :, :1]
    return img


def compute_psnr(hr, sr):
    """return psnr between hr and sr"""
    if isinstance(hr, Tensor):
        hr = hr.asnumpy()
    if isinstance(sr, Tensor):
        sr = sr.asnumpy()
    if hr.shape[0] == 3:
        hr = _getYchannel(hr)
    if sr.shape[0] == 3:
        sr = _getYchannel(sr)
    psnr = peak_signal_noise_ratio(hr / 255.0, sr / 255.0, data_range=1.0)
    return psnr


def save_losses(G_losses, D_losses, name):
    """save loss of the cell product
    Args:
        G_losses(list): it stores the netG's loss
        D_losses(list): it stores the netD's loss
    """
    plt.figure(figsize=(10, 5))
    if G_losses is not None:
        plt.plot(G_losses, 'ro-', label="G")
    if D_losses is not None:
        plt.plot(D_losses, 'bo--', label="D")

    plt.xlabel("epoch")
    plt.ylabel("Losses")
    plt.legend()
    if D_losses is None:
        plt.title("Generator Loss During Training")
        plt.savefig(name)
    else:
        plt.title("Generator and Discriminator Loss During Training")
        plt.savefig(name)


def save_psnr(psnr_list, savepath, model_type):
    """save_psnr
    Args:
        psnr_list(list): it stores the netG's loss
        savepath(list): it stores the path
        model_type(str): use the model type
    """
    titlename = "{} psnr_loss in evaldata".format(model_type)
    plt.figure(figsize=(10, 5))
    plt.title(titlename)
    plt.grid(axis='y')
    plt.plot(psnr_list, 'r*-', label="avg_psnr")
    plt.xlabel("epoch")
    plt.ylabel("compute_psnr")
    plt.legend()
    plt.savefig(savepath)



def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[1:3]
    pred = pred[:, 1 + shave_border:height - shave_border, 1 + shave_border:width - shave_border]
    gt = gt[:, 1 + shave_border:height - shave_border, 1 + shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)
