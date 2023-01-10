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
#-*- coding:utf-8 -*-
import os
import glob
from math import log10
import numpy as np
import cv2
import imageio
from skimage.metrics import peak_signal_noise_ratio
imageio.plugins.freeimage.download()

def list_all_files_sorted(folder_name, extension=""):
    return sorted(glob.glob(os.path.join(folder_name, "*" + extension)))

def read_expo_times(file_name):
    return np.power(2, np.loadtxt(file_name))

def read_images(file_names):
    imgs = []
    for img_str in file_names:
        img = cv2.imread(img_str, -1)
        # equivalent to im2single from Matlab
        img = img / 2 ** 16
        img = np.float32(img)
        img.clip(0, 1)
        imgs.append(img)
    return np.array(imgs)

def read_label(file_path, file_name):
    label = imageio.imread(os.path.join(file_path, file_name), 'hdr')
    label = label[:, :, [2, 1, 0]]  ##cv2
    return label

def ldr_to_hdr(imgs, expo, gamma):
    return (imgs ** gamma) / (expo + 1e-8)

def range_compressor(x):
    return (np.log(1 + 5000 * x)) / np.log(1 + 5000)


def psnr(x, target):
    sqrdErr = np.mean((x - target) ** 2)
    return 10 * log10(1/sqrdErr)

def batch_psnr(img, imclean, data_range):
    Img = img.asnumpy().astype(np.float32)
    Iclean = imclean.asnumpy().astype(np.float32)
    psnr1 = 0
    for i in range(Img.shape[0]):
        psnr1 += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return psnr1/Img.shape[0]

def batch_psnr_mu(img, imclean, data_range):
    img = range_compressor_cuda(img)
    imclean = range_compressor_cuda(imclean)
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr1 = 0
    for i in range(Img.shape[0]):
        psnr1 += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return psnr1/Img.shape[0]

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr * (0.5 ** (epoch // args.lr_decay_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init_parameters(net):
    """Init layer parameters"""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def radiance_writer(out_path, image):

    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" %(image.shape[0], image.shape[1]))

        brightest = np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[..., 0:3] = np.around(image[..., 0:3] * scaled_mantissa[..., None])
        rgbe[..., 3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)

def save_hdr(path, image):
    return radiance_writer(path, image)
