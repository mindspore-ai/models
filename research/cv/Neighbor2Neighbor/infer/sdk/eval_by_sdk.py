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
'''post process for 310 inference'''
import os
import argparse
import numpy as np
import PIL.Image as Image

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

parser = argparse.ArgumentParser(description='post process for 310 inference')
parser.add_argument("--result_path", type=str, required=True, help="result file path")
parser.add_argument("--noisetype", type=str, default="gauss25", help="noise type")
parser.add_argument("--img_data", type=str, required=True, help="result file path")
parser.add_argument("--pre_data", type=str, required=True, help="pre file path")
parser.add_argument("--dataset", type=str, default='Kodak', choices=['Kodak', 'BSD300'], help="result file path")
args = parser.parse_args()

class AugmentNoise():
    '''AugmentNoise'''
    def __init__(self, style):
        if style.startswith('gauss'):
            self.params = [
                float(p) / 255.0 for p in style.replace('gauss', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):
            self.params = [
                float(p) for p in style.replace('poisson', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

    def add_noise(self, x):
        '''add_noise'''
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        if self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        if self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        assert self.style == "poisson_range"
        min_lam, max_lam = self.params
        lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
        return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)

if args.dataset == 'Kodak':
    image_size = 768
elif args.dataset == 'BSD300':
    image_size = 512
def cal_acc(result_path):
    ''' calculate fina accuracy '''
    noise_generator = AugmentNoise(args.noisetype)
    files = os.listdir(result_path)
    name = []
    psnr = []   #after denoise
    ssim = []   #after denoise
    psnr_b = [] #before denoise
    ssim_b = [] #before denoise

    for file in files:
        data_name = file.split('_')[0] + ".png"
        test_data_name = file.split('_')[0] + ".bin"
        img_name = args.img_data + data_name
        prebin = args.pre_data + test_data_name
        full_rst_path = os.path.join(result_path, file)
        img_clean = np.array(Image.open(img_name), dtype='float32') / 255.0
        img = noise_generator.add_noise(img_clean)
        H = img.shape[0]
        W = img.shape[1]

        img_test = np.fromfile(prebin, dtype=np.float32).reshape((1, 3, image_size, image_size))
        img_test_ssim = np.transpose(np.squeeze(img_test), (1, 2, 0))

        img_clean_psnr = np.expand_dims(np.transpose(img_clean, (2, 0, 1)), 0)#NCHW

        prediction = np.fromfile(full_rst_path, dtype=np.float32)
        prediction = prediction[0:1769472]
        prediction = prediction.reshape((1, 3, image_size, image_size))
        y_predict = prediction[:, :, :H, :W]

        img_out = np.clip(y_predict, 0, 1)
        img_out_ssim = np.transpose(np.squeeze(img_out), (1, 2, 0))
        psnr_noise, psnr_denoised = compare_psnr(img_clean_psnr, img_test[:, :, :H, :W]), \
                                    compare_psnr(img_clean_psnr, img_out)

        ssim_noise, ssim_denoised = compare_ssim(img_clean, img_test_ssim[ :H, :W, :], multichannel=True, win_size=11,
                                                 data_range=1, gaussian_weights=True, sigma=1.5), \
                                    compare_ssim(img_clean, img_out_ssim, multichannel=True, win_size=11, data_range=1,
                                                 gaussian_weights=True, sigma=1.5)

        psnr.append(psnr_denoised)
        ssim.append(ssim_denoised)
        psnr_b.append(psnr_noise)
        ssim_b.append(ssim_noise)
    psnr_avg = sum(psnr)/len(psnr)
    ssim_avg = sum(ssim)/len(ssim)
    psnr_avg_b = sum(psnr_b)/len(psnr_b)
    ssim_avg_b = sum(ssim_b)/len(ssim_b)
    name.append('Average')
    psnr.append(psnr_avg)
    ssim.append(ssim_avg)
    psnr_b.append(psnr_avg_b)
    ssim_b.append(ssim_avg_b)

    print("Result in:%s", str(args.img_data))
    print('Before denoise: Average PSNR_b = {0:.4f}, SSIM_b = {1:.4f};'\
                    .format(psnr_avg_b, ssim_avg_b))
    print('After denoise: Average PSNR = {0:.4f}, SSIM = {1:.4f}'\
                    .format(psnr_avg, ssim_avg))
    print("testing finished....")


if __name__ == "__main__":
    cal_acc(args.result_path)
