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
'''eval'''
import datetime
import os
import time
import glob
import pandas as pd
import numpy as np
import PIL.Image as Image

import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore.common import set_seed
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore import load_checkpoint, load_param_into_net

from src.logger import get_logger
from src.models import UNet
from src.dataset import AugmentNoise
from src.config import config as cfg


def copy_data_from_obs():
    '''copy_data_from_obs'''
    if cfg.use_modelarts:
        import moxing as mox
        cfg.logger.info("copying test weights from obs to cache....")
        mox.file.copy_parallel(cfg.pretrain_path, 'cache/weight')
        cfg.logger.info("copying test weights finished....")
        cfg.pretrain_path = 'cache/weight/'

        cfg.logger.info("copying test dataset from obs to cache....")
        mox.file.copy_parallel(cfg.test_dir, 'cache/test')
        cfg.logger.info("copying test dataset finished....")
        cfg.test_dir = 'cache/test/'

def copy_data_to_obs():
    if cfg.use_modelarts:
        import moxing as mox
        cfg.logger.info("copying files from cache to obs....")
        mox.file.copy_parallel(cfg.save_dir, cfg.outer_path)
        cfg.logger.info("copying finished....")

def test(model_path):
    '''test'''
    model = UNet(in_nc=cfg.n_channel, out_nc=cfg.n_channel, n_feature=cfg.n_feature)
    cfg.logger.info("load test weights from %s", str(model_path))
    load_param_into_net(model, load_checkpoint(model_path))
    cfg.logger.info("loaded test weights from %s", str(model_path))
    noise_generator = AugmentNoise(cfg.noisetype)

    model.set_train(False)

    cast = P.Cast()
    transpose = P.Transpose()
    expand_dims = P.ExpandDims()
    compare_psnr = nn.PSNR()
    compare_ssim = nn.SSIM()

    for filename in os.listdir(cfg.test_dir):
        tem_path = os.path.join(cfg.test_dir, filename)
        out_dir = os.path.join(cfg.save_dir, filename)
        if not cfg.use_modelarts and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        name = []
        psnr = []   #after denoise
        ssim = []   #after denoise
        psnr_b = [] #before denoise
        ssim_b = [] #before denoise
        file_list = glob.glob(os.path.join(tem_path, '*'))

        cfg.logger.info('Start to test on %s', str(tem_path))
        start_time = time.time()
        for file in file_list:
            suffix = file.split('.')[-1]
            # read image
            img_clean = np.array(Image.open(file), dtype='float32') / 255.0

            img_test = noise_generator.add_noise(img_clean)
            H = img_test.shape[0]
            W = img_test.shape[1]
            val_size = (max(H, W) + 31) // 32 * 32
            img_test = np.pad(img_test,
                              [[0, val_size - H], [0, val_size - W], [0, 0]],
                              'reflect')

            img_clean = Tensor(img_clean, mindspore.float32) #HWC
            img_test = Tensor(img_test, mindspore.float32)   #HWC

            # predict
            img_clean = expand_dims(transpose(img_clean, (2, 0, 1)), 0)#NCHW
            img_test = expand_dims(transpose(img_test, (2, 0, 1)), 0)#NCHW

            prediction = model(img_test)
            y_predict = prediction[:, :, :H, :W]

            # calculate numeric metrics
            img_out = C.clip_by_value(y_predict, 0, 1)

            psnr_noise, psnr_denoised = compare_psnr(img_clean, img_test[:, :, :H, :W]), \
                                        compare_psnr(img_clean, img_out)
            ssim_noise, ssim_denoised = compare_ssim(img_clean, img_test[:, :, :H, :W]), \
                                        compare_ssim(img_clean, img_out)

            psnr.append(psnr_denoised.asnumpy()[0])
            ssim.append(ssim_denoised.asnumpy()[0])
            psnr_b.append(psnr_noise.asnumpy()[0])
            ssim_b.append(ssim_noise.asnumpy()[0])

            # save images
            filename = file.split('/')[-1].split('.')[0]    # get the name of image file
            name.append(filename)
            if not cfg.use_modelarts and cfg.save_denoised_images:
                # inner the operation 'Image.save', it will first check the file \
                # existence of same name, which is not allowed on modelarts
                img_test = cast(img_test*255, mindspore.uint8).asnumpy()
                img_test = img_test.squeeze(0).transpose((1, 2, 0)) #turn into HWC to save as an image
                img_test = Image.fromarray(img_test)
                img_test.save(os.path.join(out_dir, filename+'_noisetype'+'{}_psnr{:.2f}.'\
                                .format(cfg.noisetype, psnr_noise.asnumpy()[0])+str(suffix)))
                img_out = cast(img_out*255, mindspore.uint8).asnumpy()
                img_out = img_out.squeeze(0).transpose((1, 2, 0)) #turn into HWC to save as an image
                img_out = Image.fromarray(img_out)
                img_out.save(os.path.join(out_dir, filename+'_psnr{:.2f}.'.format(\
                             psnr_denoised.asnumpy()[0])+str(suffix)))

        psnr_avg = sum(psnr)/len(psnr)
        ssim_avg = sum(ssim)/len(ssim)
        psnr_avg_b = sum(psnr_b)/len(psnr_b)
        ssim_avg_b = sum(ssim_b)/len(ssim_b)
        name.append('Average')
        psnr.append(psnr_avg)
        ssim.append(ssim_avg)
        psnr_b.append(psnr_avg_b)
        ssim_b.append(ssim_avg_b)
        cfg.logger.info("Result in:%s", str(tem_path))
        cfg.logger.info('Before denoise: Average PSNR_b = {0:.4f}, SSIM_b = {1:.4f};'\
                        .format(psnr_avg_b, ssim_avg_b))
        cfg.logger.info('After denoise: Average PSNR = {0:.4f}, SSIM = {1:.4f}'\
                        .format(psnr_avg, ssim_avg))
        cfg.logger.info("testing finished....")
        time_used = time.time() - start_time
        cfg.logger.info("time cost:%s seconds!", str(time_used))
        if not cfg.use_modelarts:
            pd.DataFrame({'name': np.array(name), 'psnr_b': np.array(psnr_b), \
                          'psnr': np.array(psnr), 'ssim_b': np.array(ssim_b), \
                          'ssim': np.array(ssim)}).to_csv(out_dir+'/metrics.csv', index=True)

if __name__ == '__main__':
    set_seed(1)
    cfg.save_dir = os.path.join(cfg.output_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    if not cfg.use_modelarts and not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)
    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=cfg.device_target, device_id=device_id, save_graphs=False)
    cfg.logger = get_logger(cfg.save_dir, "Neighbor2Neighbor", 0)
    cfg.logger.save_args(cfg)
    copy_data_from_obs()
    test(os.path.join(cfg.pretrain_path, cfg.ckpt_name))
    copy_data_to_obs()
