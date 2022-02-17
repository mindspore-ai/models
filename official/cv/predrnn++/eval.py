# Copyright 2022 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import cv2
import numpy as np
from skimage.metrics import structural_similarity

import mindspore as ms
from mindspore import Tensor, context
from mindspore import load_checkpoint, load_param_into_net

from data_provider.mnist_to_mindrecord import create_mnist_dataset
from data_provider import preprocess
from nets.predrnn_pp import PreRNN
from config import config
import metrics

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mindrecord', type=str, default='')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--pretrained_model', type=str, default='')
    args_opt = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=args_opt.device_id)
    device_num = config.device_num
    rank = 0

    num_hidden = [int(x) for x in config.num_hidden.split(',')]
    num_layers = len(num_hidden)

    shape = [config.batch_size,
             config.seq_length,
             config.patch_size*config.patch_size*config.img_channel,
             int(config.img_width/config.patch_size),
             int(config.img_width/config.patch_size)]

    shape = list(map(int, shape))

    network = PreRNN(input_shape=shape,
                     num_layers=num_layers,
                     num_hidden=num_hidden,
                     filter_size=config.filter_size,
                     stride=config.stride,
                     seq_length=config.seq_length,
                     input_length=config.input_length,
                     tln=config.layer_norm)

    param_dict = load_checkpoint(args_opt.pretrained_model)
    load_param_into_net(network, param_dict)
    network.set_train(False)

    avg_mse = 0
    batch_id = 0
    img_mse, ssim, psnr, fmae, sharp = [], [], [], [], []

    patched_width = int(config.img_width/config.patch_size)

    mask_true = np.zeros((config.batch_size,
                          config.seq_length-config.input_length-1,
                          patched_width,
                          patched_width,
                          int(config.patch_size)**2*int(config.img_channel)))

    for i in range(config.seq_length - config.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        fmae.append(0)
        sharp.append(0)

    ds = create_mnist_dataset(dataset_files=args_opt.test_mindrecord, rank_size=device_num, \
        rank_id=rank, do_shuffle=False, batch_size=config.batch_size)

    for item in ds.create_dict_iterator(output_numpy=True):

        batch_id = batch_id + 1

        test_ims = item['input_x']
        test_ims_tp = np.transpose(test_ims, (0, 1, 3, 4, 2))
        test_ims_ori = preprocess.reshape_patch_back(test_ims_tp, config.patch_size)

        gen_images = network(Tensor(test_ims, dtype=ms.float32), Tensor(mask_true, dtype=ms.float32)).asnumpy()
        gen_images = np.transpose(gen_images, (0, 1, 3, 4, 2))
        img_gen = preprocess.reshape_patch_back(gen_images[:, 9:], config.patch_size)

        for i in range(config.seq_length - config.input_length):

            x = test_ims_ori[:, i + config.input_length, :, :, 0]
            gx = img_gen[:, i, :, :, 0]
            fmae[i] += metrics.batch_mae_frame_float(gx, x)

            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse

            real_frm = np.uint8(x * 255)
            pred_frm = np.uint8(gx * 255)
            psnr[i] += metrics.batch_psnr(pred_frm, real_frm)

            for b in range(config.batch_size):
                sharp[i] += np.max(cv2.convertScaleAbs(cv2.Laplacian(pred_frm[b], 3)))
                score, _ = structural_similarity(pred_frm[b], real_frm[b], full=True)
                ssim[i] += score

    avg_mse = avg_mse / (batch_id*config.batch_size)
    ssim = np.asarray(ssim, dtype=np.float32)/(config.batch_size*batch_id)
    psnr = np.asarray(psnr, dtype=np.float32)/batch_id
    fmae = np.asarray(fmae, dtype=np.float32)/batch_id
    sharp = np.asarray(sharp, dtype=np.float32)/(config.batch_size*batch_id)

    print('mse per frame: ' + str(avg_mse/config.input_length))
    for i in range(config.seq_length - config.input_length):
        print(img_mse[i] / (batch_id*config.batch_size))

    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(config.seq_length - config.input_length):
        print(ssim[i])

    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(config.seq_length - config.input_length):
        print(psnr[i])

    print('fmae per frame: ' + str(np.mean(fmae)))
    for i in range(config.seq_length - config.input_length):
        print(fmae[i])

    print('sharpness per frame: ' + str(np.mean(sharp)))
    for i in range(config.seq_length - config.input_length):
        print(sharp[i])
