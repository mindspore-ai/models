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
import onnxruntime

import cv2
import numpy as np

from skimage.metrics import structural_similarity
from mindspore import context
from data_provider.mnist_to_mindrecord import create_mnist_dataset
from data_provider import preprocess
from config import config
import metrics

def creat_session(onnx_path, target_device):
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {target_device}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()
    return session, input_name

def eval_net():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mindrecord', type=str, default='')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--pretrained_model', type=str, default='')
    args_opt = parser.parse_args()
    device_num = config.device_num
    rank = 0
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=args_opt.device_id)
    ds = create_mnist_dataset(dataset_files=args_opt.test_mindrecord, rank_size=device_num, \
            rank_id=rank, do_shuffle=False, batch_size=config.batch_size)
    session, input_name = creat_session(args_opt.pretrained_model, config.device_target)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim, psnr, fmae, sharp = [], [], [], [], []
    patched_width = int(config.img_width/config.patch_size)
    mask_true = np.zeros((config.batch_size,
                          config.seq_length-config.input_length-1,
                          patched_width,
                          patched_width,
                          int(config.patch_size)**2*int(config.img_channel)), dtype=np.float32)
    for i in range(config.seq_length - config.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        fmae.append(0)
        sharp.append(0)
    for item in ds.create_dict_iterator(output_numpy=True):
        batch_id = batch_id + 1
        test_ims = item['input_x']
        test_ims_tp = np.transpose(test_ims, (0, 1, 3, 4, 2))
        test_ims_ori = preprocess.reshape_patch_back(test_ims_tp, config.patch_size)
        test_ims = {input_name[0].name: item['input_x'], input_name[1].name: mask_true}
        gen_images = session.run(None, test_ims)
        gen_images = np.transpose(list(gen_images[0]), (0, 1, 3, 4, 2))
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
    avg_mse = avg_mse / (batch_id * config.batch_size)
    ssim = np.asarray(ssim, dtype=np.float32) / (config.batch_size * batch_id)
    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    fmae = np.asarray(fmae, dtype=np.float32) / batch_id
    sharp = np.asarray(sharp, dtype=np.float32) / (config.batch_size * batch_id)
    print('mse per frame: ' + str(avg_mse / config.input_length))
    for i in range(config.seq_length - config.input_length):
        print(img_mse[i] / (batch_id * config.batch_size))
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
if __name__ == '__main__':
    eval_net()
