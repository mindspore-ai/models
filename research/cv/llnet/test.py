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
"""test LLNet"""
import os
import time
from math import ceil
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import cv2
import numpy as np

from src.llnet import LLNet
from src.model_utils.config import config

def add_gaussian_noise_darkened(image, gamma_from=2, gamma_to=5, var=0.1):
    gamma = np.random.uniform(gamma_from, gamma_to)
    image_darkened = image ** gamma

    h, w = image.shape[:2]
    sigma = (np.random.uniform() * var**2)**0.5
    gauss_noise = np.random.normal(0, sigma, (h, w))

    return np.clip(image_darkened  + gauss_noise, 0, 1)

def calc_score(backbone, img_full_path, gamma_from=1.0, gamma_to=1.0, var=0.0):
    im = cv2.imread(img_full_path, cv2.IMREAD_GRAYSCALE)

    head_tail = os.path.split(img_full_path)
    base_ext = os.path.splitext(head_tail[1])

    file_base_name = base_ext[0]

    if config.enable_modelarts:
        result_path = '/home/work/user-job-dir/outputs/'
    else:
        result_path = './result_Files/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    cv2.imwrite(result_path+file_base_name + '.png', im)
    im = im / 255.
    height, width = im.shape[:2]

    if gamma_from < 1.001:
        im_lightend = add_gaussian_noise_darkened(im, 0.3, 0.3, var)
        im_tensor = Tensor(im_lightend).reshape((-1, 1, height, width)).astype(mindspore.float32)

        im_lightend = np.clip(im_lightend * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(result_path+file_base_name+'_l.png', im_lightend)
    else:
        im_tensor = Tensor(im).reshape((-1, 1, height, width)).astype(mindspore.float32)

    new_image = np.pad(im, ((0, 20), (0, 20)))

    im_reconstruct = np.zeros_like(new_image)
    im_noise = np.zeros_like(new_image)
    im1 = np.zeros_like(new_image)

    shape = (ceil(width/3) * ceil(height/3), 289)
    noise_patch_batch = np.ndarray(shape, dtype=np.float32)
    i = 0
    for x in range(0, width, 3):
        for y in range(0, height, 3):
            origin_patch = new_image[y:y + 17, x:x + 17]
            noise_patch = add_gaussian_noise_darkened(origin_patch, gamma_from, gamma_to, var)
            im_noise[y:y+17, x:x+17] += noise_patch
            noise_patch = noise_patch.reshape(-1)
            noise_patch_batch[i] = noise_patch
            i += 1
            im1[y:y + 17, x:x + 17] += 1

    rec = backbone(Tensor(noise_patch_batch)).asnumpy()

    i = 0
    for x in range(0, width, 3):
        for y in range(0, height, 3):
            im_reconstruct[y:y + 17, x:x + 17] += rec[i].reshape(17, 17)
            i += 1

    im_noise = im_noise[0:height, 0:width]
    im_reconstruct = im_reconstruct[0:height, 0:width]
    im1 = im1[0:height, 0:width]

    im1[im1 == 0] = 1
    im_reconstruct /= im1
    im_noise /= im1

    im_reconstruct_tensor = Tensor(im_reconstruct[0:height, 0:width]).reshape((1, 1, height, width))
    im_reconstruct_tensor = im_reconstruct_tensor.astype(mindspore.float32)

    ssim = nn.SSIM()
    psnr = nn.PSNR()

    psnr_score = psnr(im_tensor, im_reconstruct_tensor)
    ssim_score = ssim(im_tensor, im_reconstruct_tensor)

    psnr_score = psnr_score.asnumpy().item()
    ssim_score = ssim_score.asnumpy().item()

    im_reconstruct = np.clip(im_reconstruct * 255, 0, 255).astype(np.uint8)
    im_noise = np.clip(im_noise * 255, 0, 255).astype(np.uint8)

    if gamma_from < 1.0001:
        cv2.imwrite(result_path+file_base_name+'_l_reconstruct.png', im_reconstruct)
    else:
        if var <= 0.0001:
            cv2.imwrite(result_path+file_base_name+'_d.png', im_noise)
            cv2.imwrite(result_path+file_base_name+'_d_reconstruct.png', im_reconstruct)
        else:
            if var <= 0.07001:
                cv2.imwrite(result_path+file_base_name+'_d_GN18.png', im_noise)
                cv2.imwrite(result_path+file_base_name+'_d_GN18_reconstruct.png', im_reconstruct)
            else:
                cv2.imwrite(result_path+file_base_name+'_d_GN25.png', im_noise)
                cv2.imwrite(result_path+file_base_name+'_d_GN25_reconstruct.png', im_reconstruct)

    return psnr_score, ssim_score

def test(backbone):
    file_list = []

    if config.enable_modelarts:
        file_path = os.path.join(config.data_path, 'dataset/test_images/')
    else:
        file_path = os.path.join(config.dataset_path, 'test_images/')
    for root, _, files in os.walk(file_path):
        for file_name in files:
            file_list.append(os.path.join(root, file_name))

    file_list.sort()
    file_count = 0

    average_psnr_1 = 0.0
    average_psnr_2 = 0.0
    average_psnr_3 = 0.0
    average_psnr_4 = 0.0

    average_ssim_1 = 0.0
    average_ssim_2 = 0.0
    average_ssim_3 = 0.0
    average_ssim_4 = 0.0

    print('                                           PSNR(dB) /  SSIM')

    for img_full_path in file_list:
        head_tail = os.path.split(img_full_path)
        base_ext = os.path.splitext(head_tail[1])

        psnr_score_01, ssim_score_01 = calc_score(backbone=backbone, img_full_path=img_full_path,
                                                  gamma_from=1.0, gamma_to=1.0, var=0.0)
        print("%-40s   %5.2f       %.2f" %(base_ext[0], psnr_score_01, ssim_score_01))

        psnr_score_02, ssim_score_02 = calc_score(backbone=backbone, img_full_path=img_full_path,
                                                  gamma_from=3.0, gamma_to=3.0, var=0.0)
        print("%-40s   %5.2f       %.2f" %(base_ext[0] + '-D', psnr_score_02, ssim_score_02))

        psnr_score_03, ssim_score_03 = calc_score(backbone=backbone, img_full_path=img_full_path,
                                                  gamma_from=3.0, gamma_to=3.0, var=0.07)
        print("%-40s   %5.2f       %.2f" %(base_ext[0] + '-D+GN18', psnr_score_03, ssim_score_03))

        psnr_score_04, ssim_score_04 = calc_score(backbone=backbone, img_full_path=img_full_path,
                                                  gamma_from=3.0, gamma_to=3.0, var=0.1)
        print("%-40s   %5.2f       %.2f" %(base_ext[0] + '-D+GN25', psnr_score_04, ssim_score_04))

        file_count = file_count + 1

        average_psnr_1 = average_psnr_1 + psnr_score_01
        average_psnr_2 = average_psnr_2 + psnr_score_02
        average_psnr_3 = average_psnr_3 + psnr_score_03
        average_psnr_4 = average_psnr_4 + psnr_score_04

        average_ssim_1 = average_ssim_1 + ssim_score_01
        average_ssim_2 = average_ssim_2 + ssim_score_02
        average_ssim_3 = average_ssim_3 + ssim_score_03
        average_ssim_4 = average_ssim_4 + ssim_score_04

    average_psnr_1 = average_psnr_1 / file_count
    average_psnr_2 = average_psnr_2 / file_count
    average_psnr_3 = average_psnr_3 / file_count
    average_psnr_4 = average_psnr_4 / file_count

    average_ssim_1 = average_ssim_1 / file_count
    average_ssim_2 = average_ssim_2 / file_count
    average_ssim_3 = average_ssim_3 / file_count
    average_ssim_4 = average_ssim_4 / file_count

    average_psnr_all = (average_psnr_1 + average_psnr_2 + average_psnr_3 + average_psnr_4) / 4.0
    average_ssim_all = (average_ssim_1 + average_ssim_2 + average_ssim_3 + average_ssim_4) / 4.0

    average_psnr_d_all = (average_psnr_2 + average_psnr_3 + average_psnr_4) / 3.0
    average_ssim_d_all = (average_ssim_2 + average_ssim_3 + average_ssim_4) / 3.0

    print("Average ALL                                %5.2f       %.2f" %(average_psnr_all, average_ssim_all))
    print("Average -D*                                %5.2f       %.2f" %(average_psnr_d_all, average_ssim_d_all))
    print("Average -D+GN25                            %5.2f       %.2f" %(average_psnr_4, average_ssim_4))

if __name__ == '__main__':
    start_time = time.time()

    device_id = config.device_id
    print('device_id = ', device_id)

    if config.use_pynative_mode:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target,
                            device_id=device_id)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target,
                            device_id=device_id, save_graphs=False)

    if config.device_target == 'Ascend':
        context.set_context(enable_reduce_precision=config.enable_reduce_precision)

    dataset_path = config.dataset_path
    checkpoint = config.checkpoint
    checkpoint_dir = config.checkpoint_dir

    if config.enable_modelarts:
        import moxing
        # download dataset from obs to server
        print("=========================================================")
        print("config.data_url  =", config.data_url)
        print("config.data_path =", config.data_path)

        moxing.file.copy_parallel(src_url=config.data_url, dst_url=config.data_path)
        print(os.listdir(config.data_path))
        print("=========================================================")

        # download the checkpoint from obs to server
        if config.ckpt_url != '':
            print("=========================================================")
            print("config.ckpt_url  =", config.ckpt_url)

            if not config.enable_checkpoint_dir:
                base_name = os.path.basename(config.ckpt_url)
                dst_url = os.path.join(config.load_path, base_name)
                moxing.file.copy_parallel(src_url=config.ckpt_url, dst_url=dst_url)
                checkpoint = dst_url

                print("checkpoint =", checkpoint)
            else:
                pos = config.ckpt_url.find("/model/")
                if  pos > 0:
                    cloud_checkpoint_dir = config.ckpt_url[0:pos+6]
                    print("cloud_checkpoint_dir  =", cloud_checkpoint_dir)
                    dst_url = config.load_path
                    moxing.file.copy_parallel(src_url=cloud_checkpoint_dir, dst_url=dst_url)
                    checkpoint_dir = dst_url
                    print("checkpoint_dir =", checkpoint_dir)

            print(os.listdir(config.load_path))
            print("=========================================================")

    net6 = LLNet()

    if not config.enable_checkpoint_dir:
        print("")
        print(checkpoint)

        ckpt = load_checkpoint(checkpoint)
        load_param_into_net(net6, ckpt)
        net6.set_train(False)
        net6.set_float16()

        test(net6)
        print('time: ', ceil(time.time() - start_time), ' seconds')
    else:
        ckpt_file_list = []
        for ckpt_path, _, ckpt_files in os.walk(checkpoint_dir):
            for file in ckpt_files:
                if os.path.splitext(file)[1] == '.ckpt':
                    ckpt_file_list.append(os.path.join(ckpt_path, file))
        ckpt_file_count = 0
        ckpt_file_list.sort()

        for checkpoint in ckpt_file_list:
            if not os.path.exists(checkpoint):
                continue
            ckpt_file_count += 1
            print("")
            print(checkpoint)

            ckpt = load_checkpoint(checkpoint)
            load_param_into_net(net6, ckpt)
            net6.set_train(False)
            net6.set_float16()

            test(net6)

        print('*********************************************************************')
        print(ckpt_file_count, ' checkpoints have been tested')
        print('time: ', ceil(time.time() - start_time), ' seconds')
        print('*********************************************************************')
    if config.enable_modelarts:
        import moxing
        print("=========================================================")
        print("/home/work/user-job-dir/")
        print(os.listdir("/home/work/user-job-dir/"))
        if os.path.exists('/home/work/user-job-dir/outputs'):
            print("/home/work/user-job-dir/outputs")
            print(os.listdir("/home/work/user-job-dir/outputs"))
            print("config.result_url =", config.result_url)
            moxing.file.copy_parallel(src_url='/home/work/user-job-dir/outputs', dst_url=config.result_url)
