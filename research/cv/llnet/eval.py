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
"""evaluate LLNet"""
import os
import time
from math import ceil
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.dataset import open_mindrecord_dataset
from src.llnet import LLNet
from src.model_utils.config import config

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
        # download dataset from obs to server
        import moxing
        print("=========================================================")
        print("config.data_url  =", config.data_url)
        print("config.data_path =", config.data_path)
        moxing.file.copy_parallel(src_url=config.data_url, dst_url=config.data_path)
        print(os.listdir(config.data_path))
        print("=========================================================")
        dataset_path = os.path.join(config.data_path, 'dataset/val')
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
    if dataset_path.find('/val') > 0:
        dataset_val_path = dataset_path
    else:
        dataset_val_path = os.path.join(dataset_path, 'val')
    dataset_val_path = os.path.join(dataset_val_path, 'val_1250patches_per_image.mindrecords')
    val_dataset = open_mindrecord_dataset(dataset_val_path, do_train=False, rank=device_id,
                                          columns_list=["noise_darkened", "origin"],
                                          group_size=1, batch_size=config.val_batch_size,
                                          drop_remainder=config.drop_remainder, shuffle=False)
    loss = nn.MSELoss(reduction='mean')
    net6 = LLNet()
    eval_metrics = {'Loss': nn.Loss()}
    model = Model(net6, loss_fn=loss, optimizer=None, metrics=eval_metrics)
    mean = ops.ReduceMean()
    ssim = nn.SSIM()
    psnr = nn.PSNR()
    if not config.enable_checkpoint_dir:
        print("")
        print(checkpoint)
        ckpt = load_checkpoint(checkpoint)
        load_param_into_net(net6, ckpt)
        net6.set_train(False)
        net6.set_float16()

        metrics = model.eval(val_dataset)

        avg_ssim = 0.0
        avg_psnr = 0.0
        idx_counter = 0
        for idx, data in enumerate(val_dataset):
            idx_counter = idx_counter + 1
            y = net6(data[0])
            img1 = y.reshape((-1, 1, 17, 17))
            img2 = data[1].reshape((-1, 1, 17, 17))
            avg_ssim = avg_ssim + mean(ssim(img1, img2))
            avg_psnr = avg_psnr + mean(psnr(img1, img2))
        avg_ssim = avg_ssim / idx_counter
        avg_psnr = avg_psnr / idx_counter

        print("metric: ", metrics)
        print('PSNR = ', avg_psnr, ' SSIM = ', avg_ssim)
        print("time: ", ceil(time.time() - start_time), " seconds")
    else:
        file_list = []
        for root, _, files in os.walk(checkpoint_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.ckpt':
                    file_list.append(os.path.join(root, file))
        file_count = 0
        best_loss = 1000.0
        best_loss_checkpoint = ''
        best_psnr = 0.0
        best_psnr_checkpoint = ''
        best_ssim = 0.0
        best_ssim_checkpoint = ''

        file_list.sort()

        for checkpoint in file_list:
            if not os.path.exists(checkpoint):
                continue
            file_count += 1
            print("")
            print(checkpoint)
            ckpt = load_checkpoint(checkpoint)
            load_param_into_net(net6, ckpt)
            net6.set_train(False)
            net6.set_float16()

            metrics = model.eval(val_dataset)

            avg_ssim = 0.0
            avg_psnr = 0.0
            idx_counter = 0
            for idx, data in enumerate(val_dataset):
                idx_counter = idx_counter + 1
                y = net6(data[0])
                img1 = y.reshape((-1, 1, 17, 17))
                img2 = data[1].reshape((-1, 1, 17, 17))
                avg_ssim = avg_ssim + mean(ssim(img1, img2))
                avg_psnr = avg_psnr + mean(psnr(img1, img2))
            avg_ssim = avg_ssim / idx_counter
            avg_psnr = avg_psnr / idx_counter
            print("metric: ", metrics)
            print('PSNR = ', avg_psnr, ' SSIM = ', avg_ssim)
            loss_value = metrics['Loss']
            if  loss_value < best_loss:
                best_loss = loss_value
                best_loss_checkpoint = checkpoint
                print('*********************************************************************')
            if  best_psnr < avg_psnr:
                best_psnr = avg_psnr
                best_psnr_checkpoint = checkpoint
                print('*********************************************************************')
            if  best_ssim < avg_ssim:
                best_ssim = avg_ssim
                best_ssim_checkpoint = checkpoint
                print('*********************************************************************')
        print('*********************************************************************')
        print(file_count, ' checkpoints have been evaluated')
        print('Best Loss is ', best_loss, ' on ', best_loss_checkpoint)
        print('Best PSNR is ', best_psnr, ' on ', best_psnr_checkpoint)
        print('Best SSIM is ', best_ssim, ' on ', best_ssim_checkpoint)
        print('time: ', ceil(time.time() - start_time), ' seconds')
        print('*********************************************************************')
