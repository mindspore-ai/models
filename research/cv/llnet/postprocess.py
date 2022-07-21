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
"""
##############post process for Ascend 310 infer###########################################
"""
import os
import time
from math import log10, ceil
import numpy as np
from skimage.metrics import structural_similarity

from mindspore import context

from src.model_utils.config import config

def calc_psnr(origin_bin, infered_bin):
    mse = np.mean((origin_bin - infered_bin) ** 2)
    if mse <= 1.0e-10:
        return 100.0
    max_pixel = 1.0
    psnr = 10.0 * log10(max_pixel / mse)
    return psnr

def calc_ssim(bin_1, bin_2, max_val=1.0):
    return structural_similarity(bin_1, bin_2, data_range=1.0, channel_axis=None)

def calc_metrics():
    start_time = time.time()
    device_id = config.device_id
    dataset_path = config.dataset_path
    print('dataset_path = ', dataset_path)
    if config.use_pynative_mode:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target,
                            device_id=device_id)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target,
                            device_id=device_id, save_graphs=False)
    if dataset_path.find('/val') > 0:
        dataset_val_path = dataset_path
    else:
        dataset_val_path = os.path.join(dataset_path, 'val')

    if dataset_val_path.find('/origin') > 0:
        origin_dir = dataset_val_path
    else:
        origin_dir = os.path.join(dataset_val_path, 'origin')
    infered_dir = "./result_Files"

    file_count = 0
    ssim_sum = 0
    psnr_sum = 0

    for _, file_name in enumerate(os.listdir(origin_dir)):
        origin_file_name = os.path.join(origin_dir, file_name)
        infered_file_name = os.path.join(infered_dir, file_name)
        origin_bin = np.fromfile(origin_file_name, dtype=np.float32)
        infered_bin = np.fromfile(infered_file_name, dtype=np.float32)
        ssim_score = calc_ssim(origin_bin, infered_bin)
        psnr_score = calc_psnr(origin_bin, infered_bin)
        ssim_sum += ssim_score
        psnr_sum += psnr_score
        file_count += 1
        if file_count % 10000 == 0:
            print(file_count)
            print('PSNR = %6.3f, SSIM = %.3f' % (psnr_sum / file_count, ssim_sum / file_count))
    print(file_count)
    print('PSNR = %6.3f , SSIM = %.3f' % (psnr_sum / file_count, ssim_sum / file_count))
    print("time: ", ceil(time.time() - start_time), " seconds")

if __name__ == '__main__':
    calc_metrics()
