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
##############pre process for Ascend 310 infer############################################
"""
import os
import time
from math import ceil
from mindspore import context

from src.dataset import open_mindrecord_dataset
from src.model_utils.config import config

def write_bin_files():
    start_time = time.time()
    device_id = config.device_id
    dataset_path = config.dataset_path
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
    dataset_val_filename = os.path.join(dataset_val_path, 'val_1250patches_per_image.mindrecords')
    val_dataset = open_mindrecord_dataset(dataset_val_filename, do_train=False, rank=device_id,
                                          columns_list=["noise_darkened", "origin"],
                                          group_size=1, batch_size=1,
                                          drop_remainder=config.drop_remainder, shuffle=False)
    noise_darkened_dir = os.path.join(dataset_val_path, 'noise_darkened')
    origin_dir = os.path.join(dataset_val_path, 'origin')
    if not os.path.exists(noise_darkened_dir):
        os.mkdir(noise_darkened_dir)
    if not os.path.exists(origin_dir):
        os.mkdir(origin_dir)
    file_count = 0
    for idx, data in enumerate(val_dataset):
        noise_darkened = data[0].asnumpy()
        origin = data[1].asnumpy()
        noise_darkened_file_name = os.path.join(noise_darkened_dir, "%07d.bin"%(idx))
        origin_file_name = os.path.join(origin_dir, "%07d.bin"%(idx))
        noise_darkened.tofile(noise_darkened_file_name)
        origin.tofile(origin_file_name)
        file_count += 1
        if file_count % 10000 == 0:
            print(file_count)
    print(file_count)
    print("time: ", ceil(time.time() - start_time), " seconds")

if __name__ == '__main__':
    write_bin_files()
