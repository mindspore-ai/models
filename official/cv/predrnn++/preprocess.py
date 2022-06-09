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
import os
import argparse
from pathlib import Path
import numpy as np
from data_provider.mnist_to_mindrecord import create_mnist_dataset
from config import config

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mindrecord', type=str, default='')
    parser.add_argument('--device_id', type=int, default=0)
    args_opt = parser.parse_args()
    device_num = config.device_num
    rank = 0
    ds = create_mnist_dataset(dataset_files=args_opt.test_mindrecord, rank_size=device_num, \
        rank_id=rank, do_shuffle=False, batch_size=config.batch_size)

    source_ids_path = os.path.join(str(ROOT/config.result_path), "00_data")
    target_ids_path = os.path.join(str(ROOT/config.result_path), "01_data")
    if not os.path.exists(source_ids_path):
        os.makedirs(source_ids_path)
    if not os.path.exists(target_ids_path):
        os.makedirs(target_ids_path)
    patched_width = int(config.img_width/config.patch_size)
    mask_true = np.zeros((config.batch_size,
                          config.seq_length-config.input_length-1,
                          patched_width,
                          patched_width,
                          int(config.patch_size)**2*int(config.img_channel)), dtype=np.float32)

    for i, data in enumerate(ds.create_dict_iterator(output_numpy=True, num_epochs=1)):
        test_ims = data['input_x']
        file_name = "predrnn++_bs" + str(config.batch_size) + "_" + str(i) + ".bin"
        test_ims.tofile(os.path.join(source_ids_path, file_name))
        mask_true.tofile(os.path.join(target_ids_path, file_name))

    print("="*20, "export bin files finished", "="*20)
