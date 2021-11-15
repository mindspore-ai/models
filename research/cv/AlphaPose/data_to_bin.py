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
'''
This file generates binary files.
'''
from __future__ import division

import os
import numpy as np
from src.dataset import CreateDatasetCoco

def search_dir(save_path):
    '''
    search_dir
    '''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    bin_path = save_path+"images"
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)

def data_to_bin(val_dataset, save_path):
    '''
    data_to_bin
    '''
    i = 0
    centers = []
    scales = []
    scores = []
    ids = []
    for item in val_dataset.create_dict_iterator():
        inputs = item['image'].asnumpy()
        inputs_flipped = inputs[:, :, :, ::-1]
        c = item['center'].asnumpy()
        s = item['scale'].asnumpy()
        score = item['score'].asnumpy()
        d = item['id'].asnumpy()
        inputs.tofile(save_path+"images//"+str(i)+".bin")
        inputs_flipped.tofile(save_path+"images//flipped"+str(i)+".bin")
        centers.append(c.astype(np.float32))
        scales.append(s.astype(np.float32))
        scores.append(score.astype(np.float32))
        ids.append(d.astype(np.float32))
        i = i+1
    np.save(os.path.join(save_path, "centers.npy"), np.array(centers, dtype=np.float32))
    np.save(os.path.join(save_path, "scales.npy"), np.array(scales, dtype=np.float32))
    np.save(os.path.join(save_path, "scores.npy"), np.array(scores, dtype=np.float32))
    np.save(os.path.join(save_path, "ids.npy"), np.array(ids, dtype=np.float32))

def main():
    valid_dataset = CreateDatasetCoco(
        train_mode=False,
        num_parallel_workers=2,
    )
    save_path = "data_bin//"
    search_dir(save_path)
    data_to_bin(valid_dataset, save_path)

if __name__ == '__main__':
    main()
