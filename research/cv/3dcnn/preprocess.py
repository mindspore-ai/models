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
"""
python preprocess.py
"""
import os
import numpy as np

from src.config import config
from src.dataset import vox_generator_test


if __name__ == '__main__':
    # test dataset
    test_files = []
    with open(config.test_path) as f:
        for line in f:
            test_files.append(line[:-1])

    data_gen_test = vox_generator_test(config.data_path, test_files, config.correction)

    OFFSET_H = config.offset_height
    OFFSET_W = config.offset_width
    OFFSET_C = config.offset_channel
    HSIZE = config.height_size
    WSIZE = config.width_size
    CSIZE = config.channel_size
    PSIZE = config.pred_size

    OFFSET_PH = (config.height_size - config.pred_size) // 2
    OFFSET_PW = (config.width_size - config.pred_size) // 2
    OFFSET_PC = (config.channel_size - config.pred_size) // 2

    batches_w = int(np.ceil((config.image_width - config.width_size) / float(config.offset_width))) + 1
    batches_h = int(np.ceil((config.image_height - config.height_size) / float(config.offset_height))) + 1
    batches_c = int(np.ceil((config.image_channel - config.channel_size) / float(config.offset_channel))) + 1

    data_count = 0
    pre_result_path = os.path.join(config.pre_result_path)
    for i in range(len(test_files)):
        print('postprocess %s' % test_files[i])
        x, x_n, y = data_gen_test.__next__()
        for hi in range(batches_h):
            offset_h = min(OFFSET_H * hi, config.image_height - HSIZE)
            offset_ph = offset_h + OFFSET_PH
            for wi in range(batches_w):
                offset_w = min(OFFSET_W * wi, config.image_width - WSIZE)
                offset_pw = offset_w + OFFSET_PW
                for ci in range(batches_c):
                    offset_c = min(OFFSET_C * ci, config.image_channel - CSIZE)
                    offset_pc = offset_c + OFFSET_PC
                    data = x[offset_h:offset_h + HSIZE, offset_w:offset_w + WSIZE, offset_c:offset_c + CSIZE, :]
                    data_norm = x_n[offset_h:offset_h + HSIZE, offset_w:offset_w + WSIZE, offset_c:offset_c + CSIZE, :]
                    data_norm = np.expand_dims(data_norm, 0)
                    if not np.max(data) == 0 and np.min(data) == 0:
                        flair_t2_node = data_norm[:, :, :, :, :2]
                        t1_t1ce_node = data_norm[:, :, :, :, 2:]
                        flair_t2_node = np.transpose(flair_t2_node, axes=[0, 4, 1, 2, 3])
                        t1_t1ce_node = np.transpose(t1_t1ce_node, axes=[0, 4, 1, 2, 3])
                        flair_t2_node = flair_t2_node.astype(np.float32)
                        t1_t1ce_node = t1_t1ce_node.astype(np.float32)

                        flair_t2_path = os.path.join(pre_result_path, "flair_t2",
                                                     "flair_t2_node" + '_' + str(data_count) + '.bin')
                        t1_t1ce_path = os.path.join(pre_result_path, "t1_t1ce",
                                                    "t1_t1ce_node" + '_' + str(data_count) + '.bin')
                        flair_t2_node.tofile(flair_t2_path)
                        t1_t1ce_node.tofile(t1_t1ce_path)

                        data_count = data_count + 1
