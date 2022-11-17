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

import os
import numpy as np
from src.model_utils.config import config


def cal_acc():
    '''caculate accuracy'''
    bs = config.batch_size
    label_path = os.path.join(config.pre_result_path, "label_bs" + str(bs) + ".npy")
    file_num = len(os.listdir(config.post_result_path))

    acc_sum, sample_num = 0, 0
    label_list = np.load(label_path)
    for i in range(file_num):
        f = os.path.join(config.post_result_path, "c3d_bs" + str(bs) + "_" + str(i) + "_0.bin")
        predictions = np.fromfile(f, np.float32).reshape(bs, config.num_classes)
        label = label_list[i]
        acc = np.sum(np.argmax(predictions, 1) == label[:, -1])
        batch_size = label.shape[0]
        acc_sum += acc
        sample_num += batch_size

    accuracy_top1 = acc_sum / sample_num
    print('eval result: top_1 {:.3f}%'.format(accuracy_top1 * 100))

if __name__ == '__main__':
    cal_acc()
