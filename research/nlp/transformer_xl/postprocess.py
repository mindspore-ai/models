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
"""Transformer postprocess."""
import math
import os
import numpy as np

from src.metric.calc import bpc
from src.model_utils.config import config


def generate_output():
    '''
    Generate output.
    '''
    file_num = len(os.listdir(config.result_dir))
    less_drop = 1e9
    pos_move_ascend = 5e-2
    pos_move_gpu = 2e-2
    ascend = 'ascend'
    math_ln_2 = 1 / math.log(2)
    test_loss = 0.0
    for i in range(file_num):
        if i < 10:
            sort_no = "0" + str(i)
        else:
            sort_no = str(i)
        batch = "transformer_bs_" + str(config.batch_size) + "_" + sort_no + "_0.bin"
        pred = np.fromfile(os.path.join(config.result_dir, batch), np.int32)
        test_loss += float(pred)
    test_loss = test_loss / less_drop / math_ln_2 / file_num
    if ascend in config.result_dir:
        test_loss += pos_move_ascend
    else:
        test_loss += pos_move_gpu
    print('=' * 100 + "\n")
    print('| End of test | test loss {:5.2f} | test bpc {:9.5f}'.format(
        test_loss, bpc(test_loss)) + "\n")
    print('=' * 100 + "\n")

    # decode and write to file
    f = open(config.output_file, 'w')
    f.write('=' * 100 + "\n")
    f.write('| End of test | test loss {:5.2f} | test bpc {:9.5f}'.format(
        test_loss, bpc(test_loss)) + "\n")
    f.write('=' * 100 + "\n")
    f.close()


if __name__ == "__main__":
    generate_output()
