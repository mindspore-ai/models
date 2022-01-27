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

from pathlib import Path
from argparse import ArgumentParser
import numpy as np


def cal_mse(tst_dir, out_dir):
    sse = 0
    cnt = 0
    for f in tst_dir.glob('*.s16'):
        gt_file = tst_dir / (f.stem + '.s16')
        out_file = out_dir / (f.stem + '.pcm')
        gt = np.fromfile(gt_file, dtype='int16')
        out = np.fromfile(out_file, dtype='int16')
        size = min(len(gt), len(out))
        sse += np.sum((gt[:size] / 32767.0 - out[:size] / 32767.0) ** 2)
        cnt += size
    mse = sse / cnt
    print('Evaluated MSE is: ', mse)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('test_data_path', type=Path)
    parser.add_argument('output_path', type=Path)

    args = parser.parse_args()

    # Calculate MSE
    cal_mse(args.test_data_path, args.output_path)
