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
import json
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

info_data = {}
def cal_mse(tst_dir, out_dir, mode="mxbase"):
    sse = 0
    cnt = 0
    loop = 0
    gt = None
    for f in out_dir.glob('*.pcm'):
        loop += 1
        if mode == "mxbase":
            gt_file = tst_dir / (info_data[f.stem] + '.s16')
            out_file = out_dir / (f.stem + '.pcm')
            with open(out_file) as f:
                mxbase = f.read().split()
                mxbase = [int(i) for i in mxbase]
                out = np.array(mxbase).astype("int16")

        else:
            gt_file = tst_dir / (f.stem + '.s16')
            out_file = out_dir / (f.stem + '.pcm')
            out = np.fromfile(out_file, dtype='int16')
        gt = np.fromfile(gt_file, dtype='int16')
        size = min(len(gt), len(out))
        sse += np.sum((gt[:size] / 32767.0 - out[:size] / 32767.0) ** 2)
        cnt += size
    mse = sse / cnt
    print("loop is ", loop)
    print('Evaluated MSE is: ', mse)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--test_data_path', default='../data/eval-data', type=Path)
    parser.add_argument('--output_path', default='../result/mxbase', type=Path)
    parser.add_argument('--json_path', default='../dataprocess/info.json', type=str)
    parser.add_argument('--mode', default='mxbase', type=str)

    args = parser.parse_args()

    # Calculate MSE
    if args.mode == 'mxbase':
        f1 = open(args.json_path, 'r')
        info_data = json.load(f1)
    MODE = args.mode
    cal_mse(args.test_data_path, args.output_path, MODE)
