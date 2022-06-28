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

'''
postprocess script.
'''

import os
import argparse
import numpy as np
from src.utils import compute_corrector_prf, compute_sentence_level_prf

parser = argparse.ArgumentParser(description="postprocess")
parser.add_argument("--batch_size", type=int, default=2, help="Eval batch size, default is 2")
parser.add_argument("--result_dir_00", type=str, default="./result_files/result_00", help="infer result Files")
parser.add_argument("--result_dir_01", type=str, default="./result_files/result_01", help="infer result Files")
parser.add_argument("--result_dir_02", type=str, default="./result_files/result_02", help="infer result Files")
parser.add_argument("--result_dir_03", type=str, default="./result_files/result_03", help="infer result Files")
parser.add_argument("--result_dir_04", type=str, default="./result_files/result_04", help="infer result Files")
parser.add_argument("--result_dir_05", type=str, default="./result_files/result_05", help="infer result Files")

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    file_names = os.listdir(args.result_dir_00)
    results = []
    max_seq_len = 512
    min_seq_len = 1
    for f in file_names:
        f_name_00 = os.path.join(args.result_dir_00, f)
        f_name_01 = os.path.join(args.result_dir_01, f)
        f_name_02 = os.path.join(args.result_dir_02, f)
        f_name_03 = os.path.join(args.result_dir_03, f)
        f_name_04 = os.path.join(args.result_dir_04, f)
        f_name_05 = os.path.join(args.result_dir_05, f)

        result_00 = np.fromfile(f_name_00, np.float32)
        result_01 = np.fromfile(f_name_01, np.float32)
        result_02 = np.fromfile(f_name_02, np.float32)
        result_03 = np.fromfile(f_name_03, np.float32)
        result_04 = np.fromfile(f_name_04, np.float32)
        result_05 = np.fromfile(f_name_05, np.float32)
        # (2, 512)
        result_00 = result_00.reshape(args.batch_size, max_seq_len)
        result_01 = result_01.reshape(args.batch_size, max_seq_len)
        result_02 = result_02.reshape(args.batch_size, max_seq_len)
        result_03 = result_03.reshape(args.batch_size, max_seq_len)
        result_04 = result_04.reshape(args.batch_size, max_seq_len)
        result_05 = result_05.reshape(args.batch_size, min_seq_len)
        for i in range(args.batch_size):
            original_tokens = result_00[i].reshape(1, result_00[i].size)
            cor_y = result_01[i].reshape(1, result_01[i].size)
            cor_y_hat = result_02[i].reshape(1, result_02[i].size)
            det_y_hat = result_03[i].reshape(1, result_03[i].size)
            det_labels = result_04[i].reshape(1, result_04[i].size)
            batch_seq_len = result_05[i].reshape(1, result_05[i].size)
            for src, tgt, predict, det_predict, det_label, seq_len in zip(original_tokens, cor_y, cor_y_hat, det_y_hat,
                                                                          det_labels, batch_seq_len):
                # src: incorrect original, tgt: correct article word segmentation, ids predict: model predicted word segmentation, ids det_predict: predicted, det det_label: DET label
                seq_len_ = int(seq_len[0] - 2)
                _src = src[1: seq_len_ + 1].tolist()
                _tgt = tgt[1: seq_len_ + 1].tolist()
                _predict = predict[1: seq_len_ + 1].tolist()
                results.append((_src, _tgt, _predict,))

    compute_corrector_prf(results)
    compute_sentence_level_prf(results)
