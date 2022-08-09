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
"""Jasper postprocess"""
import os
import numpy as np
from src.config import infer_config, symbols
from src.decoder import GreedyDecoder
import mindspore
from mindspore import Tensor


def format_text(s):
    s = s.strip().split()
    s.pop()
    while s[len(s)-1] == 'e' or s[len(s)-1] == 'ee' or s[len(s)-1] == 'eee' \
        or s[len(s)-1] == 'a' or s[len(s)-1] == 'o':
        s.pop()
    return ' '.join(s)


def compute_wer(s_file, t_file, decoder):
    total_wer, num_tokens = 0, 0
    with open(s_file, 'r', encoding='utf-8') as s, open(t_file, 'r', encoding='utf-8') as t:
        for trans, refer in zip(s, t):
            wer_inst = decoder.wer(trans.strip(), refer.strip())
            total_wer += wer_inst
            num_tokens += len(refer.strip().split())
    wer = float(total_wer) / num_tokens
    return wer


def generate_output():
    '''
    Generate output and write to file.
    '''
    config = infer_config
    if config.LMConfig.decoder_type == 'greedy':
        decoder = GreedyDecoder(labels=symbols, blank_index=len(symbols) - 1)
    else:
        raise NotImplementedError("Only greedy decoder is supported now")

    # get model out from .bin files
    predictions = []
    file_num = int(len(os.listdir(config.result_dir)) / 2)
    for i in range(file_num):
        out = "jasper_bs_" + str(
            config.batch_size_infer) + "_" + str(i) + "_0.bin"
        out_size = "jasper_bs_" + str(
            config.batch_size_infer) + "_" + str(i) + "_1.bin"
        out = np.fromfile(os.path.join(config.result_dir, out),
                          np.float32).reshape(1, -1, 29)
        out_size = np.fromfile(os.path.join(config.result_dir, out_size),
                               np.int32).reshape(-1)
        predictions.append([out, out_size])

    # decode and write to file
    f = open(config.post_out, 'w')
    for out, _ in predictions:
        out = Tensor(out, dtype=mindspore.float32)
        decoded_output, _ = decoder.decode(out)
        for d_output in decoded_output:
            transcript = d_output[0].lower()
            f.write(format_text(transcript) + '\n')

    f.close()
    print("Finished inference.")
    print("You can see the result in {}.".format(config.post_out))
    wer = compute_wer(config.post_out, 'target.txt', decoder)
    print('The average WER: ', wer)


if __name__ == "__main__":
    generate_output()
