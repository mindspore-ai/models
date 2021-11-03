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

"""ctc evaluation"""
import os
import numpy as np
from src.metric import LER
from src.model_utils.config import config
from mindspore import Tensor


def run_eval():
    '''eval_function'''
    file_name = os.listdir(config.label_dir)
    metrics = LER(beam=config.beam)
    for f in file_name:
        f_name = os.path.join(config.result_dir, f.split('.')[0] + '_0.bin')
        logits = np.fromfile(f_name, np.float32).reshape(config.max_sequence_length, -1, config.n_class)
        logits = Tensor(logits)
        labels = np.fromfile(os.path.join(config.label_dir, f), np.int32).reshape(config.test_batch_size, -1)
        labels = Tensor(labels)
        seq_len = np.fromfile(os.path.join(config.seqlen_dir, f), np.int32).reshape(-1)
        seq_len = Tensor(seq_len)
        metrics.update(logits, labels, seq_len)
    print("Ler(310) is: ", metrics.eval())
    metrics.clear()


if __name__ == "__main__":
    run_eval()
