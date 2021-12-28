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
""" preprocess. """

import os
import numpy as np

from src.dataset import create_dataset
from src.model_utils.config import config


def generate_bin():
    """ Generate bin Files """

    _, _, test_dataloader_head, test_dataloader_tail = create_dataset(
        data_path=config.data_path,
        config=config,
        is_train=False
    )

    positive_head_path = os.path.join(config.result_path, "00_positive_head")
    negative_head_path = os.path.join(config.result_path, "01_negative_head")
    filter_bias_head_path = os.path.join(config.result_path, "02_filter_bias_head")
    positive_tail_path = os.path.join(config.result_path, "00_positive_tail")
    negative_tail_path = os.path.join(config.result_path, "01_negative_tail")
    filter_bias_tail_path = os.path.join(config.result_path, "02_filter_bias_tail")

    os.makedirs(positive_head_path)
    os.makedirs(negative_head_path)
    os.makedirs(filter_bias_head_path)
    os.makedirs(positive_tail_path)
    os.makedirs(negative_tail_path)
    os.makedirs(filter_bias_tail_path)

    for i, data in enumerate(test_dataloader_head.create_dict_iterator(output_numpy=True)):
        file_name = "wn18rr_head_bs" + str(config.test_batch_size) + "_" + str(i) + ".bin"
        positive = data['positive'].astype(np.int32)
        positive.tofile(os.path.join(positive_head_path, file_name))

        negative = data['negative'].astype(np.int32)
        negative.tofile(os.path.join(negative_head_path, file_name))

        filter_bias = data['filter_bias'].astype(np.float32)
        filter_bias.tofile(os.path.join(filter_bias_head_path, file_name))

    print("=" * 20, "export head bin files finished", "=" * 20)

    for i, data in enumerate(test_dataloader_tail.create_dict_iterator(output_numpy=True)):
        file_name = "wn18rr_tail_bs" + str(config.test_batch_size) + "_" + str(i) + ".bin"
        positive = data['positive'].astype(np.int32)
        positive.tofile(os.path.join(positive_tail_path, file_name))

        negative = data['negative'].astype(np.int32)
        negative.tofile(os.path.join(negative_tail_path, file_name))

        filter_bias = data['filter_bias'].astype(np.float32)
        filter_bias.tofile(os.path.join(filter_bias_tail_path, file_name))

    print("=" * 20, "export tail bin files finished", "=" * 20)


if __name__ == '__main__':
    generate_bin()
