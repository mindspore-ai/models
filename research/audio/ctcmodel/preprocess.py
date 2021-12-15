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
"""GRU preprocess script."""
import os
from src.dataset import create_dataset
from src.model_utils.config import config

if __name__ == "__main__":
    mindrecord_file = config.test_path
    if not os.path.exists(mindrecord_file):
        print("dataset file {} not exists, please check!".format(mindrecord_file))
        raise ValueError(mindrecord_file)
    dataset = create_dataset(mindrecord_file, False, config.test_batch_size)
    feature_path = os.path.join(config.result_path, "00_data")
    masks_path = os.path.join(config.result_path, "01_data")
    label_path = os.path.join(config.result_path, "02_data")
    seqlen_path = os.path.join(config.result_path, "03_data")
    os.makedirs(feature_path)
    os.makedirs(masks_path)
    os.makedirs(label_path)
    os.makedirs(seqlen_path)

    for i, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        file_name = "ctc_bs" + str(config.test_batch_size) + "_" + str(i) + ".bin"
        data["feature"].tofile(os.path.join(feature_path, file_name))
        data["masks"].tofile(os.path.join(masks_path, file_name))
        data["label"].tofile(os.path.join(label_path, file_name))
        data["seq_len"].tofile(os.path.join(seqlen_path, file_name))
    print("=" * 20, "export bin files finished", "=" * 20)
