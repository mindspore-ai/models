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
"""GRU preprocess script."""
import os
import argparse
from src.dataset import create_transformer_dataset
from src.dataset import MsAudioDataset
parser = argparse.ArgumentParser('preprocess')
parser.add_argument('--data_path', type=str, default='', help='eval data dir')
parser.add_argument('--chars_dict_path', type=str, default='', help='your/path/dataset/lang_1char/train_chars.txt')


if __name__ == "__main__":
    args = parser.parse_args()
    mindrecord_file = args.data_path
    if not os.path.exists(mindrecord_file):
        print("dataset file {} not exists, please check!".format(mindrecord_file))
        raise ValueError(mindrecord_file)
    result_path = "./preprocess_Result"
    chars_dict_path = args.chars_dict_path
    char_list, _, _ = MsAudioDataset.process_dict(chars_dict_path)
    test_batch_size = 1

    dataset = create_transformer_dataset(epoch_count=1, rank_size=1, rank_id=0, do_shuffle="true",
                                         data_json_path=mindrecord_file, chars_dict_path=chars_dict_path, batch_size=1)
    source_eos_features_path = os.path.join("./preprocess_Result", "00_data")
    source_eos_mask_path = os.path.join("./preprocess_Result", "01_data")
    target_sos_ids_path = os.path.join("./preprocess_Result", "02_data")
    target_sos_mask_path = os.path.join(result_path, "03_data")
    target_eos_ids_path = os.path.join(result_path, "04_data")
    target_eos_mask_path = os.path.join(result_path, "05_data")
    os.makedirs(source_eos_features_path)
    os.makedirs(source_eos_mask_path)
    os.makedirs(target_sos_ids_path)
    os.makedirs(target_sos_mask_path)
    os.makedirs(target_eos_ids_path)
    os.makedirs(target_eos_mask_path)
    for i, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        target_eos_ids = data["target_eos_ids"]
        file_name = "speech_bs" + str(test_batch_size) + "_" + str(i) + ".bin"
        data["source_eos_features"].tofile(os.path.join(source_eos_features_path, file_name))
        data["source_eos_mask"].tofile(os.path.join(source_eos_mask_path, file_name))
        data["target_sos_ids"].tofile(os.path.join(target_sos_ids_path, file_name))
        data["target_sos_mask"].tofile(os.path.join(target_sos_mask_path, file_name))
        data["target_eos_ids"].tofile(os.path.join(target_eos_ids_path, file_name))
        data["target_eos_mask"].tofile(os.path.join(target_eos_mask_path, file_name))
    print("=" * 20, "export bin files finished", "=" * 20)
