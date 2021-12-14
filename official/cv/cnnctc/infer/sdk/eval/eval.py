# coding=utf-8

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
import sys
import numpy as np
import lmdb

from convert import CTCLabelConverter


def getLablesFromDataset(TEST_DATASET_PATH):
    CHARACTER = "0123456789abcdefghijklmnopqrstuvwxyz"

    max_len = int((26 + 1) // 2)

    converter = CTCLabelConverter(CHARACTER)

    env = lmdb.open(TEST_DATASET_PATH, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    if not env:
        print('cannot create lmdb from %s' % (TEST_DATASET_PATH))
        sys.exit(0)

    with env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode()))
        nSamples = nSamples

        # Filtering
        filtered_index_list = []
        for index in range(nSamples):
            index += 1  # lmdb starts with 1
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')

            if len(label) > max_len:
                continue

            illegal_sample = False
            for char_item in label.lower():
                if char_item not in CHARACTER:
                    illegal_sample = True
                    break
            if illegal_sample:
                continue

            filtered_index_list.append(index)

    length_ret = []
    text_ret = []

    print(f'num of samples in IIIT dataset: {len(filtered_index_list)}')

    for index in filtered_index_list:
        with env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
        label = label.lower()

        text, length = converter.encode([label])
        text = text.astype(np.int32)
        text_ret.append(text)
        length_ret.append(length)

    return text_ret, length_ret


def test(TEST_DATASET_PATH, prediction_file_path):
    CHARACTER = "0123456789abcdefghijklmnopqrstuvwxyz"

    converter = CTCLabelConverter(CHARACTER)

    # ds = test_dataset_creator(TEST_DATASET_PATH)
    text_ret, length_ret = getLablesFromDataset(TEST_DATASET_PATH)

    count = 0
    correct_count = 0
    preds_texts = []
    with open(prediction_file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            preds_texts.append(line)

    for i in range(0, len(text_ret)):
        text, length = text_ret[i], length_ret[i]

        preds_str = preds_texts[count]
        label_str = converter.reverse_encode(text, length)

        print("Prediction samples: {}, Ground truth:{}".format(preds_str, label_str))
        for pred, label in zip(preds_str, label_str):
            if pred == label:
                correct_count += 1
            else:
                print("hint")
            count += 1

    print('accuracy: ', correct_count / count)


if __name__ == '__main__':
    test(sys.argv[1], sys.argv[2])
