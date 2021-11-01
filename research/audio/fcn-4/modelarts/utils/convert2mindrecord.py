# coding: utf-8
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
'''python prepare_dataset.py'''

import os
import pandas as pd
import numpy as np
from mindspore.mindrecord import FileWriter

def get_data(features_data, labels_data):
    data_list = []
    for i, (label, feature) in enumerate(zip(labels_data, features_data)):
        data_json = {"id": i, "feature": feature, "label": label}
        data_list.append(data_json)
    return data_list

def generator_md(info_name, file_path, num_classes):
    """
    generate numpy array from features of all audio clips

    Args:
        info_path (str): path to the tagging information file.
        file_path (str): path to the npy files.
        num_classes (int): number of tagging classes
    Returns:
        2 numpy array.

    """
    df = pd.read_csv(info_name, header=None)
    df.columns = [str(i) for i in range(num_classes)] + ["mp3_path"]
    data = []
    label = []
    for i in range(len(df)):
        try:
            data.append(np.load(os.path.join(file_path, df.mp3_path.values[i][:-4] + '.npy')).reshape(1, 96, 1366))
            label.append(np.array(df[df.columns[:-1]][i:i + 1])[0])
        except FileNotFoundError:
            pass
    return np.array(data), np.array(label, dtype=np.int32)

def convert_to_mindrecord(info_name, file_path, store_path, mr_name,
                          num_classes):
    """ convert dataset to mindrecord """
    num_shard = 4
    data, label = generator_md(info_name, file_path, num_classes)
    schema_json = {
        "id": {
            "type": "int32"
        },
        "feature": {
            "type": "float32",
            "shape": [1, 96, 1366]
        },
        "label": {
            "type": "int32",
            "shape": [num_classes]
        }
    }
    writer = FileWriter(
        os.path.join(store_path, '{}.mindrecord'.format(mr_name)), num_shard)
    datax = get_data(data, label)
    writer.add_schema(schema_json, "music_tagger_schema")
    writer.add_index(["id"])
    writer.write_raw_data(datax)
    writer.commit()

def prepare_train_data(info_path="config/", npy_path="npy_path/",
                       mindrecord_path="mindrecord_path/", num_classes=50):
    """
    prepare mindrecord data for model training
    Args:
        info_path (str): path to the tagging information file.
        npy_path (str): path to the npy files.
        mindrecord_path (str): path to save mindrecord files.
    """
    for cmn in ['train', 'val']:
        convert_to_mindrecord(os.path.join(info_path, 'music_tagging_{}_tmp.csv'.format(cmn)),
                              npy_path, mindrecord_path, cmn, num_classes)
    print("successfully convert npy file to mindrecord fata")
