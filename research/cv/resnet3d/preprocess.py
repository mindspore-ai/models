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
"""
Pre-process for 310 infer
"""
import os
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

from src.dataset import create_eval_dataset
from src.config import config as cfg
from src.videodataset_multiclips import get_target_path

if __name__ == '__main__':
    cfg.batch_size = 1
    print(cfg)
    if cfg.n_classes == 101:
        dataset_name = 'ucf101'
    elif cfg.n_classes == 51:
        dataset_name = 'hmdb51'
    else:
        dataset_name = ''
    predict_data = create_eval_dataset(
        cfg.video_path, cfg.annotation_path, cfg)
    data_path = os.path.abspath(os.path.dirname(
        __file__)) + "/scripts/preprocess_Result/data/"
    label_path = os.path.abspath(os.path.dirname(
        __file__)) + "/scripts/preprocess_Result/label/"
    print(data_path)
    if os.path.exists(data_path):
        print("=====================flag=================")
        os.system('rm -rf ' + data_path)
    os.makedirs(data_path)
    total_target_path = get_target_path(cfg.annotation_path)
    with total_target_path.open('r') as f:
        total_target_data = json.load(f)
    results = {'results': defaultdict(list)}
    step_size = predict_data.get_dataset_size()
    label_list = {}
    for step, data in enumerate(predict_data.create_dict_iterator(output_numpy=True)):
        x, label = data['data'][0], data['label'].tolist()
        video_ids, segments = zip(
            *total_target_data['targets'][str(label[0])])
        x_list = np.split(x, x.shape[0], axis=0)
        target = []
        for idx, clip in enumerate(x_list):
            file_name = dataset_name + '_bs' + \
                str(cfg.batch_size) + '_' + str(step) + "_" + str(idx) + '.bin'
            file_path = data_path + file_name
            clip.tofile(file_path)
            target.append({
                'file_name': file_name,
                'idx': idx
            })
        label_list[label[0]] = target
        print("Processing {} / {}".format(step, step_size))
    if os.path.exists(label_path):
        os.system('rm -rf ' + label_path)
    os.makedirs(label_path)
    label_path = label_path + 'label.json'
    with Path(label_path).open('w') as dst_file:
        json.dump(label_list, dst_file)
    print("="*20, 'export bin file finished', '='*20)
