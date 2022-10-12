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
"""preprocess"""
import os
import json
from pathlib import Path
import numpy as np

from mindspore.common import set_seed

from src.config import get_arguments
from src.dataset import createDataset, createDatasetJson

set_seed(1)

def preLauch():
    """parse the console argument"""
    parser = get_arguments()

    # PreProcess
    parser.add_argument('--mode', type=str, default='preprocess for inference310')
    parser.add_argument('--output_dir', type=str, default='./preprocess_result', \
        help="path to save binary files and json files")
    return parser.parse_args()

if __name__ == '__main__':
    args = preLauch()
    _, test_dataset = createDataset(args.dataset_path, args.category)
    root_path = args.output_dir
    test_path = os.path.join(root_path, 'img')
    os.makedirs(test_path, exist_ok=True)
    label_path = os.path.join(root_path, 'label')
    os.makedirs(label_path, exist_ok=True)

    test_label = {}
    for j, data in enumerate(test_dataset.create_dict_iterator()):
        test_single_lable = {}

        img = data['img'].asnumpy().astype(np.float32)
        assert img.shape == (1, 3, 256, 256)

        gt = data['gt'].asnumpy().astype(np.float32)
        assert gt.shape == (1, 1, 256, 256)

        label = data['label'].asnumpy()
        idx = data['idx'].asnumpy()

        # save img
        file_name_img = "data_img" + "_" + str(j).zfill(3) + ".bin"
        file_path = os.path.join(test_path, file_name_img)
        img.tofile(file_path)

        test_single_lable['gt'] = gt.tolist()
        test_single_lable['label'] = label.tolist()
        test_single_lable['idx'] = idx.tolist()

        test_label['{}'.format(j)] = test_single_lable

        test_json_path = createDatasetJson(
            dataset_path=args.dataset_path,
            category=args.category,
            test_data=test_dataset,
            phase='test')
    test_label['test_json_path'] = test_json_path

    test_label_json_path = Path(os.path.join(label_path, 'test_label.json'))
    with test_label_json_path.open('w') as json_path:
        json.dump(test_label, json_path)
