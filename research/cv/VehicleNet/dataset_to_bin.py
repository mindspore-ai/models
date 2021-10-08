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
"""dataset to bin"""
import os
import numpy as np
from dataset import create_vehiclenet_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VehicleNet train.')
    parser.add_argument('--device_id', type=int, default=None,
                        help='device id of GPU or Ascend. (Default: None)')
    parser.add_argument('--dataset_path', default=None, help='The path of dataset.')
    parser.add_argument('--result_path', default=None, help='The path of result.')
    args_opt = parser.parse_args()

    test_dataset = create_vehiclenet_dataset(dataset_path + 'test_VehicleNet.mindrecord',
                                             batch_size=1, is_training=False)
    query_dataset = create_vehiclenet_dataset(dataset_path + 'query_VehicleNet.mindrecord',
                                              batch_size=1, is_training=False)

    # test
    data_path = os.path.join(result_path, 'test')
    os.makedirs(data_path)
    img_path = os.path.join(data_path, 'img')
    os.makedirs(img_path)
    label_path = os.path.join(data_path, 'label')
    os.makedirs(label_path)
    camera_path = os.path.join(data_path, 'camera')
    os.makedirs(camera_path)

    for i, item in enumerate(test_dataset.create_dict_iterator(output_numpy=True)):
        file_name = "VehicleNet_test_bs" + str(1) + "_" + str(format(i, '08d')) + ".bin"
        file_path = img_path + "/" + file_name
        item['image'].tofile(file_path)

        file_name = "VehicleNet_test_bs" + str(1) + "_" + str(format(i, '08d')) + ".npy"
        np.save(os.path.join(label_path, file_name), item['label'])
        np.save(os.path.join(camera_path, file_name), item['camera'])
    print("="*20, "test: export bin file finished", "="*20)

    # query
    data_path = os.path.join(result_path, 'query')
    os.makedirs(data_path)
    img_path = os.path.join(data_path, 'img')
    os.makedirs(img_path)
    label_path = os.path.join(data_path, 'label')
    os.makedirs(label_path)
    camera_path = os.path.join(data_path, 'camera')
    os.makedirs(camera_path)

    for i, item in enumerate(query_dataset.create_dict_iterator(output_numpy=True)):
        file_name = "VehicleNet_query_bs" + str(1) + "_" + str(format(i, '08d')) + ".bin"
        file_path = img_path + "/" + file_name
        item['image'].tofile(file_path)

        file_name = "VehicleNet_query_bs" + str(1) + "_" + str(format(i, '08d')) + ".npy"
        np.save(os.path.join(label_path, file_name), item['label'])
        np.save(os.path.join(camera_path, file_name), item['camera'])
    print("="*20, "test: export bin file finished", "="*20)
