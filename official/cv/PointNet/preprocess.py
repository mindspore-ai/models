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
"""pre process for 310 inference"""
import os
import argparse
from mindspore import context
import mindspore.dataset as ds
import numpy as np
from src.dataset import ShapeNetDataset

parser = argparse.ArgumentParser(description="lenet preprocess data")
parser.add_argument("--dataset_path", type=str, default=
                    '/home/pointnet/shapenetcore_partanno_segmentation_benchmark_v0', help="dataset path.")
parser.add_argument("--output_path", type=str, default='./datapath_BS1/', help="output path.")
parser.add_argument("--device_target", type=str, default='Ascend', help="output path.")
parser.add_argument("--device_id", default=4, help="output path.")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument(
    '--batchSize', type=int, default=1, help='input batch size')
args = parser.parse_args()

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=args.device_id)
if __name__ == '__main__':
    dataset_generator = ShapeNetDataset(
        root=args.dataset_path,
        classification=False,
        split='test',
        class_choice=[args.class_choice],
        data_augmentation=False)
    dataset = ds.GeneratorDataset(dataset_generator, column_names=["point", "label"])
    dataset = dataset.batch(args.batchSize)

    data_path = os.path.join(args.output_path, '00_data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    label_list = []
    for i, data in enumerate(dataset.create_dict_iterator()):
        print(data['label'].shape)
        file_name = 'shapenet_data_bs'+str(args.batchSize)+'_%03d'%i+'.bin'
        file_path = os.path.join(data_path, file_name)
        data['point'].asnumpy().tofile(file_path)

        label_list.append(data['label'].asnumpy())
        print('loading ', i)
    print('begin saving label')
    print(len(label_list))
    save_path = os.path.join(args.output_path, 'labels_ids.npy')
    np.save(save_path, np.array(label_list))
    print('='*20, 'export bin file finished', '='*20)
