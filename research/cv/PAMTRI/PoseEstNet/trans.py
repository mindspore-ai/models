# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
########################## transform veri dataset ##########################
train veri dataset and get MultiTaskNet dataset:
python trans.py --cfg config.yaml --ckpt_path Your.ckpt --data_dir datapath
"""

import os
import argparse
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

from mindspore import context
from mindspore.dataset.transforms.transforms import Compose
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.model import get_pose_net
from src.config import cfg, update_config
from src.utils.function import output_preds
from src.dataset import VeRiTransDataset

parser = argparse.ArgumentParser(description='Transform veri dataset')

parser.add_argument('--cfg', required=True, type=str)
parser.add_argument('--ckpt_path', type=str, default='')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--device_target', type=str, default="Ascend")

args = parser.parse_args()

if __name__ == '__main__':
    update_config(cfg, args)

    target = args.device_target
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False, device_id=device_id)

    test_data = VeRiTransDataset(cfg, args.data_dir, 'test')
    query_data = VeRiTransDataset(cfg, args.data_dir, 'query')
    train_data = VeRiTransDataset(cfg, args.data_dir, 'train')

    test_dataloader = ds.GeneratorDataset(test_data, column_names=['input', 'center', 'scale'],
                                          num_parallel_workers=1, shuffle=False, num_shards=1, shard_id=0)

    query_dataloader = ds.GeneratorDataset(query_data, column_names=['input', 'center', 'scale'],
                                           num_parallel_workers=1, shuffle=False, num_shards=1, shard_id=0)

    train_dataloader = ds.GeneratorDataset(train_data, column_names=['input', 'center', 'scale'],
                                           num_parallel_workers=1, shuffle=False, num_shards=1, shard_id=0)

    trans = Compose([
        vision.ToTensor(),
        vision.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), is_hwc=False)
    ])

    test_dataloader = test_dataloader.map(operations=trans, input_columns="input", num_parallel_workers=1)
    test_dataloader = test_dataloader.batch(batch_size=32, drop_remainder=False, num_parallel_workers=1)

    query_dataloader = query_dataloader.map(operations=trans, input_columns="input", num_parallel_workers=1)
    query_dataloader = query_dataloader.batch(batch_size=32, drop_remainder=False, num_parallel_workers=1)

    train_dataloader = train_dataloader.map(operations=trans, input_columns="input", num_parallel_workers=1)
    train_dataloader = train_dataloader.batch(batch_size=32, drop_remainder=False, num_parallel_workers=1)

    network = get_pose_net(cfg)
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(network, param_dict)

    output_preds(cfg, test_dataloader, test_data, network, args.data_dir, 'test', args.data_dir)
    output_preds(cfg, query_dataloader, query_data, network, args.data_dir, 'query', args.data_dir)
    output_preds(cfg, train_dataloader, train_data, network, args.data_dir, 'train', args.data_dir)
