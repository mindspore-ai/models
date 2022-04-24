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
"""run export"""
import argparse
import os

import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from mindspore.train.serialization import export

from src.utils import get_config
from src.utils import get_model_dataset
from src.utils import get_params_for_net


def run_export(args):
    """run export"""
    cfg_path = args.cfg_path
    ckpt_path = args.ckpt_path
    file_name = args.file_name
    file_format = args.file_format

    cfg = get_config(cfg_path)

    device_target = cfg['train_config']['device_target']
    device_id = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target, device_id=device_id)

    pointpillarsnet, _ = get_model_dataset(cfg)

    params = load_checkpoint(ckpt_path)
    new_params = get_params_for_net(params)
    load_param_into_net(pointpillarsnet, new_params)

    v = cfg['eval_input_reader']['max_number_of_voxels']
    p = cfg['model']['voxel_generator']['max_number_of_points_per_voxel']
    n = cfg['model']['num_point_features']
    voxels = Tensor(np.zeros((1, v, p, n)), mstype.float32)
    num_points = Tensor(np.zeros((1, v)), mstype.int32)
    coors = Tensor(np.zeros((1, v, 4)), mstype.int32)
    if cfg['model']['use_bev']:
        pc_range = np.array(cfg['model']['voxel_generator']['point_cloud_range'])
        voxel_size = np.array(cfg['model']['voxel_generator']['voxel_size'])
        x, y, z = ((pc_range[3:] - pc_range[:3]) / voxel_size).astype('int32')
        bev_map = Tensor(np.zeros((1,) + (z, x * 2, y * 2)), mstype.float32)

        export(pointpillarsnet, voxels, num_points, coors, bev_map, file_name=file_name, file_format=file_format)
    else:
        export(pointpillarsnet, voxels, num_points, coors, file_name=file_name, file_format=file_format)

    print(f'{file_name}.mindir exported successfully!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', required=True, help='')
    parser.add_argument('--ckpt_path', required=True, help='')
    parser.add_argument('--file_name', default='model', help='')
    parser.add_argument('--file_format', default='MINDIR', choices=['MINDIR', 'AIR'], help='')

    parse_args = parser.parse_args()
    run_export(parse_args)
