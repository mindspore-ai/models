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
########################## eval PoseEstNet ##########################
eval lenet according to model file:
python eval.py --cfg config.yaml --ckpt_path Your.ckpt --data_dir datapath
"""
import os
import json
import argparse
from pathlib import Path
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.model import get_pose_net
from src.dataset import create_dataset, get_label
from src.utils.function import validate
from src.config import cfg, update_config

parser = argparse.ArgumentParser(description='Eval PoseEstNet')

parser.add_argument('--cfg', type=str, default='')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--ckpt_path', type=str, default='')
parser.add_argument('--device_target', type=str, default="Ascend")

args = parser.parse_args()

if __name__ == '__main__':
    update_config(cfg, args)

    target = args.device_target
    device_id = int(os.getenv('DEVICE_ID'))

    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False, device_id=device_id)

    data, dataset = create_dataset(cfg, args.data_dir, is_train=False)
    json_path = get_label(cfg, args.data_dir)
    dst_json_path = Path(json_path)
    with dst_json_path.open('r') as dst_file:
        allImage = json.load(dst_file)
    ckpt_path = args.ckpt_path

    # define net
    network = get_pose_net(cfg)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(network, param_dict)

    print("ckpt path :{}".format(ckpt_path))
    print("============== Starting Testing ==============")

    perf_indicator = validate(cfg, dataset, data, network, allImage)
