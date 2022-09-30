#!/bin/bash
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
"""
########################## eval_onnx PoseEstNet ##########################
eval lenet according to model file:
python eval_onnx.py --cfg config.yaml --data_dir datapath --onnx_path onnxpath
"""
import json
import argparse
from pathlib import Path
import onnxruntime as ort
from src.dataset import create_dataset, get_label
from src.utils.function import onnx_validate
from src.config import cfg, update_config

parser = argparse.ArgumentParser(description='Eval PoseEstNet')

parser.add_argument('--cfg', type=str, default='')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--onnx_path', type=str, default='')
parser.add_argument('--device_target', type=str, default="GPU")

args = parser.parse_args()

def create_session(onnx_path, target_device, is_train):
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f'Unsupported target device {target_device!r}. Expected one of: "CPU", "GPU"')
    sessions = ort.InferenceSession(onnx_path, providers=providers)
    input_names = sessions.get_inputs()[0].name
    onnx_train = is_train
    return sessions, input_names, onnx_train


if __name__ == '__main__':
    update_config(cfg, args)
    target = args.device_target
    data, dataset = create_dataset(cfg, args.data_dir, is_train=False)
    json_path = get_label(cfg, args.data_dir)
    dst_json_path = Path(json_path)

    with dst_json_path.open('r') as dst_file:
        allImage = json.load(dst_file)


    print(target)
    session, input_name, train = create_session(args.onnx_path, target, False)

    print("============== Starting Testing ==============")

    perf_indicator = onnx_validate(cfg, dataset, data, session, input_name, train, allImage)
