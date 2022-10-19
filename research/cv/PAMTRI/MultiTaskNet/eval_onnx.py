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
"""MultiTaskNet onnx_eval"""
import ast
import argparse
from src.utils.evaluate import onnx_test
from src.dataset.dataset import eval_create_dataset
import onnxruntime as ort
parser = argparse.ArgumentParser(description='eval MultiTaskNet')

parser.add_argument('--device_target', type=str, default="GPU")
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--root', type=str, default='./data', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='veri', help="name of the dataset")
parser.add_argument('--height', type=int, default=256, help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=256, help="width of an image (default: 256)")
parser.add_argument('--test-batch', default=1, type=int, help="test batch size")
parser.add_argument('--heatmapaware', type=ast.literal_eval, default=False, help="embed heatmaps to images")
parser.add_argument('--segmentaware', type=ast.literal_eval, default=False, help="embed segments to images")
parser.add_argument('--onnx_path', type=str, default='')
args = parser.parse_args()

def create_session(onnx_path, target_device):
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f'Unsupported target device {target_device!r}. Expected one of: "CPU", "GPU"')

    sessions = ort.InferenceSession(onnx_path, providers=providers)
    input_names1 = sessions.get_inputs()[0].name
    input_names2 = sessions.get_inputs()[1].name
    input_names = [input_names1, input_names2]
    return sessions, input_names

if __name__ == '__main__':
    target = args.device_target
    train_dataset_path = args.root

    query_dataloader, gallery_dataloader, num_train_vids, \
        num_train_vcolors, num_train_vtypes, _vcolor2label, \
            _vtype2label = eval_create_dataset(dataset_dir=args.dataset,
                                               root=train_dataset_path,
                                               width=args.width,
                                               height=args.height,
                                               keyptaware=True,
                                               heatmapaware=args.heatmapaware,
                                               segmentaware=args.segmentaware,
                                               train_batch=args.test_batch)

    session, input_name = create_session(args.onnx_path, 'GPU')

    _distmat = onnx_test(session, input_name, True, True, query_dataloader, gallery_dataloader, _vcolor2label,
                         _vtype2label, return_distmat=True)
