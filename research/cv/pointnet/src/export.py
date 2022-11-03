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
"""export"""
import os
import argparse
import numpy as np
from mindspore import Tensor, export, load_checkpoint, context
from src.network import PointNetDenseCls
parser = argparse.ArgumentParser(description='MindSpore Pointnet Segmentation')
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--device_id', type=int, default=4, help='device id')
parser.add_argument('--device_target', default='Ascend', help='device id')
parser.add_argument('--file_format', type=str, default='MINDIR', help="export file format")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

args = parser.parse_args()
context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)
num_classes = 4
classifier = PointNetDenseCls(k=num_classes, feature_transform=args.feature_transform)
if not os.path.exists('./mindir'):
    os.mkdir('./mindir')
load_checkpoint(args.model, net=classifier)
input_data = np.random.uniform(0.0, 1.0, size=[1, 3, 2500]).astype(np.float32)
export(classifier, Tensor(input_data), file_name='./mindir/pointnet', file_format=args.file_format)
print("successfully export model")
