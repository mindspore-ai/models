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
"""Run evaluation for a model exported to ONNX"""
import argparse
import mindspore.nn as nn
import onnxruntime as ort

from src.config import config
from src.dataset import create_dataset_ImageNet

parser = argparse.ArgumentParser(description='Eval Onnx')
parser.add_argument('--device_target', type=str, default='GPU', help='Device target')
parser.add_argument('--dataset_path', type=str, default='', help='Dataset path')
parser.add_argument('--onnx_path', type=str, default='',
                    help='onnx file path')
args = parser.parse_args()


def create_session(checkpoint_path, target_device):
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device in ('CPU', 'Ascend'):
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f"Unsupported target device '{target_device}'. Expected one of: 'CPU', 'GPU', 'Ascend'")
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


def run_eval():
    session, input_name = create_session(args.onnx_path, args.device_target)

    imagenet_dataset = create_dataset_ImageNet(dataset_path=args.dataset_path,
                                               do_train=False,
                                               repeat_num=1,
                                               batch_size=config.batch_size,
                                               target=args.device_target)
    metrics = {
        'top-1 accuracy': nn.Top1CategoricalAccuracy(),
        'top-5 accuracy': nn.Top5CategoricalAccuracy(),
    }

    for batch in imagenet_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        y_pred = session.run(None, {input_name: batch['image']})[0]
        for metric in metrics.values():
            metric.update(y_pred, batch['label'])

    return {name: metric.eval() for name, metric in metrics.items()}


if __name__ == '__main__':
    results = run_eval()

    for name, value in results.items():
        print(f'{name}: {value:.4f}')
