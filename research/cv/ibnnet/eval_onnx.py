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
eval
"""

import argparse
import mindspore.nn as nn
import onnxruntime as ort

from src.dataset import create_evalset



def create_session(onnx_path, target_device):
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {target_device}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name

def onnx_eval(onnx_path, dataset_path, batch_size, device_target):
    # create dataset
    ds = create_evalset(dataset_path=dataset_path,
                        do_train=False,
                        repeat_num=1,
                        batch_size=batch_size,
                        target=device_target)
    # load onnx
    session, input_name = create_session(onnx_path=onnx_path,
                                         target_device=device_target)
    # evaluation
    metrics = {
        'top-1 accuracy': nn.Top1CategoricalAccuracy(),
        'top-5 accuracy': nn.Top5CategoricalAccuracy(),
    }
    for batch in ds.create_dict_iterator(num_epochs=1, output_numpy=True):
        y_pred = session.run(None, {input_name: batch['image']})[0]
        for metric in metrics.values():
            metric.update(y_pred, batch['label'])
    return {name: metric.eval() for name, metric in metrics.items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    # onnx parameter
    parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
    parser.add_argument('--onnx_path', type=str, default=None, help='ONNX file path')
    parser.add_argument('--device_target', type=str, default='GPU', help='Device target')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    args = parser.parse_args()

    results = onnx_eval(args.onnx_path, args.dataset_path, args.batch_size, args.device_target)
    for name, value in results.items():
        print(f'{name}: {value:.4f}')
