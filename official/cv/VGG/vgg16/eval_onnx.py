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

import mindspore.nn as nn
import onnxruntime as ort

from model_utils.config import get_config
from src.dataset import classification_dataset, vgg_create_dataset


def create_session(checkpoint_path, target_device):
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {target_device}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


def run_eval(checkpoint_path, dataset_name, data_dir, batch_size, image_size, target_device):
    session, input_name = create_session(checkpoint_path, target_device)

    if dataset_name == 'cifar10':
        dataset = vgg_create_dataset(data_dir, image_size, batch_size, training=False)
        metrics = {
            'accuracy': nn.Accuracy(),
        }
    elif dataset_name == 'imagenet2012':
        dataset = classification_dataset(data_dir, image_size, batch_size, mode='eval')
        metrics = {
            'top-1 accuracy': nn.Top1CategoricalAccuracy(),
            'top-5 accuracy': nn.Top5CategoricalAccuracy(),
        }
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    for batch in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        y_pred = session.run(None, {input_name: batch['image']})[0]
        for metric in metrics.values():
            metric.update(y_pred, batch['label'])

    return {name: metric.eval() for name, metric in metrics.items()}


if __name__ == '__main__':
    config = get_config()

    results = run_eval(config.file_name, config.dataset, config.data_dir,
                       config.batch_size, config.image_size, config.device_target)

    for name, value in results.items():
        print(f'{name}: {value:.4f}')
