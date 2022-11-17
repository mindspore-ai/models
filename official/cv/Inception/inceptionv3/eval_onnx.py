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
"""evaluate_imagenet_onnx"""

import onnxruntime as ort
import mindspore.nn as nn
from src.model_utils.config import config
from src.dataset import create_dataset_imagenet, create_dataset_cifar10

DS_DICT = {
    "imagenet": create_dataset_imagenet,
    "cifar10": create_dataset_cifar10,
}


def create_session(onnx_checkpoint_path, target_device):
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {target_device}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = ort.InferenceSession(onnx_checkpoint_path, providers=providers)

    input_name = session.get_inputs()[0].name
    return session, input_name


def eval_inceptionv3():
    session, input_name = create_session(config.onnx_file, config.platform)

    create_dataset = DS_DICT[config.ds_type]
    config.rank = 0
    config.group_size = 1
    config.batch_size = 1
    dataset = create_dataset(config.dataset_path, False, config)

    eval_metrics = {'Top1-Acc': nn.Top1CategoricalAccuracy(),
                    'Top5-Acc': nn.Top5CategoricalAccuracy()}

    for batch in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        y_pred = session.run(None, {input_name: batch['image']})[0]
        for metric in eval_metrics.values():
            metric.update(y_pred, batch['label'])

    return {name: metric.eval() for name, metric in eval_metrics.items()}


if __name__ == '__main__':
    results = eval_inceptionv3()
    for name, value in results.items():
        print(f'{name}: {value:.7f}')
