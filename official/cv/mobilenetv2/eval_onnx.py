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

import os

import mindspore.nn as nn
import onnxruntime as ort

from src.model_utils.config import config
from src.dataset import create_dataset


def create_session(checkpoint_path, target_device):
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f'Unsupported target device {target_device!r}. Expected one of: "CPU", "GPU"')
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


def run_eval(cfg):
    session, input_name = create_session(cfg.file_name, cfg.platform)
    dataset = create_dataset(dataset_path=cfg.dataset_path, do_train=False, config=cfg)

    accuracy = nn.Accuracy()

    for batch in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        y_pred = session.run(None, {input_name: batch['image']})[0]
        accuracy.update(y_pred, batch['label'])

    return {
        'accuracy': accuracy.eval(),
    }


if __name__ == '__main__':
    config.batch_size = config.batch_size_export
    config.dataset_path = os.path.join(config.dataset_path, 'validation_preprocess')

    results = run_eval(config)

    for name, value in results.items():
        print(f'{name}: {value:.4f}')
