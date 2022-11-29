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
"""onnx infer"""
import os
import numpy as np
import onnxruntime as ort
from mindspore.common import set_seed

from src.config import config
from src.load_dataset import load_dataset


def create_session(checkpoint_path, device_target):
    """create onnxruntime session"""
    if device_target == "GPU":
        providers = ['CUDAExecutionProvider']
    elif device_target in ['CPU', 'Ascend']:
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f"Unsupported device_target '{device_target}'. Expected one of: 'CPU', 'GPU', 'Ascend'")
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_name = []
    for in_put in session.get_inputs():
        input_name.append(in_put.name)
    return session, input_name


def run_onnx_eval():
    """run_onnx_eval"""

    data_menu = config.data_url
    eval_dataset = os.path.join(data_menu, 'test.mindrecord')

    dataset = load_dataset(input_files=eval_dataset,
                           batch_size=config.batch_size)

    set_seed(config.rseed)

    correct = 0
    count = 0
    session, input_name = create_session(config.onnx_ckpt, config.device)
    for batch in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        content = batch['content']
        sen_len = batch['sen_len']
        aspect = batch['aspect']
        in_put = [content, sen_len, aspect]
        solution = batch['solution']

        pred = session.run(None, {input_name[i]: in_put[i] for i in range(len(input_name))})[0]

        polarity_pred = np.argmax(pred, axis=1)
        polarity_label = np.argmax(solution, axis=1)

        correct += (polarity_pred == polarity_label).sum()
        count += len(polarity_label)

    acc = correct / count
    print("\n---accuracy:", acc, "---\n")


if __name__ == "__main__":
    run_onnx_eval()
