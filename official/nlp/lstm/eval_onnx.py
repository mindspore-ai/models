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
#################Inference On Onnx########################
"""
import onnxruntime as ort
import mindspore.nn as nn
from src.model_utils.config import config
from src.dataset import lstm_create_dataset

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

def eval_lstm():
    """ eval lstm on onnx"""
    print('\neval_onnx.py config: \n', config)
    session, input_name = create_session(config.onnx_file, config.onnx_target)
    dataset = lstm_create_dataset(config.preprocess_path, config.batch_size, training=False)
    eval_metrics = {'acc': nn.Accuracy(), 'recall': nn.Recall(), 'f1': nn.F1()}

    for batch in dataset:
        y_pred = session.run(None, {input_name: batch[0].asnumpy()})[0]
        for metric in eval_metrics.values():
            metric.update(y_pred, batch[1].asnumpy())
    return {name: metric.eval() for name, metric in eval_metrics.items()}

if __name__ == '__main__':
    result = eval_lstm()
    print("=================================Inference Result=================================")
    for name, value in result.items():
        print(name, value)
    print("=================================================================================")
