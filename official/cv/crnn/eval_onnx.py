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
"""ONNX evaluation script"""
import onnxruntime as ort
from mindspore.common import set_seed

from src.dataset import create_dataset
from src.metric import CRNNAccuracy
from src.model_utils.config import config


def create_session(checkpoint_path, target_device):
    """Create onnxruntime session"""
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device in ('CPU', 'Ascend'):
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f"Unsupported target device '{target_device}'. Expected one of: 'CPU', 'GPU', 'Ascend'")
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


def crnn_eval():
    """CRNN evaluation"""
    set_seed(1)
    config.batch_size = 1
    # create dataset
    dataset = create_dataset(name=config.eval_dataset,
                             dataset_path=config.eval_dataset_path,
                             batch_size=config.batch_size,
                             is_training=False,
                             config=config)
    # load checkpoint
    session, input_name = create_session(config.file_name, config.device_target)
    # start evaluation
    metrics = CRNNAccuracy(config)
    for batch in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        y_pred = session.run(None, {input_name: batch['image']})[0]
        metrics.update(y_pred, batch['label'])
    res = metrics.eval()
    print("result:", res, flush=True)


if __name__ == '__main__':
    crnn_eval()
