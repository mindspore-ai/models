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
python eval_onnx.py
"""
from mindspore.nn.metrics import Accuracy
import onnxruntime as ort

from model_utils.config import config
from src.dataset import MovieReview, SST2, Subjectivity


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


def eval_net():
    '''eval net'''
    if config.dataset == 'MR':
        instance = MovieReview(root_dir=config.data_path, maxlen=config.word_len, split=0.9)
    elif config.dataset == 'SUBJ':
        instance = Subjectivity(root_dir=config.data_path, maxlen=config.word_len, split=0.9)
    elif config.dataset == 'SST2':
        instance = SST2(root_dir=config.data_path, maxlen=config.word_len, split=0.9)

    session, input_name = create_session(config.onnx_file, config.device_target)

    dataset = instance.create_test_dataset(batch_size=config.batch_size)

    eval_metrics = {'acc': Accuracy()}

    for batch in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        y_pred = session.run(None, {input_name: batch['data']})[0]
        for metric in eval_metrics.values():
            metric.update(y_pred, batch['label'])

    return {name: metric.eval() for name, metric in eval_metrics.items()}


if __name__ == '__main__':
    results = eval_net()
    for name, value in results.items():
        print(f'{name}: {value:.16f}')
