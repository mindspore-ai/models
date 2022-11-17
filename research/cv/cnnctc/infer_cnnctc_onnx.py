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


import onnxruntime
import numpy as np
from mindspore.dataset import GeneratorDataset
from src.dataset import IIIT_Generator_batch
from src.util import CTCLabelConverter
from src.model_utils.config import config


def run_eval():
    print(config.device_target)
    if config.device_target == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif config.device_target == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {config.device_target}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = onnxruntime.InferenceSession(config.onnx_path, provider_options=providers)
    ds = GeneratorDataset(IIIT_Generator_batch, ['img', 'label_indices', 'text', 'sequence_length', 'label_str'])
    converter = CTCLabelConverter(config.CHARACTER)

    count = 0
    correct_count = 0
    for data in ds.create_tuple_iterator():
        img_np, _, text, _, length = data

        inputs = {session.get_inputs()[0].name: img_np.asnumpy()}
        model_predict = session.run(None, inputs)
        model_predict = np.expand_dims(np.squeeze(model_predict), axis=0)

        preds_size = np.array([model_predict.shape[1]] * config.TEST_BATCH_SIZE)
        preds_index = np.argmax(model_predict, 2)
        preds_index = np.reshape(preds_index, [-1])
        preds_str = converter.decode(preds_index, preds_size)

        label_str = converter.reverse_encode(text.asnumpy(), length.asnumpy())

        print("Prediction samples: \n", preds_str[:5])
        print("Ground truth: \n", label_str[:5])
        for pred, label in zip(preds_str, label_str):
            if pred == label:
                correct_count += 1
            count += 1

    print('Onnx accuracy: ', correct_count / count)


if __name__ == '__main__':
    run_eval()
