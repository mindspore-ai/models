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

import os
import datetime
import onnxruntime
import numpy as np
from src.dataset import classification_dataset
from src.model_utils.config import config


def run_eval():
    config.load_type = 'test'
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
    session = onnxruntime.InferenceSession(config.onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    # logger
    config.outputs_dir = os.path.join(config.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    dataset, _ = classification_dataset(config.batch_size, 1, shuffle=True, repeat_num=1,
                                        drop_remainder=True)

    batch_num = dataset.get_dataset_size()
    acc_sum, sample_num = 0, 0
    for index, (input_data, label) in enumerate(dataset):
        predictions = session.run(None, {input_name: input_data.asnumpy()})
        predictions = np.squeeze(predictions)
        label = label.asnumpy()
        if label.shape[0] > 1:
            acc = np.sum(np.argmax(predictions, 1) == label[:, -1])
        else:
            acc = np.sum(np.argmax(predictions) == label[:, -1])
        batch_size = label.shape[0]
        acc_sum += acc
        sample_num += batch_size
        if index % 20 == 0:
            print("setep: {}/{}, acc: {}".format(index + 1, batch_num, acc / batch_size))

    accuracy_top1 = acc_sum / sample_num
    print('eval result: top_1 {:.3f}%'.format(accuracy_top1 * 100))


if __name__ == '__main__':
    run_eval()
