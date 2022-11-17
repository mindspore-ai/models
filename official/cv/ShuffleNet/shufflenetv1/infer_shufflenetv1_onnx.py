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
"""test ShuffleNetV1"""
import onnxruntime
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from src.dataset import create_dataset
from src.model_utils.config import config


def test():
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

    # create dataset
    dataset = create_dataset(config.onnx_dataset_path, do_train=False, device_num=1, rank=0)

    cnt = 0
    correct_top1 = 0
    correct_top5 = 0
    topk = ops.TopK(sorted=True)
    k = 5
    for data in dataset:
        images_np = data[0]
        labels = data[1]
        inputs = {session.get_inputs()[0].name: images_np.asnumpy()}
        model_predict = session.run(None, inputs)
        model_predict = np.expand_dims(np.squeeze(model_predict), axis=0)
        for predict, label in zip(model_predict[0], labels):
            cnt = cnt + 1
            input_x = Tensor(predict, ms.float16)
            _, k_label = topk(input_x, k)
            if k_label[0] == label:
                correct_top1 = correct_top1 + 1
            if label in k_label:
                correct_top5 = correct_top5 + 1
        print("dealwith image num = " + str(cnt))
    print("correct_top1 = " + str((1.0 * correct_top1 / cnt)))
    print("correct_top5 = " + str((1.0 * correct_top5 / cnt)))


if __name__ == '__main__':
    test()
