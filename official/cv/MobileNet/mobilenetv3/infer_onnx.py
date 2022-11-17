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

"""test Mobilenetv3"""

import argparse
import onnxruntime
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from src.dataset import create_dataset
from src.dataset import create_dataset_cifar
from src.config import config_gpu
from src.config import config_cpu

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--onnx_path', type=str,
                    default=None, help='ONNX file path')
parser.add_argument('--dataset_path', type=str,
                    default=None, help='Dataset path')
parser.add_argument('--device_target', type=str,
                    default="GPU", help='run device_target')
args_opt = parser.parse_args()

if __name__ == '__main__':
    config = None
    if args_opt.device_target == "GPU":
        config = config_gpu
        providers = ['CUDAExecutionProvider']
        dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=False,
                                 config=config, device_target=args_opt.device_target, batch_size=1)
    elif args_opt.device_target == "CPU":
        providers = ['CPUExecutionProvider']
        config = config_cpu
        dataset = create_dataset_cifar(
            dataset_path=args_opt.dataset_path, do_train=False, batch_size=1)
    else:
        raise ValueError("Unsupported device_target.")
    session = onnxruntime.InferenceSession(
        args_opt.onnx_path, providers=providers)
    cnt = 0
    correct_top1 = 0
    correct_top5 = 0
    topk = ops.TopK(sorted=True)
    k = 5
    for data in dataset:
        images_np = data[0]
        labels = data[1]
        cnt = cnt + 1
        inputs = {session.get_inputs()[0].name: images_np.asnumpy()}
        model_predict = session.run(None, inputs)
        model_predict = np.expand_dims(np.squeeze(model_predict), axis=0)

        input_x = Tensor(model_predict[0], ms.float16)
        _, k_label = topk(input_x, k)
        if k_label[0] == labels:
            correct_top1 = correct_top1 + 1
        if labels in k_label:
            correct_top5 = correct_top5 + 1
    print("Inferred the num of images = " + str(cnt))
    print("ACC_TOP1 = " + str((1.0 * correct_top1 / cnt)))
    print("ACC_TOP5 = " + str((1.0 * correct_top5 / cnt)))
