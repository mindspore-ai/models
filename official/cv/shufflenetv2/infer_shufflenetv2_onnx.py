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

"""test ShuffleNetV2"""
import argparse
import numpy as np
import onnxruntime
import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from src.dataset import create_dataset


def test(onnx_path, onnx_dataset_path, device_target, device_id):
    print(device_target)
    if device_target == 'GPU':
        providers = [['CUDAExecutionProvider'], [{'device_id': device_id}]]
    elif device_target == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {device_target}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = onnxruntime.InferenceSession(
        onnx_path, provider_options=providers)

    # create dataset
    dataset = create_dataset(
        onnx_dataset_path, do_train=False, rank=device_id, group_size=1, batch_size=1)
    cnt = 0
    correct_top1 = 0
    correct_top5 = 0
    topk = ops.TopK(sorted=True)
    k = 5
    for data in dataset:
        images_np = data[0]
        labels = data[1]
        cnt = cnt + 1
        print("Inferring image ID", cnt)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shufflentnetv2 ONNX Infer')
    parser.add_argument('--onnx_path', type=str, default='',
                        help='the path of ShuffleNetV2 ONNX (Default: None)')
    parser.add_argument('--onnx_dataset_path', type=str,
                        default='../imagenet', help='Dataset path')
    parser.add_argument('--platform', type=str, default='GPU', choices=('CPU', 'GPU'),
                        help='run platform(Default:GPU)')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id(Default:0)')
    args_opt = parser.parse_args()

    test(args_opt.onnx_path, args_opt.onnx_dataset_path,
         args_opt.platform, args_opt.device_id)
