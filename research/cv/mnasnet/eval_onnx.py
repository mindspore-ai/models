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

import argparse
import mindspore.nn as nn
from mindspore.common import set_seed
from src.dataset import create_dataset
import onnxruntime as ort


set_seed(1)

def create_session(checkpoint_path, target_device):
    '''
    create onnx session
    '''
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device in ('CPU', 'Ascend'):
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f'Unsupported target device {target_device!r}. Expected one of: "CPU", "GPU","Ascend"')
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_name = session.get_inputs()[0].name

    return session, input_name




def main():
    parser = argparse.ArgumentParser(description='Image classification by onnx model')

    # modelarts parameter
    parser.add_argument('--dataset_path', type=str, default='/home/sda/ILSVRC2012/val', help='Dataset path')

    # device target
    parser.add_argument("--device_target", type=str, choices=["GPU", "CPU", "Ascend"], default="GPU",
                        help="device target")

    parser.add_argument('--onnx_url', default='./MNasNet.onnx', type=str, help='onnx path')

    args_opt = parser.parse_args()


    session, input_name = create_session(args_opt.onnx_url, args_opt.device_target)

    # create dataset
    dataset = create_dataset(dataset_path=args_opt.dataset_path,
                             do_train=False,
                             batch_size=1)

    eval_metrics = {'Top1-Acc': nn.Top1CategoricalAccuracy(),
                    'Top5-Acc': nn.Top5CategoricalAccuracy()}
    for batch in dataset.create_dict_iterator(num_epochs=256, output_numpy=True):
        y_pred = session.run(None, {input_name: batch['image']})[0]
        for metric in eval_metrics.values():
            metric.update(y_pred, batch['label'])

    return {name: metric.eval() for name, metric in eval_metrics.items()}



if __name__ == '__main__':
    results = main()
    for name, value in results.items():
        print(f'{name}: {value:.4f}')
