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
"""Run evaluation for a model exported to ONNX"""
import argparse
import mindspore.nn as nn
import onnxruntime as ort

parser = argparse.ArgumentParser(description='Eval Onnx')
parser.add_argument('--device_target', type=str, default='GPU', help='Device target')
parser.add_argument('--dataset_path', type=str, default='', help='Dataset path')
parser.add_argument('--onnx_path', type=str, default='', help='onnx file path')
parser.add_argument('--net_name', type=str,
                    choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "seresnet50"],
                    default='', help='The name of net')
parser.add_argument('--dataset', type=str, choices=["imagenet2012", "cifar10"],
                    default='', help='The name of dataset')
args = parser.parse_args()

if args.net_name in ("resnet18", "resnet34", "resnet50", "resnet152"):
    if args.dataset == "cifar10":
        from src.dataset import create_dataset1 as create_dataset
    else:
        from src.dataset import create_dataset2 as create_dataset
elif args.net_name == "resnet101":
    from src.dataset import create_dataset3 as create_dataset
else:
    from src.dataset import create_dataset4 as create_dataset

def create_session(checkpoint_path, target_device):
    """create session"""

    # set environment
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device in ('CPU', 'Ascend'):
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(f"Unsupported target device '{target_device}'. Expected one of: 'CPU', 'GPU', 'Ascend'")

    # create ONNX session
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name

def run_eval():
    # create session
    session, input_name = create_session(args.onnx_path, args.device_target)

    # create dataset
    if args.net_name in ("resnet18", "resnet34", "resnet50"):
        if args.dataset == "imagenet2012":
            batch_size = 256
            eval_image_size = 224
        else:
            batch_size = 32
            eval_image_size = 224
    elif args.net_name in ("resnet101", "resnet152"):
        batch_size = 32
        eval_image_size = 224
    else:
        batch_size = 32
        eval_image_size = 256

    dataset = create_dataset(dataset_path=args.dataset_path,
                             do_train=False,
                             batch_size=batch_size,
                             eval_image_size=eval_image_size,
                             target=args.device_target)

    # define metric
    metrics = {
        'top-1 accuracy': nn.Top1CategoricalAccuracy(),
        'top-5 accuracy': nn.Top5CategoricalAccuracy(),
    }

    # eval
    for batch in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        y_pred = session.run(None, {input_name: batch['image']})[0]
        for metric in metrics.values():
            metric.update(y_pred, batch['label'])

    # return result
    return {name: metric.eval() for name, metric in metrics.items()}


if __name__ == '__main__':
    results = run_eval()
    print("the result of %s_%s:"%(args.net_name, args.dataset))
    for name, value in results.items():
        print(f'{name}: {value:.4f}')
