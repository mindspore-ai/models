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
"""eval"""

from mindspore import nn
from src.args import args
from src.tools.get_misc import get_dataset
import onnxruntime

def main():
    if args.device_target == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif args.device_target == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {args.device_target}, '
            f'Expected: "CPU", "GPU"'
        )

    model = onnxruntime.InferenceSession(args.pretrained, providers=providers)
    input_name = model.get_inputs()[0].name
    print(onnxruntime.get_device(), flush=True)
    data = get_dataset(args, training=False)
    eval_metrics = {'top-1 accuracy': nn.Top1CategoricalAccuracy(),
                    'top-5 accuracy': nn.Top5CategoricalAccuracy()}
    for batch in data.val_dataset:
        y_pred = model.run(None, {input_name: batch[0].asnumpy()})
        for metric in eval_metrics.values():
            metric.update(y_pred[0], batch[1].asnumpy())
    result = {name: metric.eval() for name, metric in eval_metrics.items()}
    print(f"=> begin eval")
    for name, value in result.items():
        print(name, value)
    print(f"=> eval results:{result}")
    print(f"=> eval success")

if __name__ == '__main__':
    main()
