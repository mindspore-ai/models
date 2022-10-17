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

from mindspore import context
from mindspore import nn
from mindspore.common import set_seed

from src.args import args
from src.tools.get_misc import get_dataset, set_device

set_seed(args.seed)

def main():
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    if args.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True)
    set_device(args)

    if args.device_target == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif args.device_target == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {args.device_target}, '
            f'Expected: "GPU"'
        )
    import onnxruntime
    model = onnxruntime.InferenceSession('./DDRNet23.onnx', providers)
    input_name = model.get_inputs()[0].name
    print(input_name)
    eval_metrics = {'Loss': nn.Loss(),
                    'Top1-Acc': nn.Top1CategoricalAccuracy(),
                    'Top5-Acc': nn.Top5CategoricalAccuracy()}
    print(onnxruntime.get_device(), flush=True)
    data = get_dataset(args, training=False)
    eval_metrics = {'Top1-Acc': nn.Top1CategoricalAccuracy(),
                    'Top5-Acc': nn.Top5CategoricalAccuracy()}
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
