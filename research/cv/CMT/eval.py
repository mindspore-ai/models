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
eval.
"""
import os
import argparse
from mindspore import context
from mindspore import nn
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.dataset import create_dataset
from src.cmt import cmt_s

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--platform', type=str, default='Ascend', help='run platform')
parser.add_argument('--model', type=str, default='cmt', help='eval model')
args_opt = parser.parse_args()


if __name__ == '__main__':
    config_platform = None
    if args_opt.platform == "Ascend":
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                            device_id=device_id, save_graphs=False)
    elif args_opt.platform == "GPU":
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=args_opt.platform, save_graphs=False)
    else:
        raise ValueError("Unsupported platform.")

    if args_opt.model == 'cmt':
        net = cmt_s()
    else:
        raise ValueError("Unsupported model.")

    if args_opt.checkpoint_path:
        param_dict = load_checkpoint(args_opt.checkpoint_path)
        load_param_into_net(net, param_dict)
    net.set_train(False)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    dataset = create_dataset(args_opt.dataset_path, do_train=False, batch_size=128)

    model = Model(net, loss_fn=loss, metrics={'acc'})
    res = model.eval(dataset, dataset_sink_mode=False)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)
