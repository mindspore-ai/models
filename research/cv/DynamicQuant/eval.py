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
import argparse

from mindspore import context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.resnet import resnet18
from src.dataset import create_dataset_val
from src.loss import CrossEntropySmooth

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--checkpoint_path', type=str, default="./res18.ckpt", help='Checkpoint file path')
parser.add_argument('--dataset_path', type=str, default="../../data/imagenet/val/", help='Dataset path')
parser.add_argument('--device_target', type=str, default="GPU", help='Run device target')
parser.add_argument('--smoothing', type=float, default=0.1, help='label smoothing (default: 0.1)')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N', help='number of label classes (default: 10)')
args = parser.parse_args()

if __name__ == '__main__':
    config_device_target = None
    device_id = int(os.getenv('DEVICE_ID', '0'))
    if args.device_target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                            device_id=device_id, save_graphs=False)
    elif args.device_target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU",
                            device_id=device_id, save_graphs=False)
    else:
        raise ValueError("Unsupported device target: {}.".format(args.device_target))


    network = resnet18()
    # load pre-trained quant ckpt
    if args.checkpoint_path:
        param_dict = load_checkpoint(args.checkpoint_path)
        not_load_param, _ = load_param_into_net(network, param_dict)
        if not_load_param:
            raise ValueError("Load param into net fail!")

    network.set_train(False)

    # define model
    loss = CrossEntropySmooth(smooth_factor=args.smoothing, num_classes=args.num_classes)
    model = Model(network, loss_fn=loss, metrics={'acc'})

    # define dataset
    dataset = create_dataset_val(val_data_url=args.dataset_path)

    print("============== Starting Validation ==============")
    res = model.eval(dataset)
    print("result:", res, "ckpt=", args.checkpoint_path)
    print("============== End Validation ==============")
