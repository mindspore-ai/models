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
"""export checkpoint file into mindir models"""

import os
import argparse
import numpy as np
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net
from mindspore.train.serialization import export, Tensor, context

from src.vgg import CifarVGG


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distillation')
    parser.add_argument('--device', type=str, default='Ascend', help='device type')
    parser.add_argument('--device_id', type=int, default=0, help='device id')
    parser.add_argument('--n_classes', type=int, default=100, help='number of classes')
    parser.add_argument('--t_net_name', type=str, default='ResNet14', help='teacher net name')
    parser.add_argument('--net_name', type=str, default='VGG8', help='student net name')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpts/', help='checkpoint path')
    parser.add_argument('--ckpt_name', type=str, default='cifar100-ResNet14.ckpt', help='checkpoint fname')

    myargs = parser.parse_args()
    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target=myargs.device,
        device_id=myargs.device_id
    )

    t_net_name = myargs.t_net_name
    net_name = myargs.net_name

    # load model
    net_name = myargs.net_name
    if net_name == "VGG8":
        model = CifarVGG(
            vgg_n_layer=8,
            n_classes=myargs.n_classes,
        )
    else:
        raise ValueError("No such net: {}".format(net_name))

    # load checkpoint
    fpath = os.path.join(
        myargs.ckpt_dir,
        myargs.ckpt_name
    )
    param_dict = load_checkpoint(ckpt_file_name=fpath)

    load_param_into_net(model, param_dict)
    model.set_train(False)

    feature = Tensor(np.zeros([1, 3, 32, 32]).astype(np.float32))
    fpath = os.path.join(
        myargs.ckpt_dir, "{}_mindir".format(myargs.ckpt_name.split(".")[0])
    )
    export(
        model, feature,
        file_name=fpath, file_format='MINDIR'
    )
