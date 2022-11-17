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
""" STPM Network """

import mindspore.nn as nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.resnet import resnet18


def load_ckpt_to_net(network, args):
    print(f">>>>>>>start load {args.pre_ckpt_path}", flush=True)
    param = load_checkpoint(args.pre_ckpt_path)
    new_parm = {}
    for key, val in param.items():
        # trans modelzoo resnet18 to backbone
        if "bn1d" in key:
            key = key.replace("bn1d", "bn1")
        if "bn2d" in key:
            key = key.replace("bn2d", "bn2")
        if "end_point" in key:
            key = key.replace("end_point", "fc")
        key = "model_t." + key
        new_parm[key] = val
    load_param_into_net(network, new_parm)
    print(f">>>>>>>load {args.pre_ckpt_path} success", flush=True)
    return network


class STPM(nn.Cell):
    """ STPM Network """
    def __init__(self, args, is_train=True, finetune=False):
        super(STPM, self).__init__()
        use_batch_statistics = False if is_train else None
        self.model_t = resnet18(args.num_class, use_batch_statistics=use_batch_statistics)
        if is_train and not finetune:
            self.model_t = load_ckpt_to_net(self.model_t, args)
        self.model_s = resnet18(args.num_class)

    def construct(self, x):
        features_s = self.model_s(x)
        features_t = self.model_t(x)
        return features_s, features_t
