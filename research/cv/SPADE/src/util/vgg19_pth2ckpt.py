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
""" Vgg19 ckpt from Pytorch to MindSpore """

import torchvision
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor

def main():
    vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
    print(vgg_pretrained_features)
    par_dict = vgg_pretrained_features.state_dict()
    new_params_list = []
    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
        print('name:  ', name)

        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        new_params_list.append(param_dict)

    save_checkpoint(new_params_list, './vgg/vgg19.ckpt')

if __name__ == '__main__':
    main()
