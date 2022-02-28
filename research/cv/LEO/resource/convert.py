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
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor


parser = argparse.ArgumentParser()
parser.add_argument('--torch_init', type=str, default='torch_init.pth')
args = parser.parse_args()

def pytorch2mindspore():
    par_dict = torch.load('args.torch_init', map_location=torch.device('cpu'))
    new_params_list = []

    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]

        print('========================py_name', name)
        if name == 'inner_l_rate':
            name = 'leo.inner_lr'
        elif name == 'finetuning_lr':
            name = 'leo.finetuning_lr'
        elif name == 'encoder.weight':
            name = 'leo.encoder.encoder.weight'
        elif name == 'relation_net.0.weight':
            name = 'leo.encoder.relation_network.0.weight'
        elif name == 'relation_net.2.weight':
            name = 'leo.encoder.relation_network.2.weight'
        elif name == 'relation_net.4.weight':
            name = 'leo.encoder.relation_network.4.weight'
        elif name == 'decoder.weight':
            name = 'leo.decoder.decoder.weight'
        print('========================ms_name', name)

        param_dict['name'] = name
        param_dict['data'] = Tensor(parameter.numpy())
        new_params_list.append(param_dict)

    save_checkpoint(new_params_list, 'leo_ms_init.ckpt')


if __name__ == '__main__':
    pytorch2mindspore()
