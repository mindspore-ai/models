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
##############export checkpoint file into air, mindir models#################
python export.py
"""
import numpy as np
import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from nets.predrnn_pp import PreRNN

from config import config

def run_export():

    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=config.device_id)

    num_hidden = [int(x) for x in config.num_hidden.split(',')]
    num_layers = len(num_hidden)

    shape = [config.batch_size,
             config.seq_length,
             config.patch_size*config.patch_size*config.img_channel,
             int(config.img_width/config.patch_size),
             int(config.img_width/config.patch_size)]

    shape = list(map(int, shape))

    network = PreRNN(input_shape=shape,
                     num_layers=num_layers,
                     num_hidden=num_hidden,
                     filter_size=config.filter_size,
                     stride=config.stride,
                     seq_length=config.seq_length,
                     input_length=config.input_length,
                     tln=config.layer_norm)

    param_dict = load_checkpoint(config.pretrained_model)
    load_param_into_net(network, param_dict)
    network.set_train(False)
    patched_width = int(config.img_width/config.patch_size)

    mask_true = np.zeros((config.batch_size,
                          config.seq_length-config.input_length-1,
                          patched_width,
                          patched_width,
                          int(config.patch_size)**2*int(config.img_channel)))
    input_arr = (Tensor(np.ones(shape), dtype=ms.float32), Tensor(mask_true, dtype=ms.float32))

    export(network, *input_arr, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    run_export()
