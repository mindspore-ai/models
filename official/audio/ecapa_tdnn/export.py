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
import os
import sys
from hyperpyyaml import load_hyperpyyaml
import numpy as np
import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context
from src.ecapa_tdnn import ECAPA_TDNN

def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)

def run_export(hparams):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    in_channels = hparams["in_channels"]
    channels = hparams["channels"]
    emb_size = hparams["emb_size"]
    net = ECAPA_TDNN(in_channels, channels=[channels, channels, channels, channels, channels * 3],
                     lin_neurons=emb_size, global_context=False)

    # assert config.ckpt_file is not None, "config.ckpt_file is None."
    model_path = hparams["exp_ckpt_file"]
    param_dict = load_checkpoint(model_path)
    load_param_into_net(net, param_dict)
    file_name = hparams["file_name"]
    file_format = hparams["file_format"]
    input_arr = Tensor(np.ones([1, hparams["length"], hparams["channel"]]), ms.float32)
    export(net, input_arr, file_name=file_name, file_format=file_format)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        hparams_file = sys.argv[1]
    else:
        hparams_file = "ecapa-tdnn_config.yaml"
    print("hparam:", hparams_file)
    with open(hparams_file) as fin:
        params = load_hyperpyyaml(fin)
        print(params)
    run_export(params)
