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
"""export infer om"""
import subprocess

import numpy as np
from mindspore import load_checkpoint, load_param_into_net, Tensor
from mindspore.train.serialization import export

from src.FasterRcnnInfer.faster_rcnn_r50 import FasterRcnn_Infer
from src.config import FasterRcnnConfig


def init_fasterrcnn(config, ori_image_size):
    config.test_batch_size = 1
    model = FasterRcnn_Infer(config, ori_image_size)
    print('===================init fasterrcnn infer model=====================')
    return model


def ckpt2om(ckpt_path=None, ori_image_size=(4608, 3288), save_ext_model_param=({'soc_version': 'Ascend310'},)):
    config = FasterRcnnConfig()
    base_save_name = ckpt_path.replace('.ckpt', '') if ckpt_path else './fasterrcnn'
    model = init_fasterrcnn(config, ori_image_size)
    model.set_train(False)

    if ckpt_path:
        param_dict = load_checkpoint(ckpt_path)
        load_param_into_net(net=model, parameter_dict=param_dict, strict_load=True)

    input_size = [config.img_height, config.img_width]
    air_file = ckpt2air(model, input_size, base_save_name)

    # return
    om_files = air2om(air_file, save_ext_model_param, base_save_name)
    return om_files


def ckpt2air(model, input_size, base_save_name):
    air_file = base_save_name + '.air'
    input_data = np.ones([1, 3, *input_size]).astype(np.float32)
    export(model, Tensor(input_data), file_name=base_save_name, file_format='AIR')
    print('===================export fasterrcnn to air succeed=====================')
    return air_file


def air2om(air_file, save_ext_model_param, base_save_name):
    om_files = []
    for ext_params in save_ext_model_param:
        soc_version = ext_params.get('soc_version', 'Ascend310')
        inert_op_conf = ext_params.get('insert_op_conf', 'None')
        if len(save_ext_model_param) == 1:
            output_name = base_save_name
        else:
            output_name = '{}_{}'.format(base_save_name, soc_version)
        om_files.append(output_name)

        atc_command = ['atc', '--model={}'.format(air_file),
                       '--output={}'.format(output_name),
                       '--soc_version={}'.format(soc_version),
                       '--framework=1']
        if inert_op_conf != 'None':
            atc_command.append('--insert_op_conf={}'.format(inert_op_conf))
        print('atc command: ' + ' '.join(atc_command))
        process = subprocess.Popen(atc_command)
        process.wait()
    print('===================convert fasterrcnn to om succeed=====================')
    return om_files


if __name__ == '__main__':
    ckpt2om('/home/model/faster_rcnn-12_7393.ckpt')
