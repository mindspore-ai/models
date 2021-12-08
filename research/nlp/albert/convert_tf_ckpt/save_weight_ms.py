# Copyright 2021 Huawei Technologies Co., Ltd
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
Save weight using mindspore, to load the parameters of albert model from npy file.
npy files should be in the same path with this script. Otherwise you should change the path name of the script.
"""

import os
import argparse
from mindspore import Tensor, ops
from mindspore.train.serialization import save_checkpoint
import numpy as np
import trans_dict


param_dict_list = ['albert.albert.albert_encoder.embedding_hidden_mapping_in.weight',
                   'albert.albert.albert_encoder.group.0.inner_group.0.intermediate.weight',
                   'albert.albert.albert_encoder.group.0.inner_group.0.output.dense.weight',
                   'albert.albert.albert_encoder.group.0.inner_group.0.attention.output.dense.weight',
                   'albert.albert.albert_encoder.group.0.inner_group.0.attention.attention.key_layer.weight',
                   'albert.albert.albert_encoder.group.0.inner_group.0.attention.attention.query_layer.weight',
                   'albert.albert.albert_encoder.group.0.inner_group.0.attention.attention.value_layer.weight',
                   'albert.albert.dense.weight', 'albert.cls1.dense.weight', 'albert.cls2.dense.weight']


def trans_model_parameter(ckpt_name, load_dir):
    """
    transform model parameters
    """
    file_names = [name for name in os.listdir(load_dir) if name.endswith(".npy")]
    # to find all file names with suffix '.npy' in the current path.
    new_params_list = []
    for file_name in file_names:
        var_name = file_name[:-4]
        param_dict = {"name": var_name, "data": Tensor(np.load(os.path.join(load_dir, file_name)))}
        if var_name in trans_dict.trans_dict_tf.values():
            if param_dict['name'] in param_dict_list:
                new_param = param_dict['data']
                transpose = ops.Transpose()
                perm = (1, 0)
                output = transpose(new_param, perm)
                param_dict["data"] = output
            new_params_list.append(param_dict)
            print(var_name+" has been saved")

    save_checkpoint(new_params_list, ckpt_name)
    # to load the parameters from npy files and save them as mindspore checkpoint
    print("Finished:the parameters have been saved into mindspore checkpoint.")


def main():
    parser = argparse.ArgumentParser(description="Read TensorFlow Albert model checkpoint weight")
    parser.add_argument("--load_dir", type=str, default="",
                        help="The name of input numpy file directory")
    parser.add_argument("--output_file_name", type=str, default="",
                        help="The name of output checkpoint name")
    args_opt = parser.parse_args()
    ckpt_name = args_opt.output_file_name
    load_dir = args_opt.load_dir
    trans_model_parameter(ckpt_name=ckpt_name, load_dir=load_dir)


if __name__ == "__main__":
    main()
