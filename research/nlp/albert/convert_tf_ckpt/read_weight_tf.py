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
Read weight using tensorflow
to read the parameters of the Albert pretrained model from tensorflow checkpoint
and save them into npy files for mindspore to load.

*this script is based on the gpt-2 model downloaded from openai.*
"""

import os
import argparse
import numpy as np
import tensorflow as tf

import trans_dict


def read_weight(ckpt_path, output_path):
    """
    read weight
    Args:
        ckpt_path: Any
    """
    # model path and model name
    init_vars = tf.train.list_variables(ckpt_path)
    # load the model parameters into vars
    save_param_num = 0

    for name, _ in init_vars:
        array = tf.train.load_variable(ckpt_path, name)
        # By this you can understand the next step easily
        name = name.replace(r"/", ".")
        print('name: ', name)
        # skip 'model/' and change var names to avoid path mistake
        if name not in trans_dict.trans_dict_tf.keys():
            print(name + " is not in this model")
        else:
            np.save(os.path.join(output_path, trans_dict.trans_dict_tf[name] + ".npy"), array)
            save_param_num = save_param_num + 1

    print("finished!")
    print("save {num} parameters.".format(num=save_param_num))


def main():
    parser = argparse.ArgumentParser(description="Read TensorFlow Albert model checkpoint weight")
    parser.add_argument("--ckpt_file_path", type=str, default="",
                        help="The TensorFlow Albert model checkpoint file path")
    parser.add_argument("--output_path", type=str, default="",
                        help="The name of output checkpoint path")
    args_opt = parser.parse_args()
    ckpt_path = args_opt.ckpt_file_path
    output_path = args_opt.output_path
    read_weight(ckpt_path=ckpt_path, output_path=output_path)


if __name__ == "__main__":
    main()
