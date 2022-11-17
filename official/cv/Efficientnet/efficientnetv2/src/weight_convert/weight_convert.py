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
"""weight convert"""
import argparse
import os

import numpy as np
import tensorflow as tf
from mindspore import Tensor, save_checkpoint, dtype


def parse_arguments():
    """parse_arguments"""
    parser = argparse.ArgumentParser(description="MindSpore Tensorflow weight transfer")
    parser.add_argument("--pretrained", default=None, type=str)
    parser.add_argument("--name", default="imagenet22k", choices=["imagenet22k",])
    args = parser.parse_args()
    return args


def get_mindspore_key():
    """get mindspore key for efficientnetv2"""
    with open("key_ms.txt", 'r') as f:
        keys_mindspore = f.readlines()
    keys_mindspore = [key.split("\n")[0] for key in keys_mindspore]
    print(f"params num: {len(keys_mindspore)}")
    keys_mindspore_top = []
    keys_mindspore_middle = []
    keys_mindspore_tail = []
    for key in keys_mindspore:
        if "features.0" in key or "features" not in key:
            keys_mindspore_tail.append(key)
        elif "moving_variance" in key or "moving_mean" in key:
            keys_mindspore_middle.append(key)
        else:
            keys_mindspore_top.append(key)
    keys_mindspore_all = keys_mindspore_top + keys_mindspore_middle + keys_mindspore_tail
    keys_mindspore = [key[:key.find("(") - 1] for key in keys_mindspore_all]
    return keys_mindspore


def main():
    args = parse_arguments()
    keys_mindspore = get_mindspore_key()
    reader = tf.compat.v1.train.NewCheckpointReader(args.pretrained)
    all_variables = reader.get_variable_to_shape_map()
    all_variables = list(all_variables.keys())
    all_variables.remove("global_step")
    with open("key_tf.txt", 'r') as f:
        keys_tensorflow = f.readlines()
        keys_tensorflow = [key.split("\n")[0] for key in keys_tensorflow]
        keys_tensorflow_ema = [f"{key}/ExponentialMovingAverage" for key in keys_tensorflow]
        ema_params_list = []
        for key_ms, key_tf in zip(keys_mindspore, keys_tensorflow_ema):
            params_dict = {}
            weight = np.array(reader.get_tensor(key_tf))
            if len(weight.shape) == 4:
                weight = np.transpose(weight, (3, 2, 0, 1))
                if weight.shape[0] == 1:
                    weight = np.transpose(weight, (1, 0, 2, 3))
            if "classifier" in key_ms and len(weight.shape) == 2:
                weight = np.transpose(weight, (1, 0))
            params_dict["data"] = Tensor(weight, dtype=dtype.float32)
            params_dict["name"] = key_ms

            ema_params_list.append(params_dict)
        ema_path = f"ema_efficientnets_{args.name}.ckpt"
        if os.path.exists(ema_path):
            os.remove(ema_path)
        save_checkpoint(ema_params_list, ema_path)

        params_list = []
        for key_ms, key_tf in zip(keys_mindspore, keys_tensorflow):
            params_dict = {}
            weight = np.array(reader.get_tensor(key_tf))
            if len(weight.shape) == 4:
                weight = np.transpose(weight, (3, 2, 0, 1))
                if weight.shape[0] == 1:
                    weight = np.transpose(weight, (1, 0, 2, 3))
            if "classifier" in key_ms and len(weight.shape) == 2:
                weight = np.transpose(weight, (1, 0))
            params_dict["data"] = Tensor(weight, dtype=dtype.float32)
            params_dict["name"] = key_ms
            params_list.append(params_dict)
        path = f"efficientnets_{args.name}.ckpt"
        if os.path.exists(path):
            os.remove(path)
        save_checkpoint(params_list, path)


if __name__ == '__main__':
    main()
