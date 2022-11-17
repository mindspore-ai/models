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
"""HRNet conversion from torch."""
import argparse
import pickle

import torch
from mindspore import Parameter, load_param_into_net, save_checkpoint

from src.config import config_hrnetv2_w48 as config
from src.seg_hrnet import HighResolutionNet


def parse_args():
    """Get arguments from command-line."""
    parser = argparse.ArgumentParser(description="Convert HRNetW48_seg weights from torch to mindspore.")
    parser.add_argument("--torch_path", type=str, default=None, help="Path to input torch model.")
    parser.add_argument("--numpy_path", type=str, default=None,
                        help="Path to save/load intermediate numpy representation.")
    parser.add_argument("--mindspore_path", type=str, default=None, help="Path to save result mindspore model.")
    return parser.parse_args()


def torch2numpy(input_torch, out_numpy=None):
    """
    Convert torch model to numpy

    Args:
        input_torch: path to .pth model
        out_numpy: path to save .npy model (if None will not save)

    Returns:
        dict of numpy weights
    """
    weights = torch.load(input_torch)
    weights_numpy = {k: v.detach().cpu().numpy() for k, v in weights.items()}

    if out_numpy:
        with open(out_numpy, 'wb') as fp:
            pickle.dump(weights_numpy, fp)

    return weights_numpy


def numpy2mindspore(input_numpy, out_mindspore):
    """
    Convert numpy model weights to mindspore
    Args:
        input_numpy: path to .npy weights or dict of numpy arrays
        out_mindspore: path to output mindspore model
    """
    if isinstance(input_numpy, str):
        with open(input_numpy, 'rb') as fp:
            weights_numpy = pickle.load(fp)
    else:
        weights_numpy = input_numpy

    net = HighResolutionNet(config.model, 19)
    sample_ms = net.parameters_dict()

    weights_ms = {}
    miss_in_ms = set()

    for k in weights_numpy.keys():
        if k.endswith('.num_batches_tracked'):
            continue
        new_k = k

        if k.rsplit('.', 1)[0] + '.running_mean' in weights_numpy.keys():
            new_k = new_k.replace('weight', 'gamma').replace('bias', 'beta')
            new_k = new_k.replace('running_mean', 'moving_mean').replace('running_var', 'moving_variance')
        if new_k not in sample_ms.keys():
            miss_in_ms.add(k)
            continue

        weights_ms[new_k] = Parameter(weights_numpy[k], name=new_k)

    print('Missed in mindspore:\n', miss_in_ms)
    print('Missed from mindspore:\n', set(sample_ms.keys()) - set(weights_ms.keys()))

    load_param_into_net(net, weights_ms)
    save_checkpoint(net, out_mindspore)


def convert():
    """ Full convert pipeline """
    args = parse_args()
    if not args.torch_path and not args.numpy_path:
        raise ValueError('torch_path or numpy_path must be defined as input')
    if not args.mindspore_path and not args.numpy_path:
        raise ValueError('mindspore_path or numpy_path must be defined as output')

    numpy_weights = None
    if args.torch_path:
        numpy_weights = torch2numpy(input_torch=args.torch_path, out_numpy=args.numpy_path)
        print('Converted to numpy!')

    if args.mindspore_path:
        if not numpy_weights:
            numpy_weights = args.numpy_path

        numpy2mindspore(input_numpy=numpy_weights, out_mindspore=args.mindspore_path)
        print('Converted to mindspore!')


if __name__ == '__main__':
    convert()
