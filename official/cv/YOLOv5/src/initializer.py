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
"""Parameter init."""
import math
import mindspore as ms
from mindspore import nn


def default_recurisive_init(custom_cell):
    """Initialize parameter."""
    for _, cell in custom_cell.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Dense)):
            cell.weight.set_data(ms.common.initializer.initializer(ms.common.initializer.HeUniform(math.sqrt(5)),
                                                                   cell.weight.shape, cell.weight.dtype))


def load_yolov5_params(args, network):
    """Load yolov5 backbone parameter from checkpoint."""
    if args.resume_yolov5:
        param_dict = load_checkpoint(args.resume_yolov5)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('yolo_network.'):
                param_dict_new[key[13:]] = values
                args.logger.info('in resume {}'.format(key))
            else:
                param_dict_new[key] = values
                args.logger.info('in resume {}'.format(key))

        args.logger.info('resume finished')
        load_param_into_net(network, param_dict_new)
        args.logger.info('load_model {} success'.format(args.resume_yolov5))

    if args.pretrained_checkpoint:
        param_dict = load_checkpoint(args.pretrained_checkpoint)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('yolo_network.') and key[13:] in args.checkpoint_filter_list:
                args.logger.info('remove {}'.format(key))
                continue
            elif key.startswith('yolo_network.'):
                param_dict_new[key[13:]] = values
                args.logger.info('in load {}'.format(key))
            else:
                param_dict_new[key] = values
                args.logger.info('in load {}'.format(key))

        args.logger.info('pretrained finished')
        load_param_into_net(network, param_dict_new)
        args.logger.info('load_model {} success'.format(args.pretrained_backbone))

    if args.pretrained_backbone:
        param_dict = load_checkpoint(args.pretrained_backbone)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('yolo_network.'):
                param_dict_new[key[13:]] = values
                args.logger.info('in resume {}'.format(key))
            else:
                param_dict_new[key] = values
                args.logger.info('in resume {}'.format(key))

        args.logger.info('pretrained finished')
        load_param_into_net(network, param_dict_new)
        args.logger.info('load_model {} success'.format(args.pretrained_backbone))
