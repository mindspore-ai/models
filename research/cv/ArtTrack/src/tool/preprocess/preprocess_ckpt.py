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

from mindspore import load_checkpoint, save_checkpoint
from config import check_config

def pre_top(param, param_dicts):
    '''
    preprocess_top
    '''
    param_dict = {}
    if param == 'conv1.weight':
        new_param = 'net.resnet101.conv2d_same.conv2d.1.conv.weight'
    elif param == 'bn1.gamma':
        new_param = 'net.resnet101.conv2d_same.conv2d.1.batchnorm.gamma'
    elif param == 'bn1.beta':
        new_param = 'net.resnet101.conv2d_same.conv2d.1.batchnorm.beta'
    elif param == 'bn1.moving_mean':
        new_param = 'net.resnet101.conv2d_same.conv2d.1.batchnorm.moving_mean'
    else:
        new_param = 'net.resnet101.conv2d_same.conv2d.1.batchnorm.moving_variance'

    param_dict['name'] = new_param
    param_dict['data'] = param_dicts[param]
    return param_dict

def pre_tail(param_list):
    l_ = []
    if param_list[0] == 'layer1':
        l_.append('layer.blocks_cell.0')
    elif param_list[0] == 'layer2':
        l_.append('layer.blocks_cell.1')
    elif param_list[0] == 'layer3':
        l_.append('layer.blocks_cell.2')
    elif param_list[0] == 'layer4':
        l_.append('layer.blocks_cell.3')

    l_.append('units_cells.' + param_list[1])

    if len(param_list) == 5:
        if param_list[3] == '0':
            l_.append('conv2d_shortcut.conv')
        else:
            l_.append('conv2d_shortcut.batchnorm')
    elif param_list[2][0:-1] == 'conv':
        if param_list[2][-1] == '2':
            if (param_list[0] == 'layer1' and param_list[1] == '2') or \
                    (param_list[0] == 'layer2' and param_list[1] == '3'):
                l_.append('conv2d2.conv2d.1.conv')
            else:
                l_.append('conv2d2.conv2d.conv')
        else:
            l_.append('conv2d' + param_list[2][-1] + '.conv')
    elif param_list[2][0:-1] == 'bn':
        if param_list[2][-1] == '2':
            if (param_list[0] == 'layer1' and param_list[1] == '2') or \
                    (param_list[0] == 'layer2' and param_list[1] == '3'):
                l_.append('conv2d2.conv2d.1.batchnorm')
            else:
                l_.append('conv2d2.conv2d.batchnorm')
        else:
            l_.append('conv2d' + param_list[2][-1] + '.batchnorm')

    l_.append(param_list[-1])
    return l_

def preprocess_ckpt(cfg, param_dicts):
    '''
    preprocess_ckpt
    '''
    new_params_lists = []
    for param in param_dicts:
        if param in ['end_point.weight', 'end_point.bias', 'step']:
            continue
        if param == 'global_step':
            break

        if param in ['conv1.weight', 'bn1.gamma', 'bn1.beta',
                     'bn1.moving_mean', 'bn1.moving_variance', 'bn1.moving_variance']:
            param_dict = pre_top(param, param_dicts)
        else:
            param_dict = {}
            l = param.split('.')
            l_ = pre_tail(l)
            new_param = 'net.resnet101'
            for i in l_:
                new_param = new_param + '.' + i
            param_dict['name'] = new_param
            param_dict['data'] = param_dicts[param]
        new_params_lists.append(param_dict)

    save_checkpoint(new_params_lists, cfg.under_line.DATASET_ROOT + "resnet101.ckpt")

if __name__ == "__main__":
    ckpt = "out/resnet101_ascend_v130_imagenet2012_official_cv_bs32_top1acc78.55__top5acc94.34.ckpt"
    preprocess_ckpt(check_config("./config/mpii_eval_ascend.yaml", None), load_checkpoint(ckpt))
