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
'''transfer the pth into ckpt'''
import torch
from mindspore import save_checkpoint, Tensor
from src.models import get_r2plus1d_model
from src.config import config as cfg

if __name__ == '__main__':
    model = get_r2plus1d_model(cfg.num_classes, cfg.layer_num)
    names = []
    for cell in model.parameters_and_names():
        if cell[0]:
            names.append(cell[0])

    resume_sate = torch.load("./r2plus1d_v1_resnet34_kinetics400-5102fd17.pth", \
                             map_location=torch.device('cpu'))

    ans_dict = []
    for k, value in resume_sate.items():
        if not k.split('.')[-1] == "num_batches_tracked":
            k = k.replace('bn2', 'bn2.bn2d')
            k = k.replace('bn1', 'bn1.bn2d')
            k = k.replace('downsample.1', 'downsample.1.bn2d')
        if k.split('.')[-2] in ['bn2d']:
            k = k.replace('weight', 'gamma')
            k = k.replace('bias', 'beta')
            k = k.replace('running_mean', 'moving_mean')
            k = k.replace('running_var', 'moving_variance')
        if not k.split('.')[-1] == "num_batches_tracked":
            print(k)
        if k in names:
            ans_dict.append({"name": k, "data": Tensor(value.numpy())})
    save_checkpoint(ans_dict, "r2plus1d_v1_resnet34_kinetics400.ckpt")
