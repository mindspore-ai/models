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
The Module of Test
"""

import argparse
import yaml
from yaml import Loader

import mindspore.context as context
from mindspore import load_checkpoint, load_param_into_net

from src.dataloader import create_valid_dataset
from src.model.SRFlow import SRFlowNetNllRev


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option YMAL file.')
    args = parser.parse_args()
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)

    # 由于推理在GRAPH模式下的精度存在问题，和PYNATIVE相差较大，所以推理使用PYNATIVE
    context.set_context(device_target='GPU')
    context.set_context(mode=context.PYNATIVE_MODE)
    context.set_context(max_call_depth=3000)

    val_net = SRFlowNetNllRev(opt=opt)

    ckpt_file_name = opt["test_pretrained_model_path"]
    param_dict = load_checkpoint(ckpt_file_name)
    load_param_into_net(val_net, param_dict)

    val_dataset = create_valid_dataset(opt=opt)
    val_dataset = val_dataset.create_dict_iterator()

    count = 0
    psnr_count = 0.0
    ssim_count = 0.0
    for data in val_dataset:
        count += 1
        _, psnr, ssim = val_net(hr=data['HR'], lr=data['LR'])
        psnr_count += psnr
        ssim_count += ssim
        print('Get the number {} data psnrloss: {}, ssim_loss: {}'.format(count, psnr, ssim))

    print('The mean of psnr is: {}'.format(psnr_count/count))
    print('The mean of ssim is: {}'.format(ssim_count/count))


if __name__ == '__main__':
    main()
