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

"""file for evaling"""
import argparse
import numpy as np
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore import context
from src.model.generator import RRDBNet
from src.dataset.testdataset import create_testdataset
from src.util.util import tensor2img, calculate_psnr

set_seed(1)
parser = argparse.ArgumentParser(description="ESRGAN eval")
parser.add_argument("--test_LR_path", type=str, default='/data/DIV2K/Set14/LRbicx4')
parser.add_argument("--test_GT_path", type=str, default='/data/DIV2K/Set14/GTmod12')
parser.add_argument("--res_num", type=int, default=16)
parser.add_argument("--scale", type=int, default=4)
parser.add_argument("--generator_path", type=str, default='./ckpt/195_gan_generator.ckpt')
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")
parser.add_argument('--platform', type=str, default='Ascend', choices=('Ascend', 'GPU', 'CPU'))
if __name__ == '__main__':
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_id=args.device_id, save_graphs=False)
    context.set_context(device_target=args.platform)
    test_ds = create_testdataset(1, args.test_LR_path, args.test_GT_path)
    test_data_loader = test_ds.create_dict_iterator()
    generator = RRDBNet(3, 3)
    params = load_checkpoint(args.generator_path)
    load_param_into_net(generator, params)

    psnr_list = []

    print("=======starting test=====")
    for test in test_data_loader:
        lr = test['LR']
        hr = test['HR']
        output = generator(lr)
        sr_img = tensor2img(output)
        gt_img = tensor2img(hr)
        psnr = calculate_psnr(sr_img, gt_img)
        psnr_list.append(psnr)
    print("avg PSNR:", np.mean(psnr_list))
