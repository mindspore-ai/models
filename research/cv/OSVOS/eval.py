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
"""OSVOS eval."""
import os
import argparse
import imageio
import numpy as np
from mindspore import context, load_param_into_net, load_checkpoint
from mindspore.common import set_seed
from src.config import osvos_cfg as cfg
from src.vgg_osvos import OSVOS
from src.dataset import Imagelist


parser = argparse.ArgumentParser(description='OSVOS eval running')
parser.add_argument('--device_target', type=str, default="GPU", choices=['Ascend', 'GPU', 'CPU'],
                    help='device where the code will be implemented (default: GPU)')
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
parser.add_argument("--seq_name", type=str, default='blackswan', help="the sequence name for stage 2.")
parser.add_argument("--data_path", type=str, default="./DAVIS", help="the dataset path, default is ./DAVIS")
parser.add_argument("--ckpt_path", type=str, default=None, help="the pretrained module path, default is None")

def main():
    """Main entrance for eval"""
    args = parser.parse_args()
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target, device_id=args.device_id)
    dataevallist = Imagelist(train=False, db_root_dir=args.data_path, seq_name=args.seq_name)
    save_dir = cfg.dirResult + '/images/' + args.seq_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    net = OSVOS()
    net.set_train(False)
    assert args.ckpt_path is not None, 'No ckpt file!'
    print("=> loading checkpoint '{}'".format(args.ckpt_path))
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(net, param_dict)
    for i in range(dataevallist.__len__()):
        img, fname = dataevallist.__getitem__(i)
        outputs = net(img)
        output = outputs[-1].asnumpy()[0, ...]
        output = np.transpose(output, (1, 2, 0))
        output = (output > 0).astype(np.int)
        output = np.squeeze(output)
        imageio.imwrite(os.path.join(save_dir, os.path.basename(fname) + '.png'), output)

if __name__ == '__main__':
    set_seed(1)
    main()
