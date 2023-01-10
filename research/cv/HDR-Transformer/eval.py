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
#-*- coding:utf-8 -*-
import os
import argparse
import mindspore as ms
import mindspore.dataset as mds
from mindspore import Tensor
import imageio

from models.hdr_transformer import HDRTransformer
from utils.utils import batch_psnr
from dataset.dataset_sig17 import SIG17_Validation_Dataset

parser = argparse.ArgumentParser(description="Test Setting")
parser.add_argument("--dataset_dir", type=str, default='./', help='dataset directory')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='N', help='testing batch size (default: 1)')
parser.add_argument('--num_workers', type=int, \
default=1, metavar='N', help='number of workers to fetch data (default: 1)')
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--pretrained_model', type=str, default='./checkpoints/pretrained_model.ckpt')
parser.add_argument('--test_best', action='store_true', default=False)
parser.add_argument('--save_results', action='store_true', default=False)
parser.add_argument('--save_dir', type=str, default="./results/hdr_transformer")
parser.add_argument('--model_arch', type=int, default=0)

# for evaluation with limited GPU memory

def main():
    # Settings
    args = parser.parse_args()

    # pretrained_model
    print(">>>>>>>>> Start Testing >>>>>>>>>")
    print("Load weights from: ", args.pretrained_model)

    # cuda and devices
    device_id = int(os.getenv('DEVICE_ID', '0'))
    ms.set_context(mode=ms.GRAPH_MODE, device_target='GPU', device_id=device_id, save_graphs=False)

    # model architecture
    model_dict = {
        0: HDRTransformer(embed_dim=60, depths=[6, 6, 6], num_heads=[6, 6, 6], mlp_ratio=2, in_chans=6),
    }
    model = model_dict[args.model_arch]
    param_dict = ms.load_checkpoint(args.pretrained_model)
    ms.load_param_into_net(model, param_dict)
    model.set_train(False)
    datasets = SIG17_Validation_Dataset(args.dataset_dir, crop_size=args.patch_size)
    dataloader = mds.GeneratorDataset(datasets, ["input0", "input1", "input2", "label"], shuffle=False)
    dataloader = dataloader.batch(args.test_batch_size,
                                  output_columns=["input0", "input1", "input2", "label"])
    test_dataloader = dataloader.create_dict_iterator(output_numpy=True)
    for batch_idx, batch_data in enumerate(test_dataloader):
        batch_ldr0, batch_ldr1, batch_ldr2 = Tensor(batch_data['input0']), Tensor(batch_data['input1']), \
                                             Tensor(batch_data['input2'])
        label = Tensor(batch_data['label'])
        pred = model(batch_ldr0, batch_ldr1, batch_ldr2)
        psnr = batch_psnr(pred, label, 1.0)
        pred = label[0].asnumpy().transpose(1, 2, 0)
        imageio.imsave(f'./results/{batch_idx}.jpg', pred)
        print('psnr: {}'.format(psnr))


if __name__ == '__main__':
    main()
