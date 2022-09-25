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
import os
import argparse
from pprint import pprint
import numpy as np
import cv2
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore as ms
from src.models import MODNet
from src.utils import load_config, merge
from src.transforms import create_operators, Compose
from src.dataset import MattingDataset

def preLauch():
    """parse the console argument"""
    parser = argparse.ArgumentParser(description='matting objective decomposition network !')
    parser.add_argument("-c", "--config", help="configuration file to use")
    parser.add_argument('--device_target', type=str, default='GPU',
                        help='device target, Ascend or GPU (Default: GPU)')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id of training (Default: 0)')
    parser.add_argument('--dataset_path', type=str, default='./data/',
                        help='dataset dir')
    parser.add_argument('--output_path', type=str, default='./output',
                        help='output path of training (default ./output)')
    parser.add_argument('--init_weight_path', type=str,
                        default='./init_weight.ckpt', help='checkpoint dir of init_weight')
    parser.add_argument('--ckpt_path', type=str,
                        default=None, help='checkpoint path')
    parser.add_argument('--seed', type=int,
                        default=2022, help='seed')
    args = parser.parse_args()
    ms.common.set_seed(args.seed)
    cfg = merge(args, load_config(args.config))
    pprint(cfg)

    context.set_context(mode=context.GRAPH_MODE,
                        save_graphs=False,
                        device_target=cfg.device_target,
                        device_id=cfg.device_id)

    return cfg


def get_mad(alpha_fg, image_gt):
    mad = np.abs(alpha_fg - image_gt).sum()/alpha_fg.size
    return mad

def get_mse(alpha_fg, image_gt):
    mse = np.square(alpha_fg - image_gt).sum() / alpha_fg.size
    return mse

def main():
    args = preLauch()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    # create network
    modnet = MODNet(3, backbone_pretrained=False)

    if args.ckpt_path:
        param_dict = load_checkpoint(args.ckpt_path)
        load_param_into_net(modnet, param_dict)
        print('load ckpt from', args.ckpt_path)
    modnet.set_train(False)
    data_cfg = args.val_dataset
    data_transformer_list = create_operators(data_cfg['transforms'])
    data_transformers = Compose(data_transformer_list)
    dataset_generator = MattingDataset(dataset_root=data_cfg['dataset_root'], transformers=data_transformers,
                                       img_subpath=data_cfg['img_subpath'], alpha_subpath=data_cfg['alpha_subpath'],
                                       fg_names=data_cfg['fg_names'], bg_names=data_cfg['bg_names'],
                                       name=data_cfg['name'], mode='test')

    sad_list, mse_list = [], []
    for image, alpha, image_file in dataset_generator:
        print('Process image: {0}'.format(image_file))

        _, im_h, im_w = alpha.shape
        image_ms = image[None, :, :, :]
        image_ms = ms.Tensor(image_ms, ms.float32)

        pred_matte = modnet(image_ms, inference=True)[0]
        pred_matte = pred_matte.asnumpy().squeeze()
        pred_matte = cv2.resize(pred_matte, (im_w, im_h), interpolation=cv2.INTER_AREA)
        mad, mse = get_mad(pred_matte, alpha), get_mse(pred_matte, alpha)

        sad_list.append(mad)
        mse_list.append(mse)
        print(mse, mad)

        cv2.imwrite(os.path.join(args.output_path, image_file), (pred_matte * 255).astype('uint8'))

    print('eval result, MSE: {}, MAD: {}'.format(np.asarray(mse_list).mean(), np.asarray(sad_list).mean()))

if __name__ == '__main__':
    main()
