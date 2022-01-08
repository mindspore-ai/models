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
import os
import time
import argparse
import numpy as np
from PIL import Image
import mindspore.ops as ops
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.loss import MeanShift
from src.inceptionv3 import inceptionv3
from src.model import get_model
from src.testdataset import create_testdataset

set_seed(1)


def get_args():
    """get args"""
    parser = argparse.ArgumentParser(description="style transfer train")
    # data loader
    parser.add_argument("--content_path", type=str, default='./dataset/test/content')
    parser.add_argument("--style_path", type=str, default='./dataset/test/style')
    parser.add_argument("--inception_ckpt", type=str, default='./pretrained_model/inceptionv3.ckpt')
    parser.add_argument("--style_dim", type=int, default=100,
                        help="Style vector dimension. default: 100")
    parser.add_argument("--reshape_size", type=int, default=257,
                        help="Image size of high resolution image. default: 257")
    parser.add_argument("--crop_size", type=int, default=256,
                        help="Image size of high resolution image. default: 256")
    parser.add_argument('--init_type', type=str, default='normal', choices=("normal", "xavier"), \
                        help='network initialization, default is normal.')
    parser.add_argument('--init_gain', type=float, default=0.02, \
                        help='scaling factor for normal, xavier and orthogonal, default is 0.02.')

    parser.add_argument('--platform', type=str, default='Ascend', help='Ascend or GPU')
    parser.add_argument("--batchsize", default=1, type=int, help="Batch size for training")
    parser.add_argument("--ckpt_path", type=str, default='./ckpt/style_transfer_model_0100.ckpt')
    parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")
    return parser.parse_args()


def test(opts):
    """test"""
    # data loader
    test_ds = create_testdataset(opts)
    test_data_loader = test_ds.create_dict_iterator()
    # model loader
    inception = inceptionv3(opts.inception_ckpt)
    meanshift = MeanShift()
    for p in meanshift.get_parameters():
        p.requires_grad = False
    transfer_net = get_model(opts)
    params = load_checkpoint(opts.ckpt_path)
    load_param_into_net(transfer_net, params)

    op_reduce_dim = ops.ReduceSum(keep_dims=False)
    op_concat = ops.Concat(axis=3)
    if not os.path.exists('output'):
        os.makedirs('output')

    print("=======starting test=====")
    count = 0
    mysince = time.time()
    for data in test_data_loader:
        content_img = data['content']
        style_img = data['style']

        c_img = (content_img + 1.0) / 2.0
        s_img = (style_img + 1.0) / 2.0
        s_img_shift = meanshift(s_img)

        s_in_feat = inception(s_img_shift)

        stylied_img = transfer_net(content_img, s_in_feat)

        sty_img = (stylied_img + 1.0) / 2.0
        sty_img = op_concat((c_img, s_img, sty_img))

        sty_img = op_reduce_dim(sty_img, 0)
        sty_img = sty_img.asnumpy()
        sty_img = np.clip(sty_img, 0, 1.0)
        sty_img = sty_img.transpose(1, 2, 0)
        sty_img = Image.fromarray(np.uint8(sty_img * 255))
        sty_img.save(os.path.join('output', str(count) + 'stylied.png'))

        count = count + 1
    time_elapsed = (time.time() - mysince)
    step_time = time_elapsed / count
    print('per step needs time:{:.0f}ms'.format(step_time * 1000))


def interpolation_test(opts):
    """interpolation"""
    # data loader
    test_ds = create_testdataset(opts)
    test_data_loader = test_ds.create_dict_iterator()
    # model loader
    inception = inceptionv3(opts.inception_ckpt)
    meanshift = MeanShift()
    for p in meanshift.get_parameters():
        p.requires_grad = False
    transfer_net = get_model(opts)
    params = load_checkpoint(opts.ckpt_path)
    load_param_into_net(transfer_net, params)

    op_reduce_dim = ops.ReduceSum(keep_dims=False)
    op_concat = ops.Concat(axis=3)
    if not os.path.exists('output_interpolation'):
        os.makedirs('output_interpolation')

    print("=======starting interpolation test=====")
    count = 0
    mysince = time.time()
    for data in test_data_loader:
        content_img = data['content']
        style_img = data['style']

        c_img = (content_img + 1.0) / 2.0
        c_img_shift = meanshift(c_img)
        s_img = (style_img + 1.0) / 2.0
        s_img_shift = meanshift(s_img)

        s_in_feat = inception(s_img_shift)
        c_in_feat = inception(c_img_shift)

        stylied_img = transfer_net.construct_interpolation(content_img, c_in_feat, s_in_feat)

        sty_img_1 = (stylied_img[0] + 1.0) / 2.0
        sty_img_2 = (stylied_img[1] + 1.0) / 2.0
        sty_img_3 = (stylied_img[2] + 1.0) / 2.0
        sty_img_4 = (stylied_img[3] + 1.0) / 2.0
        sty_img_5 = (stylied_img[4] + 1.0) / 2.0
        sty_img_6 = (stylied_img[5] + 1.0) / 2.0
        sty_img = op_concat((c_img, sty_img_1, sty_img_2, sty_img_3, sty_img_4, sty_img_5, sty_img_6, s_img))
        sty_img = op_reduce_dim(sty_img, 0)
        sty_img = sty_img.asnumpy()
        sty_img = np.clip(sty_img, 0, 1.0)
        sty_img = sty_img.transpose(1, 2, 0)
        sty_img = Image.fromarray(np.uint8(sty_img * 255))
        sty_img.save(os.path.join('output_interpolation', str(count) + 'interpolation_stylied.png'))

        count = count + 1
    time_elapsed = (time.time() - mysince)
    step_time = time_elapsed / count
    print('per step needs time:{:.0f}ms'.format(step_time * 1000))


if __name__ == '__main__':
    args = get_args()
    print('start')
    context.set_context(mode=context.GRAPH_MODE, device_id=args.device_id, save_graphs=False)
    test(args)
    interpolation_test(args)
    print('finish')
