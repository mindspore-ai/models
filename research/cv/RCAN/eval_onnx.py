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
"""eval script"""
import time
import numpy as np
import mindspore.dataset as ds
import onnxruntime as ort
from src.args import args
from src.data.srdata import SRData
from src.metrics import calc_psnr, quantize, calc_ssim
from src.data.div2k import DIV2K


def create_session(checkpoint_path, target_device):
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {target_device}, '
            f'Expected one of: "CPU", "GPU"'
        )
    sess = ort.InferenceSession(checkpoint_path, providers=providers)
    name = sess.get_inputs()[0].name
    return sess, name


def eval_net():
    """eval"""
    if args.epochs == 0:
        args.epochs = 100
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False
    if args.data_test[0] == 'DIV2K':
        train_dataset = DIV2K(args, name=args.data_test, train=False, benchmark=False)
    else:
        train_dataset = SRData(args, name=args.data_test, train=False, benchmark=False)
    train_de_dataset = ds.GeneratorDataset(train_dataset, ['LR', 'HR'], shuffle=False)
    train_de_dataset = train_de_dataset.batch(1, drop_remainder=True)
    train_loader = train_de_dataset.create_dict_iterator(output_numpy=True)
    print('load mindspore net successfully.')
    num_imgs = train_de_dataset.get_dataset_size()
    psnrs = np.zeros((num_imgs, 1))
    ssims = np.zeros((num_imgs, 1))
    for batch_idx, imgs in enumerate(train_loader):
        lr = imgs['LR']
        hr = imgs['HR']
        img_shape = lr.shape
        onnx_file = args.onnx_path + '//' + str(img_shape[2]) + '_' + str(img_shape[3]) + '.onnx'
        session, input_name = create_session(onnx_file, 'GPU')
        pred = session.run(None, {input_name: lr})[0]
        pred_np = pred
        pred_np = quantize(pred_np, 255)
        psnr = calc_psnr(pred_np, hr, args.scale, 255.0)
        pred_np = pred_np.reshape(pred_np.shape[-3:]).transpose(1, 2, 0)
        hr = hr.reshape(hr.shape[-3:]).transpose(1, 2, 0)
        ssim = calc_ssim(pred_np, hr, args.scale)
        print("current psnr: ", psnr)
        print("current ssim: ", ssim)
        psnrs[batch_idx, 0] = psnr
        ssims[batch_idx, 0] = ssim
    print('Mean psnr of %s x%s is %.4f' % (args.data_test[0], args.scale, psnrs.mean(axis=0)[0]))
    print('Mean ssim of %s x%s is %.4f' % (args.data_test[0], args.scale, ssims.mean(axis=0)[0]))


if __name__ == '__main__':
    time_start = time.time()
    print("Start eval function!")
    eval_net()
    time_end = time.time()
    print('eval_time: %f' % (time_end - time_start))
