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
"""
Model testing entrypoint.
"""


import os
import argparse
import time
import datetime
import moxing as mox
import mindspore as ms
from mindspore import context
from mindspore import Model, load_checkpoint, load_param_into_net
import mindspore.nn as nn

import src.options.options as option
import src.utils.util as util
from src.data.util import bgr2ycbcr
from src.data import create_dataset
from src.network import create_model, IRN_loss


def obs_data2modelarts(FLAGS):
    """
    Copy train data from obs to modelarts by using moxing api.
    """
    start = datetime.datetime.now()
    print("===>>>Copy files from obs:{} to modelarts dir:{}".format(FLAGS.data_url, FLAGS.modelarts_data_dir))
    mox.file.copy_parallel(src_url=FLAGS.data_url, dst_url=FLAGS.modelarts_data_dir)
    end = datetime.datetime.now()
    print("===>>>Copy from obs to modelarts, time use:{}(s)".format((end - start).seconds))
    files = os.listdir(FLAGS.modelarts_data_dir)
    print("===>>>Files:", files)


current_path = os.path.abspath(__file__)
root_path = os.path.dirname(current_path)
# Path to option YMAL file.
X2_TEST_YAML_FILE = os.path.join(root_path, "src", "options", "test", "test_IRN_x2.yml")
# Path to option YMAL file.
X4_TEST_YAML_FILE = os.path.join(root_path, "src", "options", "test", "test_IRN_x4.yml")


if __name__ == '__main__':
    begin = time.time()
    parser = argparse.ArgumentParser(description="IRN testing args")
    parser.add_argument("--modelarts_FLAG", type=bool, default=True,
                        help="use modelarts or not")
    parser.add_argument('--data_url', type=str, default='./data/',
                        help="local obs data path, used for obs_data2modelarts")
    parser.add_argument("--modelarts_data_dir", type=str, default="/cache/dataset/",
                        help="modelarts data path, used for obs_data2modelarts")
    parser.add_argument("--testing_dataset", type=str, default="/cache/dataset/DIV2K/DIV2K_train_HR",
                        help="modelarts dataset path, used for testing")
    parser.add_argument("--ckpt_url", type=str,
                        default="/cache/dataset/model/IRN.ckpt",
                        help="modelarts ckpt path, used for testing")

    parser.add_argument('--scale', type=int, default=4, choices=(2, 4),
                        help='Rescaling Parameter.')
    parser.add_argument('--dataset_GT_path', type=str, default='/home/nonroot/DIV2K/DIV2K_train_HR',
                        help='Path to the folder where the intended GT dataset is stored.')
    parser.add_argument('--dataset_LQ_path', type=str, default=None,
                        help='Path to the folder where the intended LQ dataset is stored.')
    parser.add_argument('--resume_state', type=str, default=None,
                        help='Path to the checkpoint.')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=("GPU", "Ascend"),
                        help="Device target, support GPU, Ascend.")

    args = parser.parse_args()

    if args.modelarts_FLAG:
        obs_data2modelarts(FLAGS=args)
        args.dataset_GT_path = args.testing_dataset
        args.resume_state = args.ckpt_url

    if args.scale == 2:
        opt = option.parse(X2_TEST_YAML_FILE, args.dataset_GT_path, args.dataset_LQ_path, is_train=False)
    elif args.scale == 4:
        opt = option.parse(X4_TEST_YAML_FILE, args.dataset_GT_path, args.dataset_LQ_path, is_train=False)
    else:
        raise ValueError("Unsupported scale.")

    try:
        device_id = int(os.getenv('DEVICE_ID'))
    except TypeError:
        device_id = 0
    # initialize context
    context.set_context(
        mode=context.PYNATIVE_MODE,
        device_target=args.device_target,
        save_graphs=False,
        device_id=device_id
    )

    # loading options for model
    opt = option.dict_to_nonedict(opt)

    # create dataset
    dataset_opt = opt['datasets']['test']
    dataset_opt['dataroot_GT'] = args.dataset_GT_path
    dataset_opt['dataroot_LQ'] = args.dataset_LQ_path
    val_dataset = create_dataset(
        args.dataset_GT_path,
        args.scale,
        do_train=False,
        batch_size=1
    )

    step_size = val_dataset.get_dataset_size()
    print("Step size : {}".format(step_size))

    # define net
    net = create_model(opt)

    # loading resume state if exists
    if args.resume_state is not None:
        param_dict = load_checkpoint(args.resume_state)
        load_param_into_net(net, param_dict)
        print("saved model restore! " + str(args.resume_state))

    # define network with loss
    loss = IRN_loss(net, opt)

    # warp network with optimizer
    optimizer = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)

    # Model
    model = Model(network=loss, optimizer=optimizer, amp_level="O3")

    val_iter = val_dataset.create_dict_iterator()

    idx = 0
    test_hr_psnr = []
    test_hr_ssim = []
    test_y_hr_psnr = []
    test_y_hr_ssim = []

    test_lr_psnr = []
    test_y_lr_psnr = []
    test_lr_ssim = []
    test_y_lr_ssim = []

    for _ in range(val_dataset.get_dataset_size()):
        idx += 1
        val = next(val_iter)
        lq = ms.Tensor(val["downscaled"], ms.float16)
        gt = ms.Tensor(val["original"], ms.float16)
        images = loss.test(lq, gt)
        sr_img = util.tensor2img(images[3])
        gt_img = util.tensor2img(images[0])
        gt_lr_img = util.tensor2img(images[1])
        lq_img = util.tensor2img(images[2])

        gt_img = gt_img / 255.
        sr_img = sr_img / 255.
        gt_lr_img = gt_lr_img / 255.
        lq_img = lq_img / 255.

        crop_size = opt['scale']
        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
        psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
        ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
        test_hr_psnr.append(psnr)
        test_hr_ssim.append(ssim)

        lr_psnr = util.calculate_psnr(gt_lr_img * 255, lq_img * 255)
        lr_ssim = util.calculate_ssim(gt_lr_img * 255, lq_img * 255)
        test_lr_psnr.append(lr_psnr)
        test_lr_ssim.append(lr_ssim)

        avg_PSNR = sum(test_hr_psnr) / len(test_hr_psnr)
        avg_SSIM = sum(test_hr_ssim) / len(test_hr_ssim)
        lr_avg_PSNR = sum(test_lr_psnr) / len(test_lr_psnr)
        lr_avg_SSIM = sum(test_lr_ssim) / len(test_lr_ssim)

        if gt_img.shape[2] == 3:  # RGB image
            sr_img_y = bgr2ycbcr(sr_img, only_y=True)
            gt_img_y = bgr2ycbcr(gt_img, only_y=True)

            cropped_sr_img_y = sr_img_y[crop_size:-
                                        crop_size, crop_size:-crop_size]
            cropped_gt_img_y = gt_img_y[crop_size:-
                                        crop_size, crop_size:-crop_size]

            psnr_y = util.calculate_psnr(
                cropped_sr_img_y * 255, cropped_gt_img_y * 255)
            ssim_y = util.calculate_ssim(
                cropped_sr_img_y * 255, cropped_gt_img_y * 255)
            test_y_hr_psnr.append(psnr_y)
            test_y_hr_ssim.append(ssim_y)

            lr_img_y = bgr2ycbcr(lq_img, only_y=True)
            lrgt_img_y = bgr2ycbcr(gt_lr_img, only_y=True)
            psnr_y_lr = util.calculate_psnr(lr_img_y * 255, lrgt_img_y * 255)
            ssim_y_lr = util.calculate_ssim(lr_img_y * 255, lrgt_img_y * 255)
            test_y_lr_psnr.append(psnr_y_lr)
            test_y_lr_ssim.append(ssim_y_lr)

            avg_PSNR_y = sum(test_y_hr_psnr) / len(test_y_hr_psnr)
            avg_SSIM_y = sum(test_y_hr_ssim) / len(test_y_hr_ssim)
            lr_avg_PSNR_y = sum(test_y_lr_psnr) / len(test_y_lr_psnr)
            lr_avg_SSIM_y = sum(test_y_lr_ssim) / len(test_y_lr_ssim)

            print('{:4d} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}. \
                 LR PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
                  format(idx, psnr, ssim, psnr_y, ssim_y, lr_psnr, lr_ssim, psnr_y_lr, ssim_y_lr))
            print('      - avg PSNR: {:.6f} dB; avg SSIM: {:.6f}; avg lr PSNR: {:.6f}; avg lr SSIM: {:.6f}.'.format(
                avg_PSNR, avg_SSIM, lr_avg_PSNR, lr_avg_SSIM))
            print('      - avg PSNR Y: {:.6f} dB; avg SSIM Y: {:.6f}; avg lr PSNR_Y: {:.6f};  \
                avg lr SSIM_Y: {:.6f}.'.format(avg_PSNR_y, avg_SSIM_y, lr_avg_PSNR_y, lr_avg_SSIM_y))
        else:
            print('{:4d} - PSNR: {:.6f} dB; SSIM: {:.6f}. LR PSNR: {:.6f} dB; SSIM: {:.6f}.'.
                  format(idx, psnr, ssim, lr_psnr, lr_ssim))
            print('      - avg PSNR: {:.6f} dB; avg SSIM: {:.6f}; avg lr PSNR: {:.6f}; avg lr SSIM: {:.6f}.'.format(
                avg_PSNR, avg_SSIM, lr_avg_PSNR, lr_avg_SSIM))

    print("eval time : ", time.time() - begin)
