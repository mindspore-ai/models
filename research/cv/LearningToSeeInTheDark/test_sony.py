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
"""test"""
from __future__ import division
import argparse
import sys
import os
import glob
import yaml
import rawpy
from PIL import Image
from skimage import metrics
import numpy as np
import mindspore
from mindspore import context, Tensor, dtype, Model

from mindspore.train.serialization import load_checkpoint, load_param_into_net


def parse_arguments(argv):
    """receive arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_target', default='Ascend',
                        help='device where the code will be implemented')
    # Ascend mode
    parser.add_argument('--data_url', required=False, default=None, help='Location of data')
    parser.add_argument('--checkpoint_path', required=False, default=None, help='ckpt file path')
    # GPU mode
    parser.add_argument('--config', type=str, default="src/Sony_config.yaml",
                        help='Directory of config.')

    return parser.parse_args(argv)


def test_gpu(args):
    from src.unet import UNet
    from src.data_utils import pack_raw, toimage

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    mindspore.context.set_context(mode=config["mode"], device_target=config["device"], device_id=config["device_id"])

    # model
    unet = UNet()
    param_dict = load_checkpoint(config["checkpoint"])
    load_param_into_net(unet, param_dict)

    model = Model(unet)

    # dataset
    # get test IDs
    test_fns = glob.glob(config["gt_dir"] + '/1*.ARW')
    test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

    if not os.path.isdir(config["result_dir"] + 'final/'):
        os.makedirs(config["result_dir"] + 'final/')

    ssmi_total = 0
    psnr_total = 0
    test_sample = 0
    for test_id in test_ids:
        # test the first image in each sequence
        in_files = glob.glob(config["input_dir"] + '%05d_00*.ARW' % test_id)
        for k in range(len(in_files)):
            in_path = in_files[k]
            in_fn = os.path.basename(in_path)
            print(in_fn)
            gt_files = glob.glob(config["gt_dir"] + '%05d_00*.ARW' % test_id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)

            raw = rawpy.imread(in_path)
            input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

            im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

            # crop
            input_full = input_full[:, 0:1024, 0:1024, :]
            gt_full = gt_full[:, 0:2048, 0:2048, :]
            scale_full = scale_full[:, 0:2048, 0:2048, :]

            input_full = np.minimum(input_full, 1.0)
            input_full = np.transpose(input_full, (0, 3, 1, 2))

            input_full = Tensor(input_full)
            output = model.predict(input_full)
            output = output.asnumpy()

            output = np.minimum(np.maximum(output, 0), 1)

            output = output[0, :, :, :]
            gt_full = gt_full[0, :, :, :]
            scale_full = scale_full[0, :, :, :]
            scale_full = scale_full * np.mean(gt_full) / np.mean(
                scale_full)  # scale the low-light image to the same mean of the groundtruth

            toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
                config["result_dir"] + 'final/%5d_00_%d_out.png' % (test_id, ratio))
            toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
                config["result_dir"] + 'final/%5d_00_%d_scale.png' % (test_id, ratio))
            toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
                config["result_dir"] + 'final/%5d_00_%d_gt.png' % (test_id, ratio))

            output = output * 255
            output = output.astype(np.uint8)
            output = np.transpose(output, (1, 2, 0))
            scale_full = scale_full * 255
            scale_full = scale_full.astype(np.uint8)
            gt_full = gt_full * 255
            gt_full = gt_full.astype(np.uint8)

            output_img = Image.fromarray(output).convert("RGB")
            gt_img = Image.fromarray(gt_full).convert("RGB")

            ssmi = metrics.structural_similarity(np.array(output_img), np.array(gt_img), multichannel=True)
            print("SSMI is {}".format(ssmi))
            psnr = metrics.peak_signal_noise_ratio(np.array(output_img), np.array(gt_img))
            print("PSNR is {}".format(psnr))

            ssmi_total += ssmi
            psnr_total += psnr
            test_sample += 1

    print("Test sample: ", test_sample)
    print("Average SSMI: ", ssmi_total / test_sample)
    print("Average PSNR: ", psnr_total / test_sample)


def test_ascend(args):
    from src.unet_parts import UNet
    from src.myutils import get_test_data

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    local_data_path = args.data_url
    input_dir = os.path.join(local_data_path, 'short/')
    gt_dir = os.path.join(local_data_path, 'long/')
    test_fns = glob.glob(gt_dir + '1*.hdf5')
    test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]
    ckpt_dir = args.checkpoint_path
    param_dict = load_checkpoint(ckpt_dir)
    net = UNet(4, 12)
    load_param_into_net(net, param_dict)

    in_ims = get_test_data(input_dir, gt_dir, test_ids)
    i = 0
    for in_im in in_ims:
        output = net(Tensor(in_im, dtype.float32))
        output = output.asnumpy()
        output = np.minimum(np.maximum(output, 0), 1)
        output = np.trunc(output[0] * 255)
        output = output.astype(np.int8)
        output = output.transpose([1, 2, 0])
        image_out = Image.fromarray(output, 'RGB')
        image_out.save('output_%d.png' % i)
        i += 1


def test(args):
    if args.device_target == "Ascend":
        test_ascend(args)
    elif args.device_target == "GPU":
        test_gpu(args)


if __name__ == "__main__":
    test(parse_arguments(sys.argv[1:]))
