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

"""RBPN test"""
import argparse
import ast
import os
import numpy as np
import cv2
import mindspore.ops
import mindspore.nn as nn
from mindspore import load_checkpoint, load_param_into_net, context
from src.model.rbpn import Net as RBPN
from src.datasets.dataset import RBPNDatasetTest, create_val_dataset
from src.util.utils import PSNR




parser = argparse.ArgumentParser(description="RBPN eval")
parser.add_argument("--device_id", type=int, default=1, help="device id, default: 0.")
parser.add_argument("--val_path", type=str, default=r'/mass_data/dataset/SPMCS')
parser.add_argument("--ckpt", type=str, default=r'./weights/rbpn_epoch209.ckpt')
parser.add_argument('--upscale_factor', type=int, default=4, choices=[2, 4, 8],
                    help="Super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--model_type', type=str, default='RBPN')
parser.add_argument('--save_eval_path', type=str, default="./Results/eval", help='save eval image path')
parser.add_argument('--file_list', type=str, default='veni5_015.txt')
parser.add_argument('--other_dataset', type=ast.literal_eval, default=True, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=ast.literal_eval, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=7)
parser.add_argument('--residual', type=ast.literal_eval, default=False)
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--save_img', type=ast.literal_eval, default=False)
parser.add_argument('--output', type=str, default="./Results", help='save img')
parser.add_argument('--pic_dir', type=str, default='Vid4')

args = parser.parse_args()
print(args)

def predict(ds, model_rbpn):
    """predict
    Args:
        ds(Dataset): eval dataset
        model_rbpn(Cell): the generate model
    """
    avg_psnr = 0
    avg_ssim = 0
    times = 0
    for index, batch in enumerate(ds.create_dict_iterator(), 1):
        x = batch['input_image']
        target = batch['target_image']
        bicubic = batch['bicubic_image']
        neighbor = batch['neighbor_image']
        flow = batch['flow_image']

        prediction = model_rbpn(x, neighbor, flow)

        if args.residual:
            prediction = prediction + bicubic

        if args.save_img:
            save_imgs(prediction, str(index), True)
            save_imgs(target, str(index), False)

        prediction_np = prediction[0].asnumpy().astype(np.float32)
        prediction_np = prediction_np * 255.

        target_np = target.squeeze().asnumpy().astype(np.float32)
        target_np = target_np * 255.

        psnr_predicted = PSNR(prediction_np, target_np, shave_border=args.upscale_factor)
        avg_psnr += psnr_predicted
        print("[{}]psnr{}:".format(index, psnr_predicted))
        ssim_predicted = ssimNet(target, prediction)
        avg_ssim += ssim_predicted
        print("[{}]ssim{}:".format(index, ssim_predicted))

        times = index
    print("avg_psnr: ", avg_psnr/times)
    print("avg_ssim: ", avg_ssim/times)




def save_imgs(img, img_name, pred_flag):
    save_img = img.squeeze()
    save_img = mindspore.ops.clip_by_value(save_img, 0, 1)
    save_img = save_img.asnumpy().astype(np.float32).transpose(1, 2, 0)

    # save img
    save_dir = os.path.join(args.output, args.pic_dir,
                            os.path.splitext(args.file_list)[0] + '_' + str(args.upscale_factor) + 'x')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if pred_flag:
        save_fn = save_dir + '/' + img_name + '_' + args.model_type + 'F' + str(args.nFrames) + '.png'
    else:
        save_fn = save_dir + '/' + img_name + '.png'
    cv2.imwrite(save_fn, cv2.cvtColor(save_img * 255, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])



if __name__ == "__main__":
    mindspore.set_seed(args.seed)
    context.set_context(mode=context.GRAPH_MODE, device_id=args.device_id)

    val_dataset = RBPNDatasetTest(args.val_path, args.nFrames, args.upscale_factor, args.file_list, args.other_dataset,
                                  args.future_frame)
    val_ds = create_val_dataset(val_dataset, args)
    print("=======> load model ckpt")
    ckpt = args.ckpt
    params = load_checkpoint(ckpt)
    print('===> Building model ', args.model_type)

    model = RBPN(num_channels=3, base_filter=256, feat=64, num_stages=3, n_resblock=5, nFrames=args.nFrames,
                 scale_factor=args.upscale_factor)
    load_param_into_net(model, params)
    ssimNet = nn.SSIM()
    predict(val_ds, model)
