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

"""DBPN test"""
import argparse
import ast
import time

from mindspore import load_checkpoint, load_param_into_net, context

from src.dataset.dataset import DatasetVal, create_val_dataset
from src.model.generator import get_generator
from src.util.utils import compute_psnr, save_img

parser = argparse.ArgumentParser(description="DBPN eval")
parser.add_argument("--device_id", type=int, default=4, help="device id, default: 0.")
parser.add_argument("--val_GT_path", type=str, default=r'/data/DBPN_data/Set5/HR')
parser.add_argument("--val_LR_path", type=str, default=r'/data/DBPN_data/Set5/LR')
parser.add_argument("--ckpt", type=str, default=r'ckpt/gen/DDBPN_best.ckpt')
parser.add_argument('--upscale_factor', type=int, default=4, choices=[2, 4, 8],
                    help="Super resolution upscale factor")
parser.add_argument('--model_type', type=str, default='DDBPN', choices=["DBPNS", "DDBPN", "DBPN", "DDBPNL"])
parser.add_argument('--vgg', type=ast.literal_eval, default=True, help="use vgg")
parser.add_argument('--isgan', type=ast.literal_eval, default=False, help="is_gan decides the way of training ")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--save_eval_path', type=str, default="./Results/eval", help='save eval image path')

args = parser.parse_args()
print(args)


def predict(ds, model):
    """predict
    Args:
        ds(Dataset): eval dataset
        model(Cell): the generate model
    """
    for index, batch in enumerate(ds.create_dict_iterator(), 1):
        lr = batch['input_image']
        hr = batch['target_image']
        t0 = time.time()
        prediction = model(lr)
        psnr_value = compute_psnr(prediction.squeeze(), hr.squeeze())
        t1 = time.time()
        print("lr shape", lr.shape)
        print("hr shape", hr.shape)
        print("prediction shape", prediction.shape)
        print("===> Processing: {} compute_psnr:{:.4f}|| Timer: {:.2f} sec.".format(index, psnr_value, (t1 - t0)))
        save_img(prediction, str(index), args.save_eval_path)


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_id=args.device_id)
    val_dataset = DatasetVal(args.val_GT_path, args.val_LR_path, args)
    val_ds = create_val_dataset(val_dataset, args)
    print("=======> load model ckpt")
    params = load_checkpoint(args.ckpt)
    print('===> Building model ', args.model_type)
    netG = get_generator(args.model_type, scale_factor=args.upscale_factor)
    load_param_into_net(netG, params)
    predict(val_ds, netG)
