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
"""eval script"""
import os
import time
import numpy as np
import mindspore.dataset as ds
from mindspore import Tensor, context
from mindspore.common import dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.args import args
from src.metric import calc_psnr, quantize, calc_ssim
from src.models.model import Dehaze
from haze_data import RESIDEDatasetGenerator

def eval_net(evds, opt):
    """eval"""
    eval_loader = evds.create_dict_iterator(output_numpy=True)
    net_m = Dehaze(3, 3)
    if opt.ckpt_path:
        param_dict = load_checkpoint(opt.ckpt_path)
        load_param_into_net(net_m, param_dict)
    net_m.set_train(False)

    print('load mindspore net successfully.')
    num_imgs = evds.get_dataset_size()
    psnrs = np.zeros((num_imgs, 1))
    ssims = np.zeros((num_imgs, 1))
    for batch_idx, imgs in enumerate(eval_loader):
        hazy = imgs['hazy']
        gt = imgs['gt']
        hazy = Tensor(hazy, mstype.float32)
        pred = net_m(hazy)
        pred_np = pred.asnumpy()
        pred_np = quantize(pred_np, 255)
        psnr = calc_psnr(pred_np, gt, 1, 255.0)
        pred_np = pred_np.reshape(pred_np.shape[-3:]).transpose(1, 2, 0)
        gt = gt.reshape(gt.shape[-3:]).transpose(1, 2, 0)
        ssim = calc_ssim(pred_np, gt, 1)
        print("current psnr: ", psnr)
        print("current ssim: ", ssim)
        psnrs[batch_idx, 0] = psnr
        ssims[batch_idx, 0] = ssim
    print('Mean psnr of %s is %.4f' % (opt.data_test[0], psnrs.mean(axis=0)[0]))
    print('Mean ssim of %s is %.4f' % (opt.data_test[0], ssims.mean(axis=0)[0]))

if __name__ == '__main__':
    device_id = int(os.getenv('DEVICE_ID', '0'))

    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_id=device_id)
    if args.modelArts_mode:
        context.set_context(device_target="Ascend")
        import moxing as mox
        local_data_url = '/cache/data'
        mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_url)
        args.dir_data = local_data_url
    else:
        context.set_context(device_target=args.device_target)

    eval_dataset = RESIDEDatasetGenerator(args, train=False)
    print(f"Eval {len(eval_dataset)} images")
    eval_ds = ds.GeneratorDataset(eval_dataset, ["hazy", "gt"], shuffle=False)
    eval_ds = eval_ds.batch(1, drop_remainder=True)

    context.set_context(max_call_depth=10000)
    time_start = time.time()
    print("Start eval function!")
    eval_net(eval_ds, args)
    time_end = time.time()
    print('eval_time: %f' % (time_end - time_start))
