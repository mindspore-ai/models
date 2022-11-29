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
import numpy as np
import mindspore.ops
import mindspore.dataset as ds
from mindspore import Tensor, context
from mindspore.common import dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.CSNLN import CSNLN
from src.dataset.benchmark import Benchmark
from model_utils.metrics import calc_psnr, quantize
from model_utils.config import config as cfg


def sub_mean(x):
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, 0, :, :] -= red_channel_mean
    x[:, 1, :, :] -= green_channel_mean
    x[:, 2, :, :] -= blue_channel_mean
    return x


def add_mean(x):
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, 0, :, :] += red_channel_mean
    x[:, 1, :, :] += green_channel_mean
    x[:, 2, :, :] += blue_channel_mean
    return x


def eval_net():
    """eval"""
    device_id = 0
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target, device_id=device_id)
    if cfg.epoches == 0:
        cfg.epoches = 1e8

    train_dataset = Benchmark(cfg, name=cfg.data_test, train=False, benchmark=False)
    train_de_dataset = ds.GeneratorDataset(train_dataset, ['LR', 'HR'], shuffle=False)
    train_de_dataset = train_de_dataset.batch(1, drop_remainder=True)
    train_loader = train_de_dataset.create_dict_iterator(output_numpy=True)

    net_m = CSNLN(cfg)

    def forward_chop(x, shave=10, min_size=3600):
        scale = cfg.scale[0]
        if scale == 4:
            min_size = 3600
        n_GPUs = min(cfg.device_num, 4)
        n, c, h, w = x.shape
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        h_size += scale - h_size % scale
        w_size += scale - w_size % scale
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]
        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = mindspore.ops.Concat(axis=0)(lr_list[i:(i + n_GPUs)])
                sr_batch = net_m(lr_batch)
                sr_list.extend(mindspore.ops.Split(axis=0, output_num=n_GPUs)(sr_batch))
        else:
            sr_list = [
                forward_chop(patch, shave=shave, min_size=min_size) for patch in lr_list
            ]
        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale
        output = mindspore.ops.Zeros()((n, c, h, w), mindspore.float32)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
        return output

    if cfg.test_only:
        param_dict = load_checkpoint(cfg.ckpt_file)
        load_param_into_net(net_m, param_dict)
    total_params = 0

    for param in net_m.trainable_params():
        total_params += np.prod(param.shape)
    print('params:', total_params)
    net_m.set_train(False)
    print('load mindspore net successfully.')
    num_imgs = train_de_dataset.get_dataset_size()
    psnrs = np.zeros((num_imgs, 1))
    for batch_idx, imgs in enumerate(train_loader):
        lr = imgs['LR']
        hr = imgs['HR']
        lr = Tensor(lr, mstype.float32)

        pred = forward_chop(lr)

        pred_np = pred.asnumpy()
        pred_np = quantize(pred_np, 1)
        psnr = calc_psnr(pred_np, hr, cfg.scale[0], 1)
        print(batch_idx, ":", "current psnr: ", psnr)
        psnrs[batch_idx, 0] = psnr
    print('Mean psnr of %s x%s is %.4f' % (cfg.data_test[0], cfg.scale[0], psnrs.mean(axis=0)[0]))


if __name__ == '__main__':
    print("Start eval function!")
    eval_net()
