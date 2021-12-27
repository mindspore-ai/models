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
"""srcnn test"""
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.ops as ops

import PIL.Image as pil_image
import numpy as np

from src.srcnn import SRCNN
from src.metric import SRCNNpsnr
from src.utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr

from src.model_utils.config import config
from src.model_utils.moxing_adapter import sync_data

def run_test():
    cfg = config
    if cfg.device_target == "GPU" or cfg.device_target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=cfg.device_target,
                            save_graphs=False)
    else:
        raise ValueError("Unsupported device target.")

    if cfg.enable_modelarts == "True":
        sync_data(cfg.data_url, cfg.data_path)
        sync_data(cfg.checkpoint_url, cfg.checkpoint_path)
    net = SRCNN()
    lr = Tensor(config.lr, ms.float32)
    opt = nn.Adam(params=net.trainable_params(), learning_rate=lr, eps=1e-07)
    loss = nn.MSELoss(reduction='mean')

    param_dict = load_checkpoint(cfg.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'PSNR': SRCNNpsnr()})

    image_path = cfg.data_path
    if cfg.test_pic != '':
        image_path = image_path + '/' + cfg.test_pic

    image = pil_image.open(image_path).convert('RGB')

    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.
    y = Tensor.from_numpy(y)
    expand_dims = ops.ExpandDims()
    y = expand_dims(expand_dims(y, 0), 0)
    preds = model.predict(y)
    preds = preds.asnumpy()
    psnr = calc_psnr(y.asnumpy(), preds)
    psnr = psnr.item(0)
    print("PSNR: %.4f" % psnr)
    preds = np.multiply(preds, 255.0)
    preds = preds.squeeze(0).squeeze(0)
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(image_path.replace('.', '_srcnn_x{}.'.format(config.scale)))
    if cfg.enable_modelarts == "True":
        sync_data(image_path.replace('.', '_srcnn_x{}.'.format(config.scale)), \
            (cfg.data_url + cfg.test_pic).replace('.', '_srcnn_x{}.'.format(config.scale)))
        sync_data(image_path.replace('.', '_bicubic_x{}.'.format(config.scale)), \
            (cfg.data_url + cfg.test_pic).replace('.', '_bicubic_x{}.'.format(config.scale)))

if __name__ == '__main__':
    run_test()
    