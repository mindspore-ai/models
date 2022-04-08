# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
    export checkpoint file into air, onnx, mindir models
"""
import numpy as np
from mindspore import Tensor, nn, context
from mindspore.train.serialization import export
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net
from src.models.pix2pix import Pix2Pix, get_generator, get_discriminator
from src.models.loss import D_Loss, D_WithLossCell, G_Loss, G_WithLossCell, TrainOneStepCell
from src.utils.tools import get_lr
from src.utils.config import config
from src.utils.moxing_adapter import moxing_wrapper

@moxing_wrapper()
def export_pix2pix():

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=config.device_id)

    netG = get_generator()
    netD = get_discriminator()

    pix2pix = Pix2Pix(generator=netG, discriminator=netD)

    d_loss_fn = D_Loss()
    g_loss_fn = G_Loss()
    d_loss_net = D_WithLossCell(backbone=pix2pix, loss_fn=d_loss_fn)
    g_loss_net = G_WithLossCell(backbone=pix2pix, loss_fn=g_loss_fn)

    d_opt = nn.Adam(pix2pix.netD.trainable_params(), learning_rate=get_lr(), beta1=0.5, beta2=0.999, loss_scale=1)
    g_opt = nn.Adam(pix2pix.netG.trainable_params(), learning_rate=get_lr(), beta1=0.5, beta2=0.999, loss_scale=1)

    train_net = TrainOneStepCell(loss_netD=d_loss_net, loss_netG=g_loss_net, optimizerD=d_opt, optimizerG=g_opt, sens=1)
    train_net.set_train()
    train_net = train_net.loss_netG

    ckpt_url = config.ckpt
    param_G = load_checkpoint(ckpt_url)
    load_param_into_net(netG, param_G)

    input_shp = [config.batch_size, 3, config.image_size, config.image_size]
    input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))
    target_shp = [config.batch_size, 3, config.image_size, config.image_size]
    target_array = Tensor(np.random.uniform(-1.0, 1.0, size=target_shp).astype(np.float32))
    inputs = [input_array, target_array]
    file = f"{config.file_name}"
    export(train_net, *inputs, file_name=file, file_format=config.file_format)

if __name__ == '__main__':
    export_pix2pix()
