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
# ===========================================================================

"""
    Evaluate Pix2Pix Model.
"""

import os
import mindspore as ms
from mindspore import Tensor, nn
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net
from src.dataset.pix2pix_dataset import pix2pixDataset_val, create_val_dataset
from src.models.pix2pix import Pix2Pix, get_generator, get_discriminator
from src.models.loss import D_Loss, D_WithLossCell, G_Loss, G_WithLossCell, TrainOneStepCell
from src.utils.tools import save_image, get_lr
from src.utils.config import config
from src.utils.moxing_adapter import moxing_wrapper
from src.utils.device_adapter import get_device_id

@moxing_wrapper()
def pix2pix_eval():

    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())

    # Preprocess the data for evaluating
    dataset_val = pix2pixDataset_val(root_dir=config.val_data_dir)
    ds_val = create_val_dataset(dataset_val)
    print("ds:", ds_val.get_dataset_size())
    print("ds:", ds_val.get_col_names())
    print("ds.shape:", ds_val.output_shapes())

    netG = get_generator()
    netD = get_discriminator()

    pix2pix = Pix2Pix(generator=netG, discriminator=netD)

    d_loss_fn = D_Loss()
    g_loss_fn = G_Loss()
    d_loss_net = D_WithLossCell(backbone=pix2pix, loss_fn=d_loss_fn)
    g_loss_net = G_WithLossCell(backbone=pix2pix, loss_fn=g_loss_fn)

    d_opt = nn.Adam(pix2pix.netD.trainable_params(), learning_rate=get_lr(),
                    beta1=config.beta1, beta2=config.beta2, loss_scale=1)
    g_opt = nn.Adam(pix2pix.netG.trainable_params(), learning_rate=get_lr(),
                    beta1=config.beta1, beta2=config.beta2, loss_scale=1)

    train_net = TrainOneStepCell(loss_netD=d_loss_net, loss_netG=g_loss_net, optimizerD=d_opt, optimizerG=g_opt, sens=1)
    train_net.set_train()

    # Evaluating loop
    ckpt_url = config.ckpt
    print("CKPT:", ckpt_url)
    param_G = load_checkpoint(ckpt_url)
    load_param_into_net(netG, param_G)

    if not os.path.isdir(config.predict_dir):
        os.makedirs(config.predict_dir)

    data_loader_val = ds_val.create_dict_iterator(output_numpy=True, num_epochs=config.epoch_num)
    print("=======Starting evaluating Loop=======")
    for i, data in enumerate(data_loader_val):
        input_image = Tensor(data["input_images"])
        fake_image = netG(input_image)
        save_image(fake_image, config.predict_dir + str(i + 1))
        print("=======image", i + 1, "saved success=======")

if __name__ == '__main__':
    pix2pix_eval()
