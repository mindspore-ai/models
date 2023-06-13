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
# ===========================================================================

"""
    Train pix2pixHD model including distributed training
"""

import os
import time
import mindspore as ms
import mindspore.nn as nn
from src.models.loss import TrainOneStepCell, DWithLossCell, GWithLossCell
from src.models.pix2pixHD import Pix2PixHD
from src.dataset.pix2pixHD_dataset import Pix2PixHDDataset, create_train_dataset
from src.utils.tools import save_losses, save_image, get_lr, save_network
from src.utils.config import config
from src.utils.local_adapter import init_env, get_rank_id

ms.set_seed(1)


def train():
    # init env
    init_env()
    # Preprocess the data for training
    dataset = Pix2PixHDDataset(root_dir=config.data_root)
    ds = create_train_dataset(dataset, config.batch_size, config.run_distribute)
    steps_per_epoch = ds.get_dataset_size()
    pix2pixHD = Pix2PixHD()
    loss_netD = DWithLossCell(backbone=pix2pixHD)
    loss_netG = GWithLossCell(backbone=pix2pixHD)
    lr = get_lr(steps_per_epoch)

    d_opt = nn.Adam(
        pix2pixHD.trainable_params_D, learning_rate=lr, beta1=config.beta1, beta2=config.beta2, loss_scale=1
    )
    g_opt = nn.Adam(
        pix2pixHD.trainable_params_G, learning_rate=lr, beta1=config.beta1, beta2=config.beta2, loss_scale=1
    )

    train_net = TrainOneStepCell(loss_netD=loss_netD, loss_netG=loss_netG, optimizerD=d_opt, optimizerG=g_opt, sens=1)
    train_net.set_train()
    # fake image saved path
    is_save = not config.run_distribute or (config.run_distribute and get_rank_id() == 0)
    if is_save:
        fake_image_path = os.path.join(config.save_ckpt_dir, config.name, "fake_image")
        if not os.path.exists(fake_image_path):
            os.makedirs(fake_image_path)
        # loss saved path
        loss_saved_path = os.path.join(config.save_ckpt_dir, config.name, "loss_show")
        if not os.path.exists(loss_saved_path):
            os.makedirs(loss_saved_path)

        # ckpt saved path
        ckpt_saved_path = os.path.join(config.save_ckpt_dir, config.name)
        if not os.path.exists(ckpt_saved_path):
            os.makedirs(ckpt_saved_path)

    # Training loop
    G_losses = []
    D_losses = []

    data_loader = ds.create_dict_iterator(output_numpy=True)
    print("Starting Training Loop...")
    for epoch in range(config.niter + config.niter_decay):
        for i, data in enumerate(data_loader):
            start_time = time.time()
            label = data["label"]
            inst = data["inst"]
            image = data["image"]
            feat = data["feat"]

            label, inst, image, feat = pix2pixHD.encode_input(label, inst, image, feat)

            dis_loss, gen_loss = train_net(label, inst, image, feat)
            end_time = time.time()
            delta = (end_time - start_time) * 1000
            if i % 10 == 0:
                print("================start===================")
                print("Date time: ", start_time)
                print("ms per step :", delta)
                print("epoch: ", epoch + 1, "/", config.niter + config.niter_decay)
                print("step: ", i + 1, "/", steps_per_epoch)
                print("Dloss: ", dis_loss)
                print("Gloss: ", gen_loss)
                print("=================end====================")

            # Save fake_imgs
            if i == steps_per_epoch - 1:
                if is_save:
                    fake_image = pix2pixHD(label, inst, image, feat)
                    save_image(fake_image, fake_image_path + "/" + str(epoch + 1))
                    print("image generated from epoch", epoch + 1, "saved")

            D_losses.append(dis_loss.asnumpy())
            G_losses.append(gen_loss.asnumpy())

        print("epoch", epoch + 1, "saved")
        # Save losses
        if is_save:
            save_losses(loss_saved_path, G_losses, D_losses, epoch + 1)
            print("epoch", epoch + 1, "D&G_Losses saved")
            print("epoch", epoch + 1, "finished")

        # Save checkpoint
        if (epoch + 1) % 50 == 0 or (epoch + 1 == config.niter + config.niter_decay):
            if is_save:
                save_checkpoint_path = os.path.join(config.save_ckpt_dir, config.name)
                save_network(pix2pixHD, save_checkpoint_path, epoch + 1)
                print("ckpt generated from epoch", epoch + 1, "saved")

        # instead of only training the local enhancer, train the entire network after certain iterations
        if (config.niter_fix_global != 0) and (epoch + 1 == config.niter_fix_global):
            pix2pixHD.reset_netG_grads(True)
            g_opt_new = nn.Adam(
                pix2pixHD.trainable_params_G,
                learning_rate=lr[epoch:],
                beta1=config.beta1,
                beta2=config.beta2,
                loss_scale=1,
            )
            train_net.update_optimizerG(g_opt_new)


if __name__ == "__main__":
    train()
