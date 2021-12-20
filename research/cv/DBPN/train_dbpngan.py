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

"""gan network training script for decrease noise.
You need to specify the dataset ('--train_GT_path','val_LR_path','val_GT_path')
Example:
    Train a DBPN model:
       python train_dbpngan.py --device_id=0 model_type="DBPN" --train_GT_path="/data/DBPN_data/DIV2K_train_HR"
       --val_LR_path="/data/DBPN_data/Set5/LR" --val_GT_path="/data/DBPN_data/Set5/HR"
"""
import os
import time

import mindspore
from mindspore import nn, load_checkpoint, save_checkpoint, load_param_into_net, context
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from mindspore.nn.dynamic_lr import piecewise_constant_lr

from src.dataset.dataset import DBPNDataset, DatasetVal, create_train_dataset, create_val_dataset
from src.loss.withlosscell import WithLossCellPretrainedG, WithLossCellD, WithLossCellG
from src.model.discriminator import get_discriminator
from src.model.generator import get_generator
from src.util.config import get_args
from src.util.utils import save_losses, denorm, save_img, compute_psnr, save_psnr

args = get_args(is_gan=True)

print(args)
mindspore.set_seed(args.seed)

G_losses = []
D_losses = []
best_psnr = 0
eval_psnr = []

# create save eval folder
save_eval_path = os.path.join(args.Results, args.valDataset, args.model_type)
if not os.path.exists(save_eval_path):
    os.makedirs(save_eval_path)

save_loss_path = 'results/ganloss/'
if not os.path.exists(save_loss_path):
    os.makedirs(save_loss_path)


class DBPNGAN(nn.Cell):
    """
    DBPNGAN network
    Args:
        trainOneStepCellForD(Cell):
        trainOneStepCellForG(Cell):
    return:
        d_loss(Tensor): the loss of netD
        g_loss(Tensor): the loss of netG
    """
    def __init__(self, trainOneStepCellForD, trainOneStepCellForG):
        super(DBPNGAN, self).__init__(auto_prefix=True)
        self.myTrainOneStepCellForD = trainOneStepCellForD
        self.myTrainOneStepCellForG = trainOneStepCellForG

    def construct(self, hr_img, lr_img):
        """compute loss of D and G"""
        output_D = self.myTrainOneStepCellForD(hr_img, lr_img).view(-1)
        d_loss = output_D.mean()
        output_G = self.myTrainOneStepCellForG(hr_img, lr_img).view(-1)
        g_loss = output_G.mean()
        return d_loss, g_loss


def pretrain_netG(net, ds):
    """use mseloss to pretrain the DBPN network"""
    epochs = 100
    ds_steps = ds.get_dataset_size()
    learning_rate = 1e-4
    optim = nn.Adam(params=net.train_params(), learning_rate=learning_rate, loss_scale=args.sens)
    pretrain_G = WithLossCellPretrainedG(net)
    myPretrainOneStepCellFroG = nn.TrainOneStepCell(pretrain_G, optim, sens=args.sens)
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        for data in enumerate(ds.create_dict_iterator()):
            hr_img = data['target_image']
            lr_img = data['input_image']
            loss = myPretrainOneStepCellFroG(hr_img, lr_img)
            epoch_loss += loss
            G_losses.append(loss)
        mean = epoch_loss.asnumpy() / ds_steps
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, mean))
        pname = os.path.join(save_loss_path, args.valDataset + '_' + args.model_type)
        save_losses(G_losses, D_losses, pname)
    print('Pre-training finished.')
    save_path = os.path.join(args.save_folder, args.pretrained_dbpn)
    save_checkpoint(net, save_path)
    print('the checkpoint has been saved!')


def predict(net, ds):
    """predict use eval dataset"""
    global best_psnr
    sum_psnr = 0
    val_steps = ds.get_dataset_size()
    val_data_loader = ds.create_dict_iterator()
    t0 = time.time()
    for i, data in enumerate(val_data_loader):
        hr_img = data['target_image']
        lr_img = data['input_image']
        sr = net(lr_img)
        prediction = denorm(sr[0], args.vgg)
        save_img(prediction, str(i), save_eval_path)
        target = denorm(hr_img[0], args.vgg)
        s_psnr = compute_psnr(target, prediction)
        sum_psnr += s_psnr
        print("===> Processing:{} || compute_psnr :{:.4f}".format(i, s_psnr))
    t1 = time.time()
    mean = sum_psnr / val_steps
    print(" Avg_psnr:{:.4f} ||Timer:{:.2f} min".format(mean, int((t1 - t0) / 60)))
    eval_psnr.append(mean)
    savepath = "result/ganloss/{}-{}-psnr.png".format(args.valDataset, args.model_type)
    save_psnr(eval_psnr, savepath, args.model_type)
    if best_psnr < mean:
        best_psnr = mean
        save_checkpoint(netG, os.path.join(args.save_folder, 'best_G.ckpt'))


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.run_distribute:
        print("distribute")
        device_id = int(os.getenv("DEVICE_ID"))
        device_num = args.device_num
        context.set_context(device_id=device_id)
        init()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        rank = get_rank()
    else:
        device_id = args.device_id
        context.set_context(device_id=device_id)
    # create datasets
    train_dataset = DBPNDataset(args.train_GT_path, args)
    train_ds = create_train_dataset(train_dataset, args)
    train_data_loader = train_ds.create_dict_iterator()
    train_steps = train_ds.get_dataset_size()

    val_dataset = DatasetVal(args.val_GT_path, args.val_LR_path, args)
    val_ds = create_val_dataset(val_dataset, args)

    print('===> Building model ', args.model_type)
    netG = get_generator(args.model_type, scale_factor=args.upscale_factor)
    print(netG)
    netD = get_discriminator(3, 64, args.patch_size * args.upscale_factor, 0.02)
    print(netD)

    if args.pretrained:
        pretrain_netG(netG, train_ds)

    if args.load_pretrained_D:
        ckpt = os.path.join(args.save_folder, args.pretrained_D)
        print('=====> load params into discriminator')
        params = load_checkpoint(ckpt)
        load_param_into_net(netD, params)
        print('=====> finish load discriminator')

    if args.load_pretrained_G:
        ckpt = os.path.join(args.save_folder, args.pretrained_dbpn)
        print('=====> load params into generator')
        params = load_checkpoint(ckpt)
        load_param_into_net(netG, params)
        print('=====> finish load generator')

    milestone = [int(args.nEpochs / 2) * train_steps, int(args.nEpochs + 1) * train_steps]
    learning_rates = [args.lr, args.lr / 10.0]
    lr = piecewise_constant_lr(milestone, learning_rates)

    optimizerG = nn.Adam(netG.trainable_params(), learning_rate=lr, loss_scale=args.sens)
    optimizerD = nn.Adam(netD.trainable_params(), learning_rate=lr, loss_scale=args.sens)

    netD_with_criterion = WithLossCellD(netD, netG)
    netG_with_criterion = WithLossCellG(netD, netG)

    myTrainOneStepCellForD = nn.TrainOneStepCell(netD_with_criterion, optimizerD, sens=args.sens)
    myTrainOneStepCellForG = nn.TrainOneStepCell(netG_with_criterion, optimizerG, sens=args.sens)

    dbpngan = DBPNGAN(myTrainOneStepCellForD, myTrainOneStepCellForG)
    dbpngan.set_train()

    print("Starting Training Loop...")
    num_epochs = args.nEpochs
    size = train_steps
    iters = 0
    for num in range(num_epochs):
        start = time.time()
        for idx, d in enumerate(train_data_loader, 1):
            hr = d['target_image']
            lr = d['input_image']
            netD_loss, netG_loss = dbpngan(hr, lr)
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' %
                  (num + 1, num_epochs, idx, size, netD_loss.asnumpy(), netG_loss.asnumpy()))
            D_losses.append(netD_loss.asnumpy())
            G_losses.append(netG_loss.asnumpy())
        end = time.time()
        step_time = (end - start) / size
        print("===>Epoch {} Complete, per step time: {}ms.".format(num+1, step_time*1000))
        name = os.path.join(save_loss_path, args.valDataset + '_' + args.model_type)
        save_losses(G_losses, D_losses, name)
        if args.eval_flag:
            predict(netG, val_ds)
