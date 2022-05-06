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
'''train_model'''
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from src.dataset import create_dataset_train, create_dataset_train_dis
from src.gan import Generator, Discriminator
from src.gan import GenWithLossCell, DisWithLossCell, TrainOneStepCell
from src.param_parse import parameter_parser

from mindspore import nn
from mindspore import ops
from mindspore import export, load_checkpoint, load_param_into_net, Tensor, context
from mindspore.common import dtype as mstype
from mindspore.common import set_seed
from mindspore import save_checkpoint
from mindspore.communication import init
from mindspore.context import ParallelMode

os.makedirs("images", exist_ok=True)


def save_imgs(gen_imgs1, idx):
    for i3 in range(gen_imgs1.shape[0]):
        plt.subplot(100, 100, i3 + 1)
        plt.imshow(gen_imgs1[i3, 0, :, :] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")
    plt.savefig("./images/{}.png".format(idx))


reshape = ops.Reshape()

def export_modelarts(args):
    '''export the air file'''
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    # Create network
    generator = Generator(args.latent_dim)

    # Initialize dummy inputs
    test_latent_code_parzen = Tensor(np.random.normal(size=(1, args.latent_dim)), dtype=mstype.float32)

    # Load parameters from checkpoint into network
    ckpt_file = str(args.n_epochs - 1) + ".ckpt"
    param_dict = load_checkpoint(os.path.join(args.ckpt_path, ckpt_file))
    load_param_into_net(generator, param_dict)

    # Export network into MINDIR model file
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    path = os.path.join(args.ckpt_path, ckpt_file)
    export(generator, test_latent_code_parzen, file_name=path, file_format='AIR')
    print("==========================================")
    print(ckpt_file + ".air exported successfully!")
    print("==========================================")

def main():
    opt = parameter_parser()
    print(opt)
    set_seed(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    if opt.distribute:
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=device_id)
        init()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        dataset = create_dataset_train_dis(batch_size=opt.batch_size, repeat_size=1, latent_size=opt.latent_dim)

    else:
        device_id = opt.device_id
        context.set_context(device_id=device_id)
        dataset = create_dataset_train(batch_size=opt.batch_size, repeat_size=1, latent_size=opt.latent_dim)

    adversarial_loss = ops.SigmoidCrossEntropyWithLogits()

    generator = Generator(opt.latent_dim)
    discriminator = Discriminator()
    netG_with_loss = GenWithLossCell(generator, discriminator, adversarial_loss)
    netD_with_loss = DisWithLossCell(generator, discriminator, adversarial_loss)
    optimizerG = nn.Adam(generator.trainable_params(), opt.lr, beta1=0.5, beta2=0.999)
    optimizerD = nn.Adam(discriminator.trainable_params(), opt.lr, beta1=0.5, beta2=0.999)
    net_train = TrainOneStepCell(netG_with_loss, netD_with_loss, optimizerG,
                                 optimizerD)

    generator.set_train()
    discriminator.set_train()
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(opt.n_epochs):
        start = time.time()
        for (i, imgs) in enumerate(dataset):
            image = imgs[0]
            image = (image - 127.5) / 127.5
            image = reshape(image, (image.shape[0], 1, image.shape[1], image.shape[2]))
            latent_code = imgs[1]
            _, _ = net_train(image, latent_code)
            print("epoch: {} ,batch: {}".format(epoch, i))

        t = time.time() - start
        print("time of epoch {} is {:.2f}s".format(epoch, t))
        save_checkpoint(save_obj=generator, ckpt_file_name=os.path.join(opt.ckpt_path, str(epoch) + '.ckpt'))
    export_modelarts(opt)

if __name__ == '__main__':
    main()
