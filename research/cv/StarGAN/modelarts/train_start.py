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
"""Train the model."""
from time import time
import os
import argparse
import ast
import glob
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore import Tensor, context
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, _InternalCallbackParam, RunContext
from mindspore.train.serialization import export, load_param_into_net
from src.dataset import dataloader
from src.utils import get_network, resume_model
from src.cell import TrainOneStepCellGen, TrainOneStepCellDis
from src.loss import GeneratorLoss, DiscriminatorLoss, ClassificationLoss, WGANGPGradientPenalty
from src.reporter import Reporter

set_seed(1)

# Modelarts
parser = argparse.ArgumentParser(description='StarGAN_args')
parser.add_argument('--modelarts', type=ast.literal_eval, default=True, help='Dataset path')
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')
# Model configuration.
parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
parser.add_argument('--c2_dim', type=int, default=7, help='dimension of domain labels (2nd dataset)')
parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
parser.add_argument('--image_size', type=int, default=128, help='image resolution')
parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
# Training configuration.
parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
parser.add_argument('--batch_size', type=int, default=4, help='mini-batch size')
parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
parser.add_argument('--epochs', type=int, default=59, help='number of epoch')
parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--resume_iters', type=int, default=200000, help='resume training from this step')
parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                    default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
parser.add_argument('--init_type', type=str, default='normal', choices=("normal", "xavier"),
                    help='network initialization, default is normal.')
parser.add_argument('--init_gain', type=float, default=0.02,
                    help='scaling factor for normal, xavier and orthogonal, default is 0.02.')
# Test configuration.
parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')
# Train Device.
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--device_target', type=str, default='Ascend')
parser.add_argument("--run_distribute", type=int, default=0, help="Run distribute, default: false.")
parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")
parser.add_argument("--device_num", type=int, default=1, help="number of device, default: 0.")
parser.add_argument("--rank_id", type=int, default=0, help="Rank id, default: 0.")
# Directories.
parser.add_argument('--celeba_image_dir', type=str, default=r'/home/data/celeba/images')
parser.add_argument('--attr_path', type=str, default=r'/home/data/celeba/list_attr_celeba.txt')
parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
parser.add_argument('--log_dir', type=str, default='stargan/logs')
parser.add_argument('--model_save_dir', type=str, default='./models/')
parser.add_argument('--result_dir', type=str, default='./results')
# Step size.
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--sample_step', type=int, default=5000)
parser.add_argument('--model_save_step', type=int, default=5000)
parser.add_argument('--lr_update_step', type=int, default=1000)
# export
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='AIR', \
                    help='file format')
#args_opt = parser.parse_args()
args_opt, unparsed = parser.parse_known_args()
config = args_opt

if __name__ == '__main__':

    #config = get_config()
    if args_opt.modelarts:
        import moxing as mox
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
        context.set_context(device_id=device_id)
        local_data_url = './cache/data'
        local_train_url = '/cache/ckpt'
        if not os.path.isdir(local_data_url):
            os.makedirs(local_data_url)
        if not os.path.isdir(local_train_url):
            os.makedirs(local_train_url)

        # local_data_url = os.path.join(local_data_url, str(device_id))
        # local_train_url = os.path.join(local_train_url, str(device_id))

        # unzip data
        path = os.getcwd()
        print("cwd: %s" % path)
        data_url = 'obs://data/CelebA/'

        data_name = '/celeba.zip'
        print('listdir1: %s' % os.listdir('./'))

        a1time = time()
        mox.file.copy_parallel(args_opt.data_url, local_data_url)
        print('listdir2: %s' % os.listdir(local_data_url))
        b1time = time()
        print('time1:', b1time - a1time)

        a2time = time()
        zip_command = "unzip -o %s -d %s" % (local_data_url + data_name, local_data_url)
        if os.system(zip_command) == 0:
            print('Successful backup')
        else:
            print('FAILED backup')
        b2time = time()
        print('time2:', b2time - a2time)
        print('listdir3: %s' % os.listdir(local_data_url))

        # Device Environment
        if config.run_distribute:
            if config.device_target == "Ascend":
                rank = device_id
                # device_num = device_num
                context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                                  gradients_mean=True)
                init()
        else:
            rank = 0
            device_num = 1

        data_path = local_data_url + '/celeba/images'
        attr_path = local_data_url + '/celeba/list_attr_celeba.txt'
        dataset, length = dataloader(img_path=data_path,
                                     attr_path=attr_path,
                                     batch_size=config.batch_size,
                                     selected_attr=config.selected_attrs,
                                     device_num=config.num_workers,
                                     dataset=config.dataset,
                                     mode=config.mode,
                                     shuffle=True)


    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target,
                            device_id=config.device_id, save_graphs=False)
        if args_opt.run_distribute:
            if os.getenv("DEVICE_ID", "not_set").isdigit():
                context.set_context(device_id=int(os.getenv("DEVICE_ID")))
            device_num = config.device_num
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                              device_num=device_num)
            init()

            rank = get_rank()

        data_path = config.celeba_image_dir
        attr_path = config.attr_path
        local_train_url = config.model_save_dir
        dataset, length = dataloader(img_path=data_path,
                                     attr_path=attr_path,
                                     batch_size=config.batch_size,
                                     selected_attr=config.selected_attrs,
                                     device_num=config.device_num,
                                     dataset=config.dataset,
                                     mode=config.mode,
                                     shuffle=True)
    print(length)
    dataset_iter = dataset.create_dict_iterator()

    # Get and initial network
    generator, discriminator = get_network(config)

    cls_loss = ClassificationLoss()
    wgan_loss = WGANGPGradientPenalty(discriminator)

    # Define network with loss
    G_loss_cell = GeneratorLoss(config, generator, discriminator)
    D_loss_cell = DiscriminatorLoss(config, generator, discriminator)

    # Define Optimizer
    star_iter = 0
    iter_sum = config.num_iters

    Optimizer_G = nn.Adam(generator.trainable_params(), learning_rate=config.g_lr,
                          beta1=config.beta1, beta2=config.beta2)
    Optimizer_D = nn.Adam(discriminator.trainable_params(), learning_rate=config.d_lr,
                          beta1=config.beta1, beta2=config.beta2)

    # Define One step train
    G_trainOneStep = TrainOneStepCellGen(G_loss_cell, Optimizer_G)
    D_trainOneStep = TrainOneStepCellDis(D_loss_cell, Optimizer_D)

    # Train
    G_trainOneStep.set_train()
    D_trainOneStep.set_train()

    print('Start Training')

    reporter = Reporter(config)

    ckpt_config = CheckpointConfig(save_checkpoint_steps=config.model_save_step)
    ckpt_cb_g = ModelCheckpoint(config=ckpt_config, directory=local_train_url, prefix='Generator')
    ckpt_cb_d = ModelCheckpoint(config=ckpt_config, directory=local_train_url, prefix='Discriminator')

    cb_params_g = _InternalCallbackParam()
    cb_params_g.train_network = generator
    cb_params_g.cur_step_num = 0
    cb_params_g.batch_num = 4
    cb_params_g.cur_epoch_num = 0

    cb_params_d = _InternalCallbackParam()
    cb_params_d.train_network = discriminator
    cb_params_d.cur_step_num = 0
    cb_params_d.batch_num = config.batch_size
    cb_params_d.cur_epoch_num = 0
    run_context_g = RunContext(cb_params_g)
    run_context_d = RunContext(cb_params_d)
    ckpt_cb_g.begin(run_context_g)
    ckpt_cb_d.begin(run_context_d)
    start = time()

    for iterator in range(config.num_iters):
        data = next(dataset_iter)
        x_real = Tensor(data['image'], mstype.float32)
        c_trg = Tensor(data['attr'], mstype.float32)
        c_org = Tensor(data['attr'], mstype.float32)
        np.random.shuffle(c_trg)

        d_out = D_trainOneStep(x_real, c_org, c_trg)

        if (iterator + 1) % config.n_critic == 0:
            g_out = G_trainOneStep(x_real, c_org, c_trg)

        if (iterator + 1) % config.log_step == 0:
            reporter.print_info(start, iterator, g_out, d_out)
            _, _, dict_G, dict_D = reporter.return_loss_array(g_out, d_out)

        if (iterator + 1) % config.model_save_step == 0:
            cb_params_d.cur_step_num = iterator + 1
            cb_params_d.batch_num = iterator + 2
            cb_params_g.cur_step_num = iterator + 1
            cb_params_g.batch_num = iterator + 2
            ckpt_cb_g.step_end(run_context_g)
            ckpt_cb_d.step_end(run_context_d)

    if args_opt.modelarts:
        print('output dir3: %s' % os.listdir(local_train_url))
        ckpt_list = glob.glob("/cache/ckpt/*.ckpt")
        if not ckpt_list:
            print("ckpt file not generated.")

        ckpt_list.sort(key=os.path.getmtime)
        ckpt_model = ckpt_list[-1]
        print("checkpoint path", ckpt_model)
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
        G, D = get_network(config)
        # Use BatchNorm2d with batchsize=1, affine=False, training=True instead of InstanceNorm2d
        # Use real mean and varance rather than moving_men and moving_varance in BatchNorm2d
        G.set_train(True)
        param_G, _ = resume_model(config, G, D)
        load_param_into_net(G, param_G)
        input_array = Tensor(np.random.uniform(-1.0, 1.0, size=(1, 3, 128, 128)).astype(np.float32))
        input_label = Tensor(np.random.uniform(-1.0, 1.0, size=(1, 5)).astype(np.float32))
        G_file = f"StarGAN_Generator"
        export(G, input_array, input_label, file_name='/cache/ckpt/stargan', file_format=config.file_format)

        mox.file.copy_parallel(local_train_url, args_opt.train_url)
