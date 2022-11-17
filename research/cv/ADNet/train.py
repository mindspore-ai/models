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
import os
import argparse
import ast

from src.trainers.adnet_train_sl import adnet_train_sl
from src.options.general import opts
from src.models.ADNet import adnet
from src.utils.get_train_videos import get_train_videos
from src.trainers.adnet_train_rl import adnet_train_rl


from mindspore import context
from mindspore.communication.management import init
from mindspore.context import ParallelMode



parser = argparse.ArgumentParser(
    description='ADNet training')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'])
parser.add_argument('--target_device', type=int, default=0)
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--visualize', default=False, type=ast.literal_eval,
                    help='Use tensorboardx to for loss visualization')
parser.add_argument('--send_images_to_visualization', type=ast.literal_eval, default=False,
                    help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder', default='../weights', help='Location to save checkpoint models')

parser.add_argument('--save_file', default='ADNet_SL_', type=str, help='save file part of file name for SL')
parser.add_argument('--save_domain_dir', default='domain_weights', type=str, help='save ckpt from domain')
parser.add_argument('--save_file_RL', default='ADNet_RL_', type=str, help='save file part of file name for RL')
parser.add_argument('--start_epoch', default=0, type=int, help='Begin counting epochs starting from this value')

parser.add_argument('--run_supervised', default=True,
                    type=ast.literal_eval, help='Whether to run supervised learning or not')

parser.add_argument('--multidomain', default=True, type=ast.literal_eval,
                    help='Separating weight for each videos (default) or not')

parser.add_argument('--save_result_images', default=True, type=ast.literal_eval,
                    help='Whether to save the results or not. Save folder: images/')
parser.add_argument('--display_images', default=False, type=ast.literal_eval, help='Whether to display images or not')
parser.add_argument('--distributed', type=ast.literal_eval, default=False)
parser.add_argument('--run_online', type=str, default='False')
parser.add_argument('--data_url', type=str)
parser.add_argument('--train_url', type=str)
parser.add_argument('--save_path', type=str, default='')
parser.add_argument('--dataset_path', type=str, default='')

args = parser.parse_args()
if args.run_online == 'True':
    import moxing as mox
    local_data_url = "/cache/data"
    args.dataset_path = local_data_url
    # move dataset path
    mox.file.copy_parallel(args.data_url, local_data_url)
    args.save_path = '/cache/train_out'
    args.save_folder = 'weights'

if args.distributed:
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=int(os.environ["DEVICE_ID"]))
    init()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True)
else:
    context.set_context(device_target=args.device_target, mode=context.GRAPH_MODE, device_id=args.target_device)
# Supervised Learning part
if args.run_supervised:
    opts['minibatch_size'] = 128
    # train with supervised learning
    if args.run_online == 'True':
        save_path = '/cache/train_out'
        if args.resume is not None:
            import moxing
            local_weight = '/cache/weight/' + args.resume.split('/')[-1]
            #moving ckpt
            moxing.file.copy_parallel(args.resume, local_weight)
            #moving multidomain
            if not os.path.exists("/cache/weight/domain_weights/"):
                os.makedirs("/cache/weight/domain_weights/")
            moxing.file.copy_parallel(args.resume[:args.resume.rfind('/')]
                                      + '/domain_weights/', "/cache/weight/domain_weights/")
            args.resume = local_weight
    else:
        save_path = ''
    dir_path = os.path.join(args.save_path, args.save_folder, args.save_domain_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    _, _, train_videos = adnet_train_sl(args, opts)

    args.resume = os.path.join(save_path, args.save_folder, args.save_file) + 'final.ckpt'


    # reinitialize the network with network from SL
    net, domain_specific_nets = adnet(opts, trained_file=args.resume,
                                      random_initialize_domain_specific=True,
                                      multidomain=args.multidomain,
                                      distributed=args.distributed,
                                      run_online=args.run_online)

    args.start_epoch = 0
    args.start_iter = 0

else:
    assert args.resume is not None, \
        "Please put result of supervised learning or reinforcement learning with --resume (filename)"
    if args.run_online == 'True':
        import moxing
        local_data_url = "/cache/data"
        # move dataset path
        args.dataset_path = local_data_url
        moxing.file.copy_parallel(args.data_url, local_data_url)
        local_weight_url = "/cache/weight/" + args.resume.split('/')[-1]
        # moving ckpt
        moxing.file.copy_parallel(args.resume, local_weight_url)
        args.resume = local_weight_url
    train_videos = get_train_videos(opts, args)
    opts['num_videos'] = len(train_videos['video_names'])

    if args.start_iter == 0:  # means the weight came from the SL
        net, domain_specific_nets = adnet(opts, trained_file=args.resume,
                                          random_initialize_domain_specific=True,
                                          multidomain=args.multidomain,
                                          distributed=args.distributed,
                                          run_online=args.run_online)
    else:  # resume the adnet
        net, domain_specific_nets = adnet(opts, trained_file=args.resume,
                                          random_initialize_domain_specific=False,
                                          multidomain=args.multidomain,
                                          distributed=args.distributed,
                                          run_online=args.run_online)

# Reinforcement Learning part
opts['minibatch_size'] = 32

net = adnet_train_rl(net, domain_specific_nets, train_videos, opts, args)
