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
import argparse
import os
import ast

from src.options.general import opts
from src.models.ADNet import adnet
from src.trainers.adnet_test import adnet_test

from mindspore import context
from mindspore.communication.management import init
from mindspore.context import ParallelMode

parser = argparse.ArgumentParser(
    description='ADNet test')
parser.add_argument('--weight_file', default='weights/ADNet_RL_.pth', type=str, help='The pretrained weight file')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
parser.add_argument('--visualize', default=False, type=ast.literal_eval, help='Use tensorboardx to for visualization')
parser.add_argument('--send_images_to_visualization', type=ast.literal_eval,
                    default=False, help='visdom after augmentations')
parser.add_argument('--display_images', default=False, type=ast.literal_eval, help='Whether to display images or not')
parser.add_argument('--save_result_images', default='', type=str, help='save results folder')
parser.add_argument('--save_result_npy', default='../results_on_test_images_part2',
                    type=str, help='save results folder')
parser.add_argument('--initial_samples', default=3000, type=int, help='Num of training samples for the first frame.')
parser.add_argument('--online_samples', default=250, type=int, help='Num of training samples for the other frames.')
parser.add_argument('--redetection_samples', default=256, type=int, help='Num of samples for redetection.')
parser.add_argument('--initial_iteration', default=300, type=int, help='Number of iteration in initial training. T_I')
parser.add_argument('--online_iteration', default=30, type=int, help='Number of iteration in online training. T_O')
parser.add_argument('--online_adaptation_every_I_frames', default=10, type=int, help='Frequency of online training. I')

parser.add_argument('--believe_score_result', default=0, type=int, help='Believe score result after n training')

parser.add_argument('--pos_samples_ratio', default='0.5', type=float,
                    help='''The ratio of positive in all samples for online adaptation.
                         Rest of it will be negative samples. Default: 0.5''')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'])
parser.add_argument('--target_device', type=int, default=0)
parser.add_argument('--distributed', type=ast.literal_eval, default=False)
parser.add_argument('--multidomain', type=ast.literal_eval, default=True)
parser.add_argument('--run_online', type=str, default='False')
parser.add_argument('--data_url', type=str)
parser.add_argument('--train_url', type=str)
parser.add_argument('--dataset_path', type=str, default='')

args = parser.parse_args()
if args.distributed:
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=int(os.environ["DEVICE_ID"]))
    init()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True)
else:
    context.set_context(device_target=args.device_target, mode=context.GRAPH_MODE, device_id=args.target_device)
assert 0 < args.pos_samples_ratio <= 1, "the pos_samples_ratio valid range is (0, 1]"

# set opts based on the args.. especially the number of samples etc.
opts['nPos_init'] = int(args.initial_samples * args.pos_samples_ratio)
opts['nNeg_init'] = int(args.initial_samples - opts['nPos_init'])
opts['nPos_online'] = int(args.online_samples * args.pos_samples_ratio)
opts['nNeg_online'] = int(args.online_samples - opts['nPos_online'])

# just to make sure if one of nNeg is zero, the other nNeg is zero (kinda small hack...)
if opts['nNeg_init'] == 0:
    opts['nNeg_online'] = 0
    opts['nPos_online'] = args.online_samples

elif opts['nNeg_online'] == 0:
    opts['nNeg_init'] = 0
    opts['nPos_init'] = args.initial_samples

opts['finetune_iters'] = args.initial_iteration
opts['finetune_iters_online'] = args.online_iteration
opts['redet_samples'] = args.redetection_samples

if args.run_online == 'True':
    local_result = '/cache/result'
    args.save_result_npy = os.path.join(local_result, args.save_result_npy,
                                        os.path.basename(args.weight_file)[:-4] + '-' +
                                        str(args.pos_samples_ratio))
    import moxing
    local_data_url = "/cache/data"
    args.dataset_path = local_data_url
    local_weight_url = "/cache/weight/" + args.weight_file.split('/')[-1]
    # moving dataset from obs to container
    moxing.file.copy_parallel(args.data_url, local_data_url)
    # moving weight_file from obs to container
    moxing.file.copy_parallel(args.weight_file, local_weight_url)
    args.weight_file = local_weight_url + '/' + args.weight_file.split('/')[-1][:-4]
else:
    local_result = ''
    args.save_result_npy = os.path.join(args.save_result_npy, os.path.basename(args.weight_file)[:-4] + '-' +
                                        str(args.pos_samples_ratio))
if args.save_result_images is not None:
    args.save_result_images = os.path.join(local_result, args.save_result_images,
                                           os.path.basename(args.weight_file)[:-4] + '-' + str(args.pos_samples_ratio))
    if not os.path.exists(args.save_result_images):
        os.makedirs(args.save_result_images)

if not os.path.exists(args.save_result_npy):
    os.makedirs(args.save_result_npy)

if args.run_online == 'True':
    dataset_root = '/cache/data'
else:
    dataset_root = os.path.join(args.dataset_path)
vid_folders = []

for filename in os.listdir(dataset_root):
    if os.path.isdir(os.path.join(dataset_root, filename)):
        vid_folders.append(filename)
vid_folders.sort(key=str.lower)

save_root = args.save_result_images
save_root_npy = args.save_result_npy

for vid_folder in vid_folders:
    print('Loading {}...'.format(args.weight_file))
    opts['num_videos'] = 1
    net, domain_nets = adnet(opts,
                             trained_file=args.weight_file,
                             random_initialize_domain_specific=True,
                             multidomain=False, distributed=args.distributed)
    net.set_train()

    if args.save_result_images is not None:
        args.save_result_images = os.path.join(save_root, vid_folder)
        if not os.path.exists(args.save_result_images):
            os.makedirs(args.save_result_images)

    args.save_result_npy = os.path.join(save_root_npy, vid_folder)

    vid_path = os.path.join(dataset_root, vid_folder)

    # load ADNetDomainSpecific
    net.load_domain_specific(domain_nets[0])
    bboxes, t_sum = adnet_test(net, vid_path, opts, args)
