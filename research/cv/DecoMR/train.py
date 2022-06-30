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

import os
import time
from mindspore import context, nn, Tensor, save_checkpoint
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.common import set_seed
from datasets.base_dataset import create_dataset
from models import dense_cnn
from models.WithLossCellDP import WithLossCellDP
from models.WithLossCellEnd import WithLossCellEnd
from models.TrainOneStepDP import TrainOneStepDP
from models.TrainOneStepEnd import TrainOneStepEnd
from models.uv_generator import Index_UV_Generator
from models.DMR import DMR
from models import SMPL
from utils import TrainOptions
import numpy as np

set_seed(1)

options = TrainOptions().parse_args()

context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')

def train():
    if options.run_distribute:

        init()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, device_num=get_group_size(),
                                          gradients_mean=True)
        options.rank = get_rank()
        options.group_size = get_group_size()
        options.ckpt_dir = os.path.join(options.ckpt_dir, 'rank{}'.format(options.rank))
        all_dataset = create_dataset(options.dataset, options, is_train=True, use_IUV=True)
        dataset = all_dataset.batch(options.batch_size, drop_remainder=True)
    else:
        options.ckpt_dir = os.path.join(options.ckpt_dir, 'rank{}'.format(options.rank))
        context.set_context(device_id=options.device_id)
        all_dataset = create_dataset(options.dataset, options, is_train=True, use_IUV=True)
        dataset = all_dataset.batch(options.batch_size)

    smpl = SMPL()

    sampler = Index_UV_Generator(UV_height=options.uv_res, UV_width=-1, uv_type=options.uv_type)

    weight_file = 'data/weight_p24_h{:04d}_w{:04d}_{}.npy'.format(options.uv_res, options.uv_res, options.uv_type)
    uv_weight = Tensor.from_numpy(np.load(weight_file))
    uv_weight = uv_weight * sampler.mask
    uv_weight = uv_weight / uv_weight.mean()
    uv_weight = uv_weight[None, :, :, None]
    tv_factor = (options.uv_res - 1) * (options.uv_res - 1)


    CNet = dense_cnn.DPNet(warp_lv=options.warp_level, norm_type=options.norm_type)
    optimizer_CNet = nn.Adam(CNet.trainable_params(), learning_rate=options.lr, beta1=options.adam_beta1)
    CNet_with_criterion = WithLossCellDP(CNet, options)
    TrainOneStepCellDP = TrainOneStepDP(CNet_with_criterion, optimizer_CNet)

    TrainOneStepCellDP.set_train()

    if not os.path.exists(options.ckpt_dir):
        os.makedirs(options.ckpt_dir)

    iter1 = dataset.create_dict_iterator(num_epochs=options.num_epochs_dp)
    for epoch in range(options.num_epochs_dp):
        start_epoch_time = time.time()
        for i, data in enumerate(iter1):
            input_data = data
            has_dp = input_data['has_dp']
            images = input_data['img']
            gt_dp_iuv = input_data['gt_iuv']
            fit_joint_error = input_data['fit_joint_error']

            gt_dp_iuv[:, 1:] = gt_dp_iuv[:, 1:] / 255.0

            Cout = TrainOneStepCellDP(has_dp, images, gt_dp_iuv, fit_joint_error)

            print('stage_dp:', 'epoch', epoch, 'step', i, 'CLoss', Cout[0])

        print('stage_dp:', 'epoch', epoch, 'use time:', time.time() - start_epoch_time, 's')
        print('stage_dp:', 'epoch', epoch, 'performance', (time.time() - start_epoch_time)
              * 1000 / dataset.get_dataset_size(), 'ms/step')
        if (epoch + 1) % 1 == 0 and options.rank == 0:
            save_checkpoint(CNet, os.path.join(options.ckpt_dir, f"CNet_{epoch + 1}.ckpt"))

    Pretrained_CNet = dense_cnn.Pretrained_DPNet(options.warp_level, options.norm_type, pretrained=True)
    LNet = dense_cnn.get_LNet(options)

    DMR_model = DMR(Pretrained_CNet, LNet)
    optimizer_DMR_model = nn.Adam(DMR_model.trainable_params(), learning_rate=options.lr, beta1=options.adam_beta1)
    DMR_model_with_criterion = WithLossCellEnd(DMR_model, options, uv_weight, tv_factor)
    TrainOneStepCellEnd = TrainOneStepEnd(DMR_model_with_criterion, optimizer_DMR_model)
    TrainOneStepCellEnd.set_train(True)

    iter2 = dataset.create_dict_iterator(num_epochs=options.num_epochs_end)
    for epoch in range(options.num_epochs_end):
        start_epoch_time = time.time()
        for i, data in enumerate(iter2):

            input_data = data
            gt_keypoints_2d = input_data['keypoints']
            gt_keypoints_3d = input_data['pose_3d']
            has_pose_3d = input_data['has_pose_3d']

            gt_keypoints_2d_smpl = input_data['keypoints_smpl']
            gt_keypoints_3d_smpl = input_data['pose_3d_smpl']
            has_pose_3d_smpl = input_data['has_pose_3d_smpl']

            gt_pose = input_data['pose']
            gt_betas = input_data['betas']
            has_smpl = input_data['has_smpl']
            has_dp = input_data['has_dp']
            images = input_data['img']

            gt_dp_iuv = input_data['gt_iuv']
            fit_joint_error = input_data['fit_joint_error']

            gt_dp_iuv[:, 1:] = gt_dp_iuv[:, 1:] / 255.0

            gt_vertices = smpl(gt_pose, gt_betas)

            gt_uv_map = sampler.get_UV_map(gt_vertices)

            Lout = TrainOneStepCellEnd(images, has_dp, has_smpl, has_pose_3d, has_pose_3d_smpl, gt_dp_iuv, gt_uv_map,
                                       gt_vertices, fit_joint_error, gt_keypoints_2d, gt_keypoints_3d,
                                       gt_keypoints_2d_smpl, gt_keypoints_3d_smpl)

            print('stage_end:', 'epoch', epoch, 'step', i, 'CLoss', Lout[1], 'LLoss', Lout[2], 'total', Lout[0])
        print('stage_end:', 'epoch', epoch, 'use time:', time.time() - start_epoch_time, 's')
        print('stage_end:', 'epoch', epoch, 'performance', (time.time() - start_epoch_time)
              * 1000 / dataset.get_dataset_size(), 'ms/step')
        if (epoch + 1) % 1 == 0 and options.rank == 0:
            save_checkpoint(DMR_model, os.path.join(options.ckpt_dir, f"dmr_{epoch + 1}.ckpt"))

if __name__ == '__main__':
    train()
