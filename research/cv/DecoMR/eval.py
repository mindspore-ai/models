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
import numpy as np
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore import context, ops, load_checkpoint
from datasets.base_dataset import BaseDataset
from datasets.surreal_dataset import SurrealDataset
from datasets.base_dataset import create_dataset
from models import SMPL
from models import dense_cnn, DMR
from models.uv_generator import Index_UV_Generator
from utils import config as cfg
from utils import TrainOptions


context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU', device_id=0)

# Define command-line arguments
options = TrainOptions().parse_args()
options.num_workers = 4

def run_evaluation():
    # Create SMPL model
    dataset_name = options.dataset
    log_freq = options.log_freq
    smpl = SMPL()
    if dataset_name == 'surreal':
        smpl_male = SMPL(cfg.MALE_SMPL_FILE)
        smpl_female = SMPL(cfg.FEMALE_SMPL_FILE)

    if dataset_name in ['up-3d', 'surreal']:
        eval_shape = True

    CNet = dense_cnn.DPNet(warp_lv=options.warp_level, norm_type=options.norm_type)
    LNet = dense_cnn.get_LNet(options)
    DMR_model = DMR.DMR(CNet, LNet)
    DMR_model.set_train(False)
    options.group_size = options.group_size * 3
    batch_size = options.batch_size

    # Create dataloader for the dataset
    if dataset_name == 'surreal':
        all_dataset = create_dataset('surreal', options, is_train=False, use_IUV=False)
        data = SurrealDataset(options, is_train=False, use_IUV=False)
    else:
        all_dataset = create_dataset('up-3d', options, is_train=False, use_IUV=False)
        data = BaseDataset(options, 'up-3d', is_train=False, use_IUV=False)

    shape_err = np.zeros(len(data))
    dataset = all_dataset.batch(options.batch_size)

    print('data loader finish')

    iter1 = dataset.create_dict_iterator()
    start_time = time.time()
    shape_err_list = msnp.zeros((30))
    for i in range(30):
        ckpt = os.path.join(options.eval_dir, 'dmr_{}.ckpt'.format(i+1))
        load_checkpoint(ckpt, net=DMR_model)
        for step, batch in enumerate(iter1):
            # Get ground truth annotations from the batch
            gt_pose = batch['pose']
            gt_betas = batch['betas']
            gt_vertices = smpl(gt_pose, gt_betas)
            images = batch['img']
            curr_batch_size = images.shape[0]
            uv_res = options.uv_res
            uv_type = options.uv_type
            sampler = Index_UV_Generator(UV_height=uv_res, UV_width=-1, uv_type=uv_type)
            _, pred_uv_map, _ = DMR_model(images)
            pred_vertices = sampler.resample(pred_uv_map.astype("float32")).astype("float32")

            # Shape evaluation (Mean per-vertex error)
            if eval_shape:
                if dataset_name == 'surreal':
                    gender = batch['gender']
                    gt_vertices = smpl_male(gt_pose, gt_betas)
                    gt_vertices_female = smpl_female(gt_pose, gt_betas)
                    temp = gt_vertices.asnumpy()
                    temp[gender == 1, :, :] = gt_vertices_female.asnumpy()[gender == 1, :, :]
                    gt_vertices = Tensor.from_numpy(temp)

                gt_pelvis_mesh = smpl.get_eval_joints(gt_vertices)
                pred_pelvis_mesh = smpl.get_eval_joints(pred_vertices)
                gt_pelvis_mesh = (gt_pelvis_mesh[:, [2]] + gt_pelvis_mesh[:, [3]]) / 2
                pred_pelvis_mesh = (pred_pelvis_mesh[:, [2]] + pred_pelvis_mesh[:, [3]]) / 2

                opsum = ops.ReduceSum(keep_dims=True)
                opmean = ops.ReduceMean(keep_dims=True)
                sqrt = ops.Sqrt()

                se = sqrt(((pred_vertices - pred_pelvis_mesh - gt_vertices + gt_pelvis_mesh) ** 2))
                se = opsum(se, -1).squeeze()
                se = opmean(se, -1).squeeze()

                shape_err[step * batch_size:step * batch_size + curr_batch_size] = se

            # Print intermediate results during evaluation
            if step % log_freq == log_freq - 1:
                if eval_shape:
                    print('Shape Error: ' + str(1000 * shape_err[:step * batch_size].mean()))
                    print()

        shape_err_list[i] = shape_err.mean()

    shape_err = min(shape_err_list)
    # Print final results during evaluation
    print('*** Final Results ***')
    print()
    if eval_shape:
        print('Shape Error: ' + str(1000 * shape_err))
        print('Total_time: ', time.time() - start_time, 's')
        print('Performance: ', (time.time() - start_time) * 1000 / (30 * dataset.get_dataset_size()), 'ms/step')
        print()

if __name__ == '__main__':
    run_evaluation()
