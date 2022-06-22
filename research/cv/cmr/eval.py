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

"""
This script can be used to evaluate a trained model on 3D pose/shape and masks/part segmentation.
"""

from __future__ import print_function
from __future__ import division

import os

from mindspore import load_checkpoint, load_param_into_net
from mindspore import context
from mindspore import numpy as mnp

from src.utils.options import EvalOptions
from src.netCell.evalNet import CMREvalNet
from src.models.cmr import CMR
from src.models.smpl import SMPL
from src.utils.mesh import MeshCell
from src.dataset.datasets import create_eval_dataset
from src import config as cfg

import numpy as np


def run_evaluation(options, model, smpl, dataset_name, batch_size=32, num_workers=2, shuffle=False, log_freq=50):
    """
    Run evaluation on the dataset and metrics.
    """
    # Create dataset
    ds_eval = create_eval_dataset(options, dataset_name, batch_size, shuffle=shuffle, num_workers=num_workers)

    # Shape metrics
    # Mean per-vertex error
    num_samples = ds_eval.get_batch_size() * ds_eval.get_dataset_size()
    shape_err = np.zeros(num_samples)
    shape_err_smpl = np.zeros(num_samples)

    eval_shape = False

    # Choose appropriate evaluation for each dataset
    if dataset_name == 'up-3d':
        eval_shape = True

    for step, data in enumerate(ds_eval.create_dict_iterator()):
        # Get ground truth annotations from the batch
        gt_pose = data['pose']
        gt_betas = data['betas']
        gt_vertices = smpl(gt_pose, gt_betas)
        images = data['img']
        curr_batch_size = images.shape[0]

        # Run inference
        pred_vertices, pred_vertices_smpl, _, _, _ = model(images)

        # Shape evaluation (Mean per-vertex error)
        if eval_shape:
            se = mnp.sqrt(((pred_vertices - gt_vertices) ** 2).sum(axis=-1)).mean(axis=-1)
            se_smpl = mnp.sqrt(((pred_vertices_smpl - gt_vertices) ** 2).sum(axis=-1)).mean(axis=-1)
            shape_err[step * batch_size: step*batch_size+curr_batch_size] = se
            shape_err_smpl[step * batch_size: step*batch_size+curr_batch_size] = se_smpl

        if step % log_freq == log_freq - 1:
            if eval_shape:
                print('Shape Error (NonParam): ' + str(1000 * shape_err[:step * batch_size].mean()))
                print('Shape Error (Param): ' + str(1000 * shape_err_smpl[:step * batch_size].mean()))
                print()

    # Print final results during evaluation
    print('*** Final Results ***')
    print()
    if eval_shape:
        print('Shape Error (NonParam): ' + str(1000 * shape_err.mean()))
        print('Shape Error (Param): ' + str(1000 * shape_err_smpl.mean()))
        print()

    return 1000 * shape_err.mean(), 1000 * shape_err_smpl.mean()

if __name__ == '__main__':

    eval_options = EvalOptions().parse_args()
    device_id = int(os.getenv("DEVICE_ID", str(0)))
    context.set_context(mode=context.PYNATIVE_MODE, device_target=eval_options.device_target)
    context.set_context(device_id=device_id)

    # Loading model
    smpl_model = SMPL()
    smpl_params = load_checkpoint(cfg.SMPL_CKPT_FILE)
    load_param_into_net(smpl_model, smpl_params)

    mesh = MeshCell(smpl_model)
    mesh_params = load_checkpoint(cfg.MESH_CKPT_FILE)
    load_param_into_net(mesh, mesh_params)
    mesh.update_paramter()

    cmr = CMR(mesh, smpl_model, eval_options.num_layers, eval_options.num_channels)
    cmr_params = load_checkpoint(eval_options.checkpoint)
    load_param_into_net(cmr, cmr_params)
    cmr_eval_net = CMREvalNet(cmr)
    cmr_eval_net.set_train(False)

    print("Start eval")
    run_evaluation(eval_options, cmr_eval_net, smpl_model, eval_options.dataset, eval_options.batch_size,
                   eval_options.num_workers, eval_options.shuffle, eval_options.log_freq)
