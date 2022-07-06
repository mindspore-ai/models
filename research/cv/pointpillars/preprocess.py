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
"""Evaluation script"""

import argparse
import os
import warnings
import shutil

from mindspore import context
from mindspore import dataset as de

from src.utils import get_config
from src.utils import get_model_dataset
import numpy as np
warnings.filterwarnings('ignore')


def run_evaluate(args):
    """run evaluate"""
    cfg_path = args.cfg_path

    cfg = get_config(cfg_path)

    device_id = int(args.device_id)
    device_target = args.device_target

    context.set_context(mode=context.PYNATIVE_MODE, device_target=device_target, device_id=device_id)

    _, eval_dataset, _ = get_model_dataset(cfg, False)

    eval_input_cfg = cfg['eval_input_reader']
    eval_column_names = eval_dataset.data_keys

    ds = de.GeneratorDataset(
        eval_dataset,
        column_names=eval_column_names,
        python_multiprocessing=True,
        num_parallel_workers=6,
        max_rowsize=100,
        shuffle=False
    )
    batch_size = eval_input_cfg['batch_size']
    print("eval input path:", eval_input_cfg['kitti_info_path'])
    print("batch size:", batch_size)
    ds = ds.batch(batch_size, drop_remainder=False)
    data_loader = ds.create_dict_iterator(num_epochs=1)

    # data for model forward
    save_dir_np = "process_Result/num_points_data"
    save_dir_coor = "process_Result/coors_data"
    save_dir_voxel = "process_Result/voxels_data"
    save_dir_bevm = "process_Result/bev_map_data"
    if not os.path.exists(save_dir_np):
        os.mkdir(save_dir_np)
    if not os.path.exists(save_dir_coor):
        os.mkdir(save_dir_coor)
    if not os.path.exists(save_dir_voxel):
        os.mkdir(save_dir_voxel)
    if not os.path.exists(save_dir_bevm):
        os.mkdir(save_dir_bevm)


    # prepare label file
    shutil.copyfile(eval_input_cfg['kitti_info_path'], 'process_Result/kitti_infos_val.pkl')

    # data for cal acc
    save_dir_anchors = "process_Result/anchors_data"
    save_dir_rect = "process_Result/rect_data"
    save_dir_Trv2c = "process_Result/Trv2c_data"
    save_dir_P2 = "process_Result/P2_data"
    save_dir_anchors_mask = "process_Result/anchors_mask_data"
    save_dir_image_idx = "process_Result/image_idx_data"
    save_dir_imgshape = "process_Result/imgshape_data"
    if not os.path.exists(save_dir_anchors):
        os.mkdir(save_dir_anchors)
    if not os.path.exists(save_dir_rect):
        os.mkdir(save_dir_rect)
    if not os.path.exists(save_dir_Trv2c):
        os.mkdir(save_dir_Trv2c)
    if not os.path.exists(save_dir_P2):
        os.mkdir(save_dir_P2)
    if not os.path.exists(save_dir_anchors_mask):
        os.mkdir(save_dir_anchors_mask)
    if not os.path.exists(save_dir_image_idx):
        os.mkdir(save_dir_image_idx)
    if not os.path.exists(save_dir_imgshape):
        os.mkdir(save_dir_imgshape)

    for i, data in enumerate(data_loader):
        bev_map = data.get('bev_map', False)

        num_points_kitti_file_path = os.path.join(save_dir_np, "kittiVal" + str(1) + "_" + str(i) + "_num_points.bin")
        coors_kitti_file_path = os.path.join(save_dir_coor, "kittiVal" + str(1) + "_" + str(i) + "_coors.bin")
        voxels_kitti_file_path = os.path.join(save_dir_voxel, "kittiVal" + str(1) + "_" + str(i) + "_voxels.bin")
        bev_map_kitti_file_path = os.path.join(save_dir_bevm, "kittiVal" + str(1) + "_" + str(i) + "_bev_map.bin")

        data["voxels"].asnumpy().tofile(voxels_kitti_file_path)
        np.asarray(bev_map).tofile(bev_map_kitti_file_path)
        data["num_points"].asnumpy().tofile(num_points_kitti_file_path)
        data["coordinates"].asnumpy().tofile(coors_kitti_file_path)

        anchors_kitti_file_path = os.path.join(save_dir_anchors, "kittiVal" + str(1) + "_" + str(i) + "_anchors.bin")
        rect_kitti_file_path = os.path.join(save_dir_rect, "kittiVal" + str(1) + "_" + str(i) + "_rect.bin")
        Trv2c_kitti_file_path = os.path.join(save_dir_Trv2c, "kittiVal" + str(1) + "_" + str(i) + "_Trv2c.bin")
        P2_kitti_file_path = os.path.join(save_dir_P2, "kittiVal" + str(1) + "_" + str(i) + "_P2.bin")
        anchors_mask_kitti_file_path = os.path.join(save_dir_anchors_mask,
                                                    "kittiVal" + str(1) + "_" + str(i) + "_anchors_mask.bin")
        image_idx_kitti_file_path = os.path.join(save_dir_image_idx,
                                                 "kittiVal" + str(1) + "_" + str(i) + "_image_idx.bin")
        imgshape_kitti_file_path = os.path.join(save_dir_imgshape, "kittiVal" + str(1) + "_" + str(i) + "_imgshape.bin")
        data["anchors"].asnumpy().tofile(anchors_kitti_file_path)
        data["rect"].asnumpy().tofile(rect_kitti_file_path)
        data["Trv2c"].asnumpy().tofile(Trv2c_kitti_file_path)
        data["P2"].asnumpy().tofile(P2_kitti_file_path)
        data["anchors_mask"].asnumpy().tofile(anchors_mask_kitti_file_path)
        data["image_idx"].asnumpy().tofile(image_idx_kitti_file_path)
        data["image_shape"].asnumpy().tofile(imgshape_kitti_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', required=True, help='Path to config file.')
    parser.add_argument('--ckpt_path', required=True, help='Path to checkpoint.')
    parser.add_argument('--device_target', default='GPU', help='device target')
    parser.add_argument('--device_id', required=True, help='device id')

    parse_args = parser.parse_args()

    run_evaluate(parse_args)
