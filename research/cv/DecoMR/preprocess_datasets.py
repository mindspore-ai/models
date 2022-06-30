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

import argparse
from utils import config as cfg
from utils import objfile
from utils.renderer import UVRenderer
from models import SMPL
import numpy as np


from datasets.preprocess import \
    up_3d_extract, \
    process_dataset

import mindspore.context as context

context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU', device_id=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_files', default=False, action='store_true', help='Extract files needed for training')
    parser.add_argument('--eval_files', default=False, action='store_true', help='Extract files needed for evaluation')
    parser.add_argument('--gt_iuv', default=True, action='store_true', help='Extract files needed for evaluation')
    parser.add_argument('--uv_type', type=str, default='BF', choices=['BF', 'SMPL'])

    args = parser.parse_args()

    # define path to store extra files
    out_path = cfg.DATASET_NPZ_PATH

    print("cfg.UP_3D_ROOT= ", cfg.UP_3D_ROOT)
    print("out_path = ", out_path)
    if args.train_files:

        # UP-3D dataset preprocessing (trainval set)
        up_3d_extract(cfg.UP_3D_ROOT, out_path, 'trainval')


    if args.eval_files:

        # UP-3D dataset preprocessing (lsp_test set)
        up_3d_extract(cfg.UP_3D_ROOT, out_path, 'lsp_test')

    if args.gt_iuv:
        smpl = SMPL(model_file=cfg.SMPL_FILE)
        uv_type = args.uv_type
        print("uv_type = ", uv_type)
        uv_type = args.uv_type

        if uv_type == 'SMPL':
            data = objfile.read_obj_full('data/uv_sampler/smpl_fbx_template.obj')
        elif uv_type == 'BF':
            data = objfile.read_obj_full('data/uv_sampler/smpl_boundry_free_template.obj')

        vt = np.array(data['texcoords'])
        face = [f[0] for f in data['faces']]
        face = np.array(face) - 1
        vt_face = [f[2] for f in data['faces']]
        vt_face = np.array(vt_face) - 1
        renderer = UVRenderer(faces=face, tex=np.zeros([256, 256, 3]), vt=1 - vt, ft=vt_face)
        for dataset_name in ['up-3d']:
            process_dataset(dataset_name, is_train=True, uv_type=uv_type, smpl=smpl, renderer=renderer)
