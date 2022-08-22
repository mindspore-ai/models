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
from src import config as cfg
from src.dataset.preprocess.coco import coco_extract
from src.dataset.preprocess.lsp_dataset import lsp_dataset_extract
from src.dataset.preprocess.lsp_dataset_original import lsp_dataset_original_extract
from src.dataset.preprocess.mpii import mpii_extract
from src.dataset.preprocess.up_3d import up_3d_extract

parser = argparse.ArgumentParser()
parser.add_argument('--train_files', default=False, action='store_true', help='Extract files needed for training')
parser.add_argument('--eval_files', default=False, action='store_true', help='Extract files needed for evaluation')

if __name__ == '__main__':
    args = parser.parse_args()

    # define path to store extra files
    out_path = cfg.DATASET_NPZ_PATH

    if args.train_files:
        # UP-3D dataset preprocessing (trainval set)
        up_3d_extract(cfg.UP_3D_ROOT, out_path, 'trainval')
        print("UP-3D dataset preprocesses successfully")

        # LSP dataset original preprocessing (training set)
        lsp_dataset_original_extract(cfg.LSP_ORIGINAL_ROOT, out_path)
        print("LSP dataset preprocesses successfully")

        # MPII dataset preprocessing
        mpii_extract(cfg.MPII_ROOT, out_path)
        print("MPII dataset preprocesses successfully")

        # COCO dataset prepreocessing
        coco_extract(cfg.COCO_ROOT, out_path)
        print("COCO dataset preprocesses successfully")

    if args.eval_files:
        # LSP dataset preprocessing (test set)
        lsp_dataset_extract(cfg.LSP_ROOT, out_path)
        print("LSP dataset preprocesses successfully")

        # UP-3D dataset preprocessing (lsp_test set)
        up_3d_extract(cfg.UP_3D_ROOT, out_path, 'lsp_test')
        print("UP-3D dataset preprocesses successfully")
