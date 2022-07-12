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

"""generate test set for validate model"""

import os
import argparse
import shutil

import numpy as np



def do_generate_testset(par):
    """generate testset"""
    args = par.parse_args()
    split_file = np.load(args.splits_final, allow_pickle=True)
    val_list = split_file[args.fold]["val"]

    raw_data_path = os.path.join("src/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data",
                                 (os.path.basename(os.path.dirname(args.splits_final))), "imagesTr")
    labelsTr_path = os.path.join("src/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data",
                                 (os.path.basename(os.path.dirname(args.splits_final))), "labelsTr")
    imagesVal_path = os.path.join("src/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data",
                                  (os.path.basename(os.path.dirname(args.splits_final))), "imagesVal")
    labelsTs_path = os.path.join("src/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data",
                                 (os.path.basename(os.path.dirname(args.splits_final))), "labelsVal")

    if not os.path.exists(imagesVal_path):
        os.makedirs(imagesVal_path)
    if not os.path.exists(labelsTs_path):
        os.makedirs(labelsTs_path)

    for root, _, files in os.walk(raw_data_path):
        for file in files:
            if file.replace("_0000.nii.gz", "") in val_list:
                shutil.copy(os.path.join(root, file), imagesVal_path)
                shutil.copy(os.path.join(labelsTr_path, file.replace("_0000", "")), labelsTs_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_final", type=str,
                        default="src/nnUNetFrame/DATASET/nnUNet_preprocessed/Task004_Hippocampus/splits_final.pkl",
                        required=False, help="split file path")
    parser.add_argument("--fold", type=int, default=0, required=False, help="which fold for validate")
    do_generate_testset(par=parser)
