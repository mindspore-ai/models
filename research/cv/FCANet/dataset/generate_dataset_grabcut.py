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
""" generate the interactive segmentation format of grabcut dataset """
import os
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image


def save_image(path, img, if_pal=False):
    """save image with palette"""
    img = Image.fromarray(img)
    if if_pal:
        img.putpalette([0, 0, 0, 128, 0, 0] + [0, 0, 0] * 253 + [224, 224, 192])
    img.save(path)


def process(img_ids, split):
    """ process dataset split"""
    global grabcut_path, dataset_path
    f_output = open(dataset_path / "list" / (split + ".txt"), "w")
    for img_id in tqdm(img_ids):
        img = np.array(Image.open(grabcut_path / "data_GT" / img_id.name))
        save_image(dataset_path / "img" / (img_id.stem + ".png"), img, if_pal=False)
        gt = np.array(Image.open(grabcut_path / "boundary_GT" / (img_id.stem + ".bmp")))
        gt_ins = np.zeros_like(gt)
        gt_ins[gt == 255] = 1
        gt_ins[gt == 128] = 255
        id_ins = img_id.stem + "#001"
        save_image(dataset_path / "gt" / (id_ins + ".png"), gt_ins, if_pal=True)
        f_output.write(id_ins + "\n")

    f_output.close()


if __name__ == "__main__":
    # parameters of datasets path
    parser = argparse.ArgumentParser(description="Generate ISF Dataset GrabCut")
    parser.add_argument(
        "--dst_path",
        type=str,
        default="./",
        help="destination path of generated dataset",
    )
    parser.add_argument(
        "--src_grabcut_path",
        type=str,
        default="./path/to/source/grabcut/GrabCut",
        help="source path of grabcut dataset",
    )
    args = parser.parse_args()

    # create folder
    dataset_path = Path(args.dst_path) / "GrabCut"
    os.makedirs(dataset_path, exist_ok=True)
    for folder in ["img", "gt", "list"]:
        os.makedirs(dataset_path / folder, exist_ok=True)

    # set original dataset path
    grabcut_path = Path(args.src_grabcut_path)

    # get ids list
    grabcut_val_ids = [t for t in (grabcut_path / "data_GT").glob("*.*")]

    # process val split
    process(grabcut_val_ids, "val")
