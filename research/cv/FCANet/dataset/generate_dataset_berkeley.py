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
""" generate the interactive segmentation format of berkeley dataset """
import os
import shutil
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
    global berkeley_path, dataset_path
    f_output = open(dataset_path / "list" / (split + ".txt"), "w")
    for img_id in tqdm(img_ids):
        shutil.copyfile(
            berkeley_path / "images" / (img_id + ".jpg"),
            dataset_path / "img" / (img_id + ".jpg"),
        )
        gt = np.array(Image.open(berkeley_path / "masks" / (img_id + ".png")))
        gt_ins = (gt[:, :, 0] > 127).astype(np.uint8)
        id_ins = img_id + "#001"
        save_image(dataset_path / "gt" / (id_ins + ".png"), gt_ins, if_pal=True)
        f_output.write(id_ins + "\n")
    f_output.close()


if __name__ == "__main__":
    # parameters of datasets path
    parser = argparse.ArgumentParser(description="Generate ISF Dataset Berkeley")
    parser.add_argument(
        "--dst_path",
        type=str,
        default="./",
        help="destination path of generated dataset",
    )
    parser.add_argument(
        "--src_berkeley_path",
        type=str,
        default="./path/to/source/berkeley/Berkeley",
        help="source path of berkeley dataset",
    )
    args = parser.parse_args()

    # create folder
    dataset_path = Path(args.dst_path) / "Berkeley"
    os.makedirs(dataset_path, exist_ok=True)
    for folder in ["img", "gt", "list"]:
        os.makedirs(dataset_path / folder, exist_ok=True)

    # set original dataset path
    berkeley_path = Path(args.src_berkeley_path)

    # get ids list
    berkeley_val_ids = [t.stem for t in (berkeley_path / "masks").glob("*.png")]

    # process val split
    process(berkeley_val_ids, "val")
