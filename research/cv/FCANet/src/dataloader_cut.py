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
"""
Dataset loader, will be used in trainer.py
"""
import random
from pathlib import Path
import numpy as np
from PIL import Image

class GeneralCutDataset():
    """
    Dataset loader
    """
    def __init__(
            self,
            dataset_path,
            datasets,
            list_file="train.txt",
            transform=None,
            max_num=0,
            batch_size=0,
            suffix=None,
    ):
        super().__init__()
        if not isinstance(datasets, list):
            datasets = [datasets]
        data_lists = [
            str(Path(dataset_path) / dataset / "list" / list_file)
            for dataset in datasets
        ]
        # load ids
        self.imgs_list, self.gts_list = [], []
        for data_list in data_lists:
            with open(data_list) as f:
                lines = f.read().splitlines()
                if data_list.split("/")[-3] == "PASCAL_SBD":
                    lines = lines[:]

                if suffix is None:
                    img_suffix = (
                        (Path(data_list.split("list")[0]) / "img")
                        .glob("{}.*".format(lines[0].split("#")[0]))
                        .__next__()
                        .suffix
                    )
                    gt_suffix = (
                        (Path(data_list.split("list")[0]) / "gt")
                        .glob("{}.*".format(lines[0]))
                        .__next__()
                        .suffix
                    )
                    suffix_tmp = [img_suffix, gt_suffix]
                else:
                    suffix_tmp = suffix

                for line in lines:
                    self.imgs_list.append(
                        data_list.split("list")[0]
                        + "img/"
                        + line.split("#")[0]
                        + suffix_tmp[0]
                    )
                    self.gts_list.append(
                        data_list.split("list")[0] + "gt/" + line + suffix_tmp[1]
                    )

        # set actual sample number, 0 means all
        if max_num != 0 and len(self.imgs_list) > abs(max_num):
            indices = (
                random.sample(range(len(self.imgs_list)), max_num)
                if max_num > 0
                else range(abs(max_num))
            )
            self.imgs_list = [self.imgs_list[i] for i in indices]
            self.gts_list = [self.gts_list[i] for i in indices]

        # set actual sample number according to batch size, 0 means no change, positive number means cutoff, positive number means completion
        if batch_size > 0:
            actual_num = (len(self.imgs_list) // batch_size) * batch_size
            self.imgs_list = self.imgs_list[:actual_num]
            self.gts_list = self.gts_list[:actual_num]
        elif batch_size < 0:
            if len(self.imgs_list) % abs(batch_size) != 0:
                actual_num = ((len(self.imgs_list) // abs(batch_size)) + 1) * abs(
                    batch_size
                )
                add_num = actual_num - len(self.imgs_list)
                for add_i in range(add_num):
                    self.imgs_list.append(self.imgs_list[add_i])
                    self.gts_list.append(self.gts_list[add_i])

        self.ids_list = [t.split("/")[-1].split(".")[0] for t in self.gts_list]

        self.transform = transform

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        return (
            self.transform(self.get_sample(index))
            if self.transform is not None
            else self.get_sample(index)
        )

    def get_sample(self, index):
        """generate samples"""
        img, gt = (
            np.array(Image.open(self.imgs_list[index])),
            np.array(Image.open(self.gts_list[index]))
        )
        gt = (gt == 1).astype(np.uint8) * 255
        sample = {"img": img, "gt": gt}
        sample["meta"] = {"id": str(Path(self.gts_list[index]).stem), "id_num": index}
        sample["meta"]["source_size"] = np.array(gt.shape[::-1])
        sample["meta"]["img_path"] = self.imgs_list[index]
        sample["meta"]["gt_path"] = self.gts_list[index]

        return sample
