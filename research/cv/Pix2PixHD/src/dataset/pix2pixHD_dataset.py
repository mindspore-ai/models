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
# ===========================================================================

"""
    Preprocess pix2pixHD dataset.
"""

import os
from PIL import Image
import numpy as np

from mindspore import dataset as de

from src.utils.config import config
from .base_dataset import make_dataset, get_params, get_transform, normalize


class Pix2PixHDDataset:
    """
    Define train dataset.
    """

    def __init__(self, root_dir, is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train
        # input A (label maps)
        dir_A = "_A" if config.label_nc == 0 else "_label"
        self.dir_A = os.path.join(root_dir, config.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))
        # input B (real images)
        if is_train or config.use_encoded_image:
            dir_B = "_B" if config.label_nc == 0 else "_img"
            self.dir_B = os.path.join(root_dir, config.phase + dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B))

        # instance maps
        if not config.no_instance:
            self.dir_inst = os.path.join(root_dir, config.phase + "_inst")
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        # load precomputed instance-wise encoded features
        if config.load_features:
            self.dir_feat = os.path.join(root_dir, config.phase + "_feat")
            print("----------- loading features from %s ----------" % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        # input A (label maps)
        A_path = self.A_paths[index]
        A = Image.open(A_path)
        params = get_params(A.size)
        if config.label_nc == 0:
            transform_A = get_transform(params, is_train=self.is_train)
            A_tensor = transform_A(A.convert("RGB"))[0]
        else:
            transform_A = get_transform(params, method=Image.NEAREST, is_normalize=False, is_train=self.is_train)
            A_tensor = transform_A(A)[0] * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        # input B (real images)
        if self.is_train or config.use_encoded_image:
            B_path = self.B_paths[index]
            B = Image.open(B_path).convert("RGB")
            transform_B = get_transform(params)
            B_tensor = transform_B(B)[0]

        # if using instance maps
        if not config.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)[0] * 255.0
            inst_tensor = inst_tensor.astype(np.int32)

            if config.load_features:
                feat_path = self.feat_paths[index]
                feat = Image.open(feat_path).convert("RGB")
                norm = normalize()
                feat_tensor = norm(transform_A(feat)[0])

        return A_tensor, inst_tensor, B_tensor, feat_tensor, A_path


def create_train_dataset(dataset, batch_size, run_distribute=False, num_parallel_workers=8):
    """
    Create train dataset.
    """
    if run_distribute:
        from mindspore.communication.management import get_rank, get_group_size

        train_ds = de.GeneratorDataset(
            dataset,
            column_names=["label", "inst", "image", "feat", "path"],
            shuffle=not config.serial_batches,
            num_shards=get_group_size(),
            shard_id=get_rank(),
            num_parallel_workers=num_parallel_workers,
        )
    else:
        train_ds = de.GeneratorDataset(
            dataset,
            column_names=["label", "inst", "image", "feat", "path"],
            shuffle=not config.serial_batches,
            num_parallel_workers=num_parallel_workers,
        )

    train_ds = train_ds.batch(batch_size, drop_remainder=True)

    return train_ds


def create_eval_dataset(dataset, batch_size=1):
    """
    Create eval dataset.
    """

    eval_ds = de.GeneratorDataset(
        dataset, column_names=["label", "inst", "image", "feat", "path"], shuffle=not config.serial_batches
    )

    eval_ds = eval_ds.batch(batch_size, drop_remainder=True)

    return eval_ds
