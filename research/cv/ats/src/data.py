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
"""load cifar data."""

import os

import mindspore.dataset.vision as mdv
from mindspore.dataset import GeneratorDataset
from mindvision.dataset.meta import Dataset
from mindvision.classification.dataset.cifar10 import ParseCifar10


def load_data(fdir, dset, download, batch_size):
    if dset == "cifar100":
        train_set = Cifar100(
            path=fdir, split="train", shuffle=True, download=download,
            batch_size=batch_size
        )
        test_set = Cifar100(
            path=fdir, split="test", shuffle=False, download=download,
            batch_size=batch_size
        )

        train_set = train_set.run()
        test_set = test_set.run()

        train_set = GeneratorDataset(
            train_set, column_names=["xs", "ys"], python_multiprocessing=False,
            shuffle=True
        )
        test_set = GeneratorDataset(
            test_set, column_names=["xs", "ys"], python_multiprocessing=False,
            shuffle=False
        )
    else:
        raise ValueError("No such dset: {}".format(dset))

    return train_set, test_set


class Cifar100(Dataset):
    def __init__(self, path, split, batch_size, shuffle, download):
        self.parse_cifar100 = ParseCifar10(path=os.path.join(path, split))
        load_c100_data = self.parse_cifar100.parse_dataset

        super(Cifar100, self).__init__(
            path=path,
            split=split,
            load_data=load_c100_data,
            transform=None,
            target_transform=None,
            batch_size=batch_size,
            repeat_num=1,
            resize=32,
            shuffle=shuffle,
            num_parallel_workers=1,
            num_shards=None,
            shard_id=None,
            download=download
        )

    @property
    def index2label(self):
        return self.parse_cifar100.index2label

    def download_dataset(self):
        self.parse_cifar100.download_and_extract_archive()

    def default_transform(self):
        trans = []
        if self.split == "train":
            trans += [
                mdv.RandomCrop((32, 32), (4, 4, 4, 4)),
                mdv.RandomHorizontalFlip(prob=0.5)
            ]

        trans += [
            mdv.Resize(self.resize),
            mdv.Rescale(1.0 / 255.0, 0.0),
            mdv.Normalize(
                [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
            ),
            mdv.HWC2CHW()
        ]

        return trans


if __name__ == "__main__":
    tr_set, te_set = load_data(
        fdir="./data", dset="cifar100", batch_size=512, download=False
    )
    for batch in tr_set.create_dict_iterator():
        xs, ys = batch["xs"], batch["ys"]
        print(xs.shape, ys.shape, xs.min(), xs.max(), ys.min(), ys.max())
