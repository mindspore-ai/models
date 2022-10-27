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

"""Load datasets"""
import os
import random
import mindspore.dataset as ds
from mindspore.communication.management import get_rank, get_group_size
from src.transforms import create_operators, Compose

class MattingDataset:
    """
    Load datasets
    """

    def __init__(self, dataset_root, transformers,
                 fg_names='test_fg_names.txt', bg_names='test_bg_names.txt',
                 img_subpath='image', alpha_subpath='mask',
                 name='P3M-10k', mode='train'):
        super().__init__()
        self.dataset_root = dataset_root
        self.mode = mode
        self.name = name

        if name == 'AMD':
            self.fg_file = os.path.join(self.dataset_root, fg_names)
            self.bg_file = os.path.join(self.dataset_root, bg_names)
            with open(self.fg_file) as f:
                self.fg_files = f.read().splitlines()
            with open(self.bg_file) as f:
                self.bg_files = f.read().splitlines()

        self.image_path = os.path.join(self.dataset_root, img_subpath)
        self.alpha_path = os.path.join(self.dataset_root, alpha_subpath)
        self.image_files = os.listdir(self.image_path)
        if mode == 'train':
            random.shuffle(self.image_files)

        self.transformers = transformers

    def __getitem__(self, index):
        # get files path
        image_file = self.image_files[index]
        if self.name == 'AMD':
            if self.mode == 'train':
                fcount = int(image_file.split('.')[0].split('_')[0])
            else:
                fcount = int(image_file.split('.')[0].split('!')[2])

            alpha_file = self.fg_files[fcount]
        elif self.name == 'P3M-10k':
            alpha_file = image_file.replace('.jpg', '.png')
        elif self.name == 'PPM':
            alpha_file = image_file
        else:
            raise NotImplementedError

        img_file_path = os.path.join(self.image_path, image_file)
        alpha_file_path = os.path.join(self.alpha_path, alpha_file)
        data = dict(image=img_file_path, alpha=alpha_file_path, name=image_file)

        data = self.transformers(data)

        if self.mode == 'train':
            image, trimap, alpha = data['image'], data['trimap'], data['alpha']
            return image, trimap, alpha
        image, alpha = data['image'], data['alpha']
        return image, alpha, image_file

    def __len__(self):
        return len(self.image_files)


def create_dataset(cfg, usage='train', repeat_num=1):

    rank_size, rank_id = _get_rank_info()

    data_transformer_list = create_operators(cfg['transforms'])
    data_transformers = Compose(data_transformer_list)
    dataset_generator = MattingDataset(dataset_root=cfg['dataset_root'], transformers=data_transformers,
                                       img_subpath=cfg['img_subpath'], alpha_subpath=cfg['alpha_subpath'],
                                       fg_names=cfg['fg_names'], bg_names=cfg['bg_names'],
                                       name=cfg['name'], mode=usage)

    dataset = ds.GeneratorDataset(dataset_generator,
                                  column_names=["image", "trimap", "alpha"],
                                  shuffle=True,
                                  num_parallel_workers=cfg['workers'],
                                  num_shards=rank_size,
                                  shard_id=rank_id
                                  )

    dataset = dataset.batch(cfg['batch_size'], drop_remainder=True)
    dataset = dataset.repeat(repeat_num)

    return dataset


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id
