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
import random
import os
import pickle
import itertools
import numpy as np

import mindspore
import mindspore.ops as ops
from mindspore import Tensor

class Data_Utils:
    """docstring for Data_Utils:(参数解析器，配置数据)"""
    def __init__(self, train, seed, way, shot,
                 data_path, dataset_name, embedding_crop,
                 batchsize, val_batch_size, test_batch_size,
                 meta_val_steps, embedding_size, verbose):
        self.train = train
        self.seed = seed
        self.way = way
        self.shot = shot
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.embedding_crop = embedding_crop
        self.batch_size = batchsize
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.meta_val_steps = meta_val_steps
        self.embedding_size = embedding_size
        self.verbose = verbose

        if self.train:
            self.metasplit = ['train', 'val']
        else:
            self.metasplit = ['test']

        random.seed(self.seed)
        self.construct_data()

    def construct_data(self):
        # loading embeddings
        self.embedding_path = os.path.join(self.data_path, self.dataset_name, self.embedding_crop)

        self.embeddings = {}
        for d in self.metasplit:
            if self.verbose:
                print('Loading data from ' + os.path.join(self.embedding_path, d+'_embeddings.pkl') + '...')
            self.embeddings[d] = pickle.load(open(os.path.join(self.embedding_path, d+'_embeddings.pkl'), 'rb'),
                                             encoding='iso-8859-1')

        # sort images by class
        self.image_by_class = {}
        self.embed_by_name = {}
        self.class_list = {}
        for d in self.metasplit:
            self.image_by_class[d] = {}
            self.embed_by_name[d] = {}
            self.class_list[d] = set()
            keys = self.embeddings[d]["keys"]
            for i, k in enumerate(keys):
                _, class_name, img_name = k.split('-')
                if class_name not in self.image_by_class[d]:
                    self.image_by_class[d][class_name] = []
                self.image_by_class[d][class_name].append(img_name)
                self.embed_by_name[d][img_name] = self.embeddings[d]["embeddings"][i]
                self.class_list[d].add(class_name)

            self.class_list[d] = list(self.class_list[d])
            if self.verbose:
                print('Finish constructing ' + d + ' data, total %d classes.' % len(self.class_list[d]))

    def get_batch(self, metasplit):
        """N-way K-shot"""
        if metasplit == 'train':
            b_size = self.batch_size
        elif metasplit == 'val':
            b_size = self.val_batch_size
        else:
            b_size = self.test_batch_size
        K = self.shot
        N = self.way
        val_steps = self.meta_val_steps

        datasplit = ['train', 'val']
        batch = {}
        for d in datasplit:
            batch[d] = {'input': [], 'target': [], 'name': []}

        for _ in range(b_size):
            shuffled_classes = self.class_list[metasplit].copy()
            random.shuffle(shuffled_classes)

            shuffled_classes = shuffled_classes[:N]

            inp = {'train': [[] for i in range(N)], 'val': [[] for i in range(N)]}
            tgt = {'train': [[] for i in range(N)], 'val': [[] for i in range(N)]}

            for c, class_name in enumerate(shuffled_classes):
                images = np.random.choice(self.image_by_class[metasplit][class_name], K + val_steps)
                image_names = {'train': images[:K], 'val': images[K:]}

                for d in datasplit:
                    num_images = K if d == 'train' else val_steps
                    assert len(image_names[d]) == num_images
                    for i in range(num_images):
                        embed = self.embed_by_name[metasplit][image_names[d][i]]
                        inp[d][c].append(embed)
                        tgt[d][c].append(c)

            for d in datasplit:
                num_images = K if d == 'train' else val_steps

                assert len(inp['train']) == N
                assert len(inp['val']) == N

                permutations = list(itertools.permutations(range(N)))
                order = random.choice(permutations)
                inputs = [inp[d][i] for i in order]
                target = [tgt[d][i] for i in order]

                batch[d]['input'].append(np.asarray(inputs).reshape(N, num_images, -1))
                batch[d]['target'].append(np.asarray(target).reshape(N, num_images, -1))

        # convert to tensor
        for d in datasplit:
            num_images = K if d == 'train' else val_steps
            normalized_input = Tensor(np.array(batch[d]['input']), mindspore.float32)
            batch[d]['input'] = ops.L2Normalize(axis=-1)(normalized_input)
            batch[d]['target'] = Tensor.from_numpy(np.array(batch[d]['target']))

            assert batch[d]['input'].shape == (b_size, N, num_images, self.embedding_size)
            assert batch[d]['target'].shape == (b_size, N, num_images, 1)
        return batch
