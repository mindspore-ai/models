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
import math
from os.path import join
from os.path import isfile
from glob import glob
import numpy as np
from src.model_utils.frame_utils import read_gen


class StaticRandomCrop():
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1 + self.th), self.w1:(self.w1 + self.tw), :]


class StaticCenterCrop():
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size

    def __call__(self, img):
        return img[(self.h - self.th) // 2:(self.h + self.th) // 2, (self.w - self.tw) // 2:(self.w + self.tw) // 2, :]


class DistributedSampler():
    """
  Distributed sampler
  """

    def __init__(self, dataset, rank, group_size, shuffle=True, seed=0):
        self.dataset = dataset
        self.rank = rank
        self.group_size = group_size
        self.dataset_length = len(self.dataset)
        self.num_samples = int(math.ceil(self.dataset_length * 1.0 / self.group_size))
        self.total_size = self.num_samples * self.group_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.seed = (self.seed + 1) & 0xffffffff
            np.random.seed(self.seed)
            indices = np.random.permutation(self.dataset_length).tolist()
        else:
            indices = list(range(len(self.dataset_length)))
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        indices = indices[self.rank::self.group_size]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class MpiSintel:
    def __init__(self, crop_type='Random', crop_size=None, eval_size=None,
                 root='', dstype='clean', replicates=1):
        self.crop_type = crop_type
        if crop_size is None:
            crop_size = [384, 512]
        self.crop_size = crop_size
        if eval_size is None:
            eval_size = [256, 256]
        self.render_size = eval_size
        self.replicates = replicates

        flow_root = join(root, 'flow')
        image_root = join(root, dstype)

        file_list = sorted(glob(join(flow_root, '*/*.flo')))

        self.flow_list = []
        self.image_list = []

        for file in file_list:
            if 'test' in file:
                # print file
                continue

            fbase = file[len(flow_root) + 1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])

            img1 = join(image_root, fprefix + "%04d" % (fnum + 0) + '.png')
            img2 = join(image_root, fprefix + "%04d" % (fnum + 1) + '.png')

            if not isfile(img1) or not isfile(img2) or not isfile(file):
                continue

            self.image_list += [[img1, img2]]
            self.flow_list += [file]

        self.size = len(self.image_list)

        self.frame_size = read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0] % 64) or (
                self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0]) // 64) * 64
            self.render_size[1] = ((self.frame_size[1]) // 64) * 64

        # args.eval_size = self.render_size

        assert len(self.image_list) == len(self.flow_list)

    def __getitem__(self, index):

        index = index % self.size

        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])

        flow = read_gen(self.flow_list[index])

        images = [img1, img2]
        image_size = img1.shape[:2]

        if self.crop_type == 'Random':
            cropper = StaticRandomCrop(image_size, self.crop_size)
        elif self.crop_type == 'Center':
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3, 0, 1, 2)
        flow = flow.transpose(2, 0, 1)

        images = images.astype(np.float32)
        flow = flow.astype(np.float32)

        return images, flow

    def __len__(self):
        return self.size * self.replicates


class MpiSintelClean(MpiSintel):
    def __init__(self, crop_type, crop_size, eval_size, root, replicates=1):
        super(MpiSintelClean, self).__init__(crop_type=crop_type, crop_size=crop_size, eval_size=eval_size,
                                             root=root, dstype='clean', replicates=replicates)


class MpiSintelFinal(MpiSintel):
    def __init__(self, crop_type, crop_size, eval_size, root, replicates=1):
        super(MpiSintelFinal, self).__init__(crop_type=crop_type, crop_size=crop_size, eval_size=eval_size,
                                             root=root, dstype='final', replicates=replicates)


# definite a DatasetGenerator
class ChairsSDHom:
    def __init__(self, crop_type, crop_size, eval_size, root='/path/to/chairssdhom/data', dstype='train', replicates=1):
        self.crop_type = crop_type
        self.crop_size = crop_size
        self.render_size = eval_size
        self.replicates = replicates

        image1 = sorted(glob(join(root, dstype, 't0/*.png')))
        image2 = sorted(glob(join(root, dstype, 't1/*.png')))
        self.flow_list = sorted(glob(join(root, dstype, 'flow/*.pfm')))

        assert len(image1) == len(self.flow_list)

        self.image_list = []
        for i in range(len(self.flow_list)):
            im1 = image1[i]
            im2 = image2[i]
            self.image_list += [[im1, im2]]

        assert len(self.image_list) == len(self.flow_list)

        self.size = len(self.image_list)

        self.frame_size = read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0] % 64) or (
                self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0]) // 64) * 64
            self.render_size[1] = ((self.frame_size[1]) // 64) * 64

        # args.eval_size = self.render_size

    def __getitem__(self, index):
        index = index % self.size

        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])

        flow = read_gen(self.flow_list[index])
        flow = flow[::-1, :, :]

        images = [img1, img2]
        image_size = img1.shape[:2]
        if self.crop_type == 'Random':
            cropper = StaticRandomCrop(image_size, self.crop_size)
        elif self.crop_type == 'Center':
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3, 0, 1, 2)
        flow = flow.transpose(2, 0, 1)

        images = images.astype(np.float32)
        flow = flow.astype(np.float32)
        return images, flow

    def __len__(self):
        return self.size * self.replicates


class ChairsSDHomTrain(ChairsSDHom):
    def __init__(self, crop_type, crop_size, eval_size, root='', replicates=1):
        super(ChairsSDHomTrain, self).__init__(crop_type=crop_type, crop_size=crop_size, eval_size=eval_size,
                                               root=root, dstype='train', replicates=replicates)


class ChairsSDHomTest(ChairsSDHom):
    def __init__(self, crop_type, crop_size, eval_size, root='', replicates=1):
        super(ChairsSDHomTest, self).__init__(crop_type=crop_type, crop_size=crop_size, eval_size=eval_size, root=root,
                                              dstype='test', replicates=replicates)


class FlyingChairs:
    def __init__(self, crop_type, crop_size, eval_size, root='/path/to/FlyingChairs_release/data', replicates=1):
        self.crop_type = crop_type
        self.crop_size = crop_size
        self.render_size = eval_size
        self.replicates = replicates

        images = sorted(glob(join(root, '*.ppm')))

        self.flow_list = sorted(glob(join(root, '*.flo')))

        assert len(images) // 2 == len(self.flow_list)

        self.image_list = []
        for i in range(len(self.flow_list)):
            im1 = images[2 * i]
            im2 = images[2 * i + 1]
            self.image_list += [[im1, im2]]

        assert len(self.image_list) == len(self.flow_list)

        self.size = len(self.image_list)

        self.frame_size = read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0] % 64) or (
                self.frame_size[1] % 64):
            self.render_size[0] = ((self.frame_size[0]) // 64) * 64
            self.render_size[1] = ((self.frame_size[1]) // 64) * 64

        # args.eval_size = self.render_size

    def __getitem__(self, index):
        index = index % self.size

        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])

        flow = read_gen(self.flow_list[index])

        images = [img1, img2]
        image_size = img1.shape[:2]
        if self.crop_type == 'Random':
            cropper = StaticRandomCrop(image_size, self.crop_size)
        elif self.crop_type == 'Center':
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3, 0, 1, 2)
        flow = flow.transpose(2, 0, 1)

        images = images.astype(np.float32)
        flow = flow.astype(np.float32)
        return images, flow

    def __len__(self):
        return self.size * self.replicates
