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
import os
import random
from io import BytesIO
import numpy as np
from PIL import Image, ImageFilter
import mindspore.dataset as ds

EXTENSIONS = ['.jpg', '.png']

class MyGaussianBlur(ImageFilter.Filter):

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        return image.gaussian_blur(self.radius)

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png")

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class MyCoTransform:

    def __init__(self, stage, enc, augment, height, if_from_mindrecord=False):
        self.enc = enc
        self.augment = augment
        self.height = height
        self.if_from_mindrecord = if_from_mindrecord

        if not stage in (1, 2, 3, 4):
            raise RuntimeError("stage should be 1, 2, 3 or 4")
        self.stage = stage

        if self.stage == 1:
            self.ratio = 1.2
        else:
            self.ratio = 1.3

    def process_one(self, image, target, height):
        if self.augment:

            # GaussianBlur
            image = image.filter(MyGaussianBlur(radius=random.random()))

            if random.random() > 0.5: # random crop
                if self.stage == 1:
                    ratio = self.ratio # 1阶段使用
                else:
                    ratio = random.random() * (self.ratio - 1) + 1

                w = int(2048 / ratio)
                h = int(1024 / ratio)

                x = int(random.random()*(2048-w))
                y = int(random.random()*(1024-h))

                box = (x, y, x+w, y+h)
                image = image.crop(box)
                target = target.crop(box)
            img_h, img_w = image.size
            image = image.resize((height * img_h // img_w, height), Image.BILINEAR)
            target = target.resize((height * img_h // img_w, height), Image.NEAREST)

            # Random hflip
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img_h, img_w = image.size
            image = image.resize((height * img_h // img_w, height), Image.BILINEAR)
            target = target.resize((height * img_h // img_w, height), Image.NEAREST)

        image = np.array(image).astype(np.float32) / 255
        image = image.transpose(2, 0, 1)

        target = target.resize((target.size[0] // 8, target.size[1] // 8), Image.NEAREST) if self.enc else target
        target = np.array(target).astype(np.uint32)
        target[target == 255] = 19
        return image, target

    def process_one_infer(self, image, height):
        img_h, img_w = image.size
        image = image.resize((height * img_h // img_w, height), Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255
        image = image.transpose(2, 0, 1)
        return image

    def __call__(self, image, target=None):
        if self.if_from_mindrecord:
            image = Image.open(BytesIO(image))
            target = Image.open(BytesIO(target))
        if target is None:
            image = self.process_one_infer(image, self.height)
            return image
        image, target = self.process_one(image, target, self.height)
        return image, target

class cityscapes:

    def __init__(self, root, subset, enc, aug, height):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')

        self.images_root += subset
        self.labels_root += subset

        self.filenames = [os.path.join(dp, f) for dp, dn, fn in \
            os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in \
            os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

        self.transform = MyCoTransform(1, enc, aug, height)

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')
        image, label = self.transform(image, label)
        return image, label

    def __len__(self):
        return len(self.filenames)

class cityscapes_datapath:

    def __init__(self, root, subset):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')

        self.images_root += subset
        self.labels_root += subset

        self.filenames = [os.path.join(dp, f) for dp, dn, fn in \
            os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in \
            os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        return filename, filenameGt

    def __len__(self):
        return len(self.filenames)

def getCityScapesDataLoader_GeneratorDataset(CityScapesRoot, subset, batch_size, \
    enc, height, shuffle, aug, rank_id=0, global_size=1, repeat=1):

    dataset = cityscapes(CityScapesRoot, subset, enc, aug, height)
    dataloader = ds.GeneratorDataset(dataset, column_names=["images", "labels"], \
        num_parallel_workers=8, shuffle=shuffle, shard_id=rank_id, \
        num_shards=global_size, python_multiprocessing=True)
    if shuffle:
        dataloader = dataloader.shuffle(batch_size*10)
    dataloader = dataloader.batch(batch_size, drop_remainder=False)
    if repeat > 1:
        dataloader = dataloader.repeat(repeat)
    return dataloader

def getCityScapesDataLoader_mindrecordDataset(stage, data_path, batch_size, enc, height, \
    shuffle, aug, rank_id=0, global_size=1, repeat=1):

    dataloader = ds.MindDataset(data_path, columns_list=["data", "label"], \
        num_parallel_workers=4, shuffle=shuffle, shard_id=rank_id, num_shards=global_size)
    transform = MyCoTransform(stage, enc, aug, height, if_from_mindrecord=True)
    dataloader = dataloader.map(operations=transform, \
        input_columns=["data", "label"], output_columns=["data", "label"], \
        num_parallel_workers=12, python_multiprocessing=True)

    if shuffle:
        dataloader = dataloader.shuffle(batch_size*10)

    dataloader = dataloader.batch(batch_size, drop_remainder=True)
    if repeat > 1:
        dataloader = dataloader.repeat(repeat)
    return dataloader

class InferDataSet:

    def __init__(self, img_path, height):
        self.imgs_path = [os.path.join(img_path, img) for img in os.listdir(img_path)]
        self.transform = MyCoTransform(1, False, False, 512)

    def __getitem__(self, index):
        with open(self.imgs_path[index], 'rb') as f:
            image = load_image(f).convert('RGB')
        image = self.transform(image)
        return (image,)

    def __len__(self):
        return len(self.imgs_path)

def getInferDataLoader_fromfile(img_path, batch_size, height):
    shuffle = False
    global_size = 1
    rank_id = 0
    dataset = InferDataSet(img_path, height)
    dataloader = ds.GeneratorDataset(dataset, column_names=["images"], \
        num_parallel_workers=8, shuffle=shuffle, shard_id=rank_id, \
        num_shards=global_size, python_multiprocessing=True)
    dataloader = dataloader.batch(batch_size, drop_remainder=False)
    return dataloader
