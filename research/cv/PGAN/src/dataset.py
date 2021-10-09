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
"""dataset"""
import os
from src.image_transform import pil_loader


class ImageDataset:
    """
    A dataset class adapted to the specificites of our YSL fashion dataset.
    It loads both the images and an attribute dictionary describing each image's
    attribute.
    """

    def __init__(self,
                 pathdb,
                 transform=None):
        """
        Args:
            - pathdb (string): path to the directory containing the images
            - transform (torchvision.transforms): a function object list to convert image to np array

        """
        self.totAttribSize = 0
        self.pathdb = pathdb
        self.transforms = transform
        imagesPaths = os.listdir(pathdb)
        self.listImg = [imgName for imgName in imagesPaths
                        if os.path.splitext(imgName)[1] in [".jpg", ".png"]]
        print("%d images found" % len(self))

    def __len__(self):
        return len(self.listImg)

    def __getitem__(self, idx):
        imgName = self.listImg[idx]
        imgPath = os.path.join(self.pathdb, imgName)
        img = pil_loader(imgPath)
        if self.transforms is not None:
            for transform in self.transforms:
                img = transform(img)
        return img, 1

    def getName(self, idx):
        return self.listImg[idx]
