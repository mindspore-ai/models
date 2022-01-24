#!/bin/bash
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

import os
import itertools
import numpy as np

import skimage
from PIL import Image
import gin
import cv2

import mindspore.numpy as mnp

@gin.configurable
def SameTrCollate(image, prjAug, prjVal):
    image = mnp.transpose(image, (1, 2, 0))
    image = mnp.transpose(image, (2, 0, 1))

    return image

class myLoadDS:
    def __init__(self, flist, dpath, ralph=None, fmin=True, mln=None):
        if flist is not None and dpath is not None:
            self.fns = get_files(flist, dpath)
            self.tlbls = get_labels(self.fns)
            alph = get_alphabet(None)
            self.ralph = dict(zip(alph.values(), alph.keys()))
            self.alph = alph
            self.tlbls = [np.array([self.alph[c] if c in alph.keys() else 0 for c in line]) for line in self.tlbls]
        else:
            self.fns = None
            self.tlbls = None
            alph = get_alphabet(None)
            self.ralph = dict(zip(alph.values(), alph.keys()))
            self.alph = alph
        if mln is not None:
            filt = [len(x) <= mln if fmin else len(x) >= mln for x in self.tlbls]
            self.tlbls = np.asarray(self.tlbls)[filt].tolist()
            self.fns = np.asarray(self.fns)[filt].tolist()
    def __len__(self):
        return len(self.fns)

    def __getitem__(self, index):
        timgs = get_images(self.fns[index])
        h = timgs.shape[0]
        w = timgs.shape[1]
        timgs = cv2.resize(timgs, (w, h))
        timgs = timgs.transpose((2, 0, 1))

        return timgs, self.tlbls[index]

def get_files(nfile, dpath):
    fnames = open(nfile, 'r').readlines()
    fnames = [dpath + x.strip() for x in fnames]
    return fnames

def npThum(img, max_w, max_h):
    x, y = np.shape(img)[:2]

    y = min(int(y * max_h / x), max_w)
    x = max_h
    img = np.array(Image.fromarray(img).resize((y, x)))
    return img

@gin.configurable
def get_images(fname, max_w, max_h, nch):

    try:

        image_data = np.array(Image.open(fname))
        image_data = npThum(image_data, max_w, max_h)
        image_data = skimage.img_as_float32(image_data)

        if image_data.ndim < 3:
            image_data = np.expand_dims(image_data, axis=-1)

        if nch == 3 and image_data.shape[2] != 3:
            image_data = np.tile(image_data, 3)
        image_data = np.pad(image_data, ((0, 0), (0, max_w-np.shape(image_data)[1]), (0, 0)),
                            mode='constant', constant_values=(1.0))
    except IOError as e:
        print('Could not read:', fname, ':', e)

    return image_data

def get_labels(fnames):

    labels = []
    for image_file in fnames:
        fn = os.path.splitext(image_file)[0] + '.txt'
        lbl = open(fn, 'r', encoding="utf-8").read()
        lbl = ''.join(lbl.split()) #remove linebreaks if present
        labels.append(lbl)

    return labels

def get_alphabet(labels):
    if labels is not None:
        coll = ''.join(labels)
        unq = sorted(list(set(coll)))
        unq = [''.join(i) for i in itertools.product(unq, repeat=1)]
        alph = dict(zip(unq, range(len(unq))))
    else:
        alph = dict()
        for line in open('parameters/alph.gc', 'r', encoding='UTF-8'):
            chara = line.replace('\n', '').split('/')
            alph[chara[0]] = int(chara[1])

    return alph
