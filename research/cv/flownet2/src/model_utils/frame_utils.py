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

import re
from os.path import splitext
import numpy as np
from imageio import imread

def read_gen(file_name):
    ext = splitext(file_name)[-1]
    if ext in ('.png', '.jpeg', '.ppm', '.jpg'):
        im = imread(file_name)
        if im.shape[2] > 3:
            return im[:, :, :3]
        return im
    if ext in ('.bin', '.raw'):
        return np.load(file_name)
    if ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    if ext == '.pfm':
        return readPFM(file_name).astype(np.float32)
    return []

def  readFlow(fn):
    """ Read .flo file in Middlebury format"""
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            print('Magic number incorrect. Invalid .flo file')
            return None
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
        return np.resize(data, (int(h), int(w), 2))

def readPFM(file):
    file = open(file, 'rb')
    header = file.readline().rstrip()
    if header in ('PF', b'PF'):
        color = True
    elif header in ('Pf', b'Pf'):
        color = False
    else:
        raise Exception('Not a PFM file.')
    wh = bytes.decode(file.readline())
    dim_match = re.match(r'^(\d+)\s+(\d+)$', wh.strip())

    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float(file.readline().decode().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data
