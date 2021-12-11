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
"""base dataloader"""

import csv
import numpy as np

def getSoft(size, xb, yb, boundwidth=5.0):
    """getSoft"""
    xarray = np.tile(np.arange(0, size[1]), (size[0], 1))
    yarray = np.tile(np.arange(0, size[0]), (size[1], 1)).transpose()
    cxdists = []
    cydists = []
    for i in range(len(xb)):
        xba = np.tile(xb[i], (size[1], 1)).transpose()
        yba = np.tile(yb[i], (size[0], 1))
        cxdists.append(np.abs(xarray - xba))
        cydists.append(np.abs(yarray - yba))
    xdist = np.minimum.reduce(cxdists)
    ydist = np.minimum.reduce(cydists)
    manhdist = np.minimum.reduce([xdist, ydist])
    im = (manhdist + 1) / (boundwidth + 1) * 1.0
    im[im >= 1.0] = 1.0
    return im

def getfeats(featpath):
    """getfeats"""
    trans_points = np.empty([5, 2], dtype=np.int64)
    with open(featpath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for ind, row in enumerate(reader):
            trans_points[ind, :] = row
    return trans_points

def tocv2(ts):
    """tocv2"""
    img = (ts.asnumpy() / 2 + 0.5) * 255
    img = img.astype('uint8')
    img = np.transpose(img, (1, 2, 0))
    img = img[:, :, ::-1]  # rgb->bgr
    return img

def dt(img):
    """dt"""
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert to BW
    _, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    dt1 = cv2.distanceTransform(thresh1, cv2.DIST_L2, 5)
    dt2 = cv2.distanceTransform(thresh2, cv2.DIST_L2, 5)
    dt1 = dt1 / dt1.max()  # ->[0,1]
    dt2 = dt2 / dt2.max()
    return dt1, dt2
