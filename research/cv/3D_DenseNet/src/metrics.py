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
"""
This module defines evaluation metrics for training and validation
"""
import numpy as np

def eval_dice(im1, im2, tid):
    """This function is used in eval.py"""
    im1 = im1 == tid
    im2 = im2 == tid
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    #Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dsc = 2. * intersection.sum() / (im1.sum() + im2.sum())
    return dsc


def train_dice(im1, im2, tid):
    """
    This function is used in eval_call_back.py,
    in training, the output results need to be post-processed before  evaluation
    """
    im2 = np.argmax(im2, axis=1)
    im1 = im1 == tid
    im2 = im2 == tid
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    #Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dsc = 2. * intersection.sum() / (im1.sum() + im2.sum())
    return dsc
