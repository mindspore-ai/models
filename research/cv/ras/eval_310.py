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
"""eval process for 310 inference"""
import os
import argparse
import numpy as np
from PIL import Image

def parse(arg=None):
    """Define configuration of postprocess"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, default='./result_Files/')
    parser.add_argument('--gt_dir', type=str)
    return parser.parse_args(arg)

def image_loader(imagename):
    """load image from file"""
    image = Image.open(imagename).convert("L")
    return np.array(image)

def Fmeasure(predict_, groundtruth):
    """
    Args:
        predict: predict image
        gt: ground truth

    Returns:
        Calculate F-measure
    """
    sumLabel = 2 * np.mean(predict_)
    if sumLabel > 1:
        sumLabel = 1
    Label3 = predict_ >= sumLabel
    NumRec = np.sum(Label3)
    LabelAnd = Label3
    gt_t = groundtruth > 0.5
    NumAnd = np.sum(LabelAnd * gt_t)
    num_obj = np.sum(groundtruth)
    if NumAnd == 0:
        p = 0
        r = 0
        FmeasureF = 0
    else:
        p = NumAnd / NumRec
        r = NumAnd / num_obj
        FmeasureF = (1.3 * p * r) / (0.3 * p + r)
    return FmeasureF

def eval310():
    """evaluation"""
    gtfiles = sorted([args.gt_dir + gt_file for gt_file in os.listdir(args.gt_dir)])
    predictfiles = sorted([os.path.join(args.pred_dir, predictfile) for predictfile in os.listdir(args.pred_dir)])

    #calculate F-measure
    Fs = []
    for i in range(len(gtfiles)):
        gt = image_loader(gtfiles[i]) / 255
        predict = image_loader(predictfiles[i]) / 255
        fmea = Fmeasure(predict, gt)
        Fs.append(fmea)

    print("Fmeasure is %.3f" % np.mean(Fs))

if __name__ == "__main__":
    args = parse()
    eval310()
