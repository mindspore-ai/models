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
"""postprocess for 310 inference"""
import os
import numpy as np
import mindspore
import mindspore.ops as ops
import src.utils.img
from src.config import parse_args
from src.utils.inference import MPIIEval, get_img

args = parse_args()


def match_format(dic):
    """
    get match format
    """
    loc = dic["loc_k"][0, :, 0, :]
    val = dic["val_k"][0, :, :]
    ans0 = np.hstack((loc, val))
    ans1 = np.expand_dims(ans0, axis=0)
    ret = []
    ret.append(ans1)
    return ret


class HeatmapParser:
    """
    parse heatmap
    """

    def __init__(self):
        self.topk = ops.TopK(sorted=True)
        self.stack = ops.Stack(axis=3)

    def calc(self, maxm):
        """
        calc distance
        """
        det = mindspore.Tensor(maxm)
        w = det.shape[3]
        det = det.view(det.shape[0], det.shape[1], -1)
        val_k, ind = self.topk(det, 1)

        x = ind % w
        y = (ind.astype(mindspore.float32) / w).astype(mindspore.int32)
        ind_k = self.stack((x, y))
        answer = {"loc_k": ind_k, "val_k": val_k}
        return {key: answer[key].asnumpy() for key in answer}

    def adjust(self, answer, det):
        """
        adjust for joint
        """
        for people in answer:
            for ii in people:
                for joint_id, joint in enumerate(ii):
                    if joint[2] > 0:
                        y, x = joint[0:2]
                        xx, yy = int(x), int(y)
                        tmp = det[0][joint_id]
                        if tmp[xx, min(yy + 1, tmp.shape[1] - 1)] > tmp[xx, max(yy - 1, 0)]:
                            y += 0.25
                        else:
                            y -= 0.25

                        if tmp[min(xx + 1, tmp.shape[0] - 1), yy] > tmp[max(0, xx - 1), yy]:
                            x += 0.25
                        else:
                            x -= 0.25
                        answer[0][0, joint_id, 0:2] = (y + 0.5, x + 0.5)
        return answer

    def parse(self, det, maxm, adjust=True):
        """
        parse heatmap
        """
        ans0 = match_format(self.calc(maxm))
        if adjust:
            ans1 = self.adjust(ans0, det)
        return ans1


parser = HeatmapParser()


def post_process(det, maxm, mat_, trainval, cc=None, ss=None, resolution=None):
    """
    post process for parser
    """
    mat = np.linalg.pinv(np.array(mat_).tolist() + [[0, 0, 1]])[:2]
    cropped_preds = parser.parse(mindspore.Tensor(np.float32([det])), maxm)[0]

    if cropped_preds.size > 0:
        cropped_preds[:, :, :2] = src.utils.img.kpt_affine(cropped_preds[:, :, :2] * 4, mat)  # size 1x16x3

    predict = np.copy(cropped_preds)
    # revert to origin image
    if trainval != "cropped":
        for jj in range(predict.shape[1]):
            predict[0, jj, :2] = src.utils.img.transform(predict[0, jj, :2], cc, ss, resolution, invert=1)
    return predict


def infer(image, jj, cc, ss):
    """
    forward pass at test time
    calls post_process to post process results
    """
    scale_ratio = 200
    height, width = image.shape[0:2]
    center = (width / 2, height / 2)
    scale = max(height, width) / scale_ratio
    res = (args.input_res, args.input_res)

    mat_ = src.utils.img.get_transform(center, scale, res, scale_ratio)[:2]

    f_name0 = os.path.join(args.out_path, "StackedHourglass" + str(jj) + "_0.bin")
    det = np.fromfile(f_name0, np.float32).reshape(16, 64, 64)
    f_name1 = os.path.join(args.out_path, "StackedHourglass" + str(jj) + "_1.bin")
    maxm = np.fromfile(f_name1, np.float32).reshape(1, 16, 64, 64)

    return post_process(det, maxm, mat_, "valid", cc, ss, res)


if __name__ == "__main__":
    gts = []
    preds = []
    normalizing = []

    mindspore.context.set_context(device_target="CPU")

    num_eval = args.num_eval
    num_train = args.train_num_eval
    j = 0
    for anns, img, c, s, n in get_img(num_eval, num_train):
        gts.append(anns)
        ans = infer(img, j, c, s)
        j = j + 1
        if ans.size > 0:
            ans = ans[:, :, :3]

        pred = []
        for i in range(ans.shape[0]):
            pred.append({"keypoints": ans[i, :, :]})
        preds.append(pred)
        normalizing.append(n)

    mpii_eval = MPIIEval()
    mpii_eval.eval(preds, gts, normalizing, num_train)
