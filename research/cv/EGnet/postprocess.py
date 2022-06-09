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
"""post process for 310 inference"""
import os
import argparse
import cv2
import numpy as np
from PIL import Image

def parse(arg=None):
    """Define configuration of postprocess"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin_path', type=str, default='./result_Files/')
    parser.add_argument('--mask_path', type=str, default='./preprocess_Mask_Result/')
    parser.add_argument('--output_dir', type=str, default='./postprocess_Result/')
    return parser.parse_args(arg)

class Metric:
    """
    for metric
    """

    def __init__(self):
        self.epsilon = 1e-4
        self.beta = 0.3
        self.thresholds = 256
        self.mae = 0
        self.max_f = 0
        self.precision = np.zeros(self.thresholds)
        self.recall = np.zeros(self.thresholds)
        self.q = 0
        self.cnt = 0

    def update(self, pred, gt):
        assert pred.shape == gt.shape
        pred = pred.astype(np.float32)
        gt = gt.astype(np.float32)
        norm_pred = pred / 255.0
        norm_gt = gt / 255.0
        self.compute_mae(norm_pred, norm_gt)
        self.compute_precision_and_recall(pred, gt)
        self.compute_s_measure(norm_pred, norm_gt)
        self.cnt += 1

    def print_result(self):
        f_measure = (1 + self.beta) * (self.precision * self.recall) / (self.beta * self.precision + self.recall)
        argmax = np.argmax(f_measure)
        print("Max F-measure:", f_measure[argmax] / self.cnt)
        print("Precision:    ", self.precision[argmax] / self.cnt)
        print("Recall:       ", self.recall[argmax] / self.cnt)
        print("MAE:          ", self.mae / self.cnt)
        print("S-measure:    ", self.q / self.cnt)

    def compute_precision_and_recall(self, pred, gt):
        """
        compute the precision and recall for pred
        """
        for th in range(self.thresholds):
            a = np.zeros_like(pred).astype(np.int32)
            b = np.zeros_like(pred).astype(np.int32)
            a[pred > th] = 1
            a[pred <= th] = 0
            b[gt > th / self.thresholds] = 1
            b[gt <= th / self.thresholds] = 0
            ab = np.sum(np.bitwise_and(a, b))
            a_sum = np.sum(a)
            b_sum = np.sum(b)
            self.precision[th] += (ab + self.epsilon) / (a_sum + self.epsilon)
            self.recall[th] += (ab + self.epsilon) / (b_sum + self.epsilon)

    def compute_mae(self, pred, gt):
        """
        compute mean average error
        """
        self.mae += np.abs(pred - gt).mean()

    def compute_s_measure(self, pred, gt):
        """
        compute s measure score
        """

        alpha = 0.5
        y = gt.mean()
        if y == 0:
            x = pred.mean()
            q = 1.0 - x
        elif y == 1:
            x = pred.mean()
            q = x
        else:
            gt[gt >= 0.5] = 1
            gt[gt < 0.5] = 0
            q = alpha * self._s_object(pred, gt) + (1 - alpha) * self._s_region(pred, gt)
            if q < 0 or np.isnan(q):
                q = 0
        self.q += q

    def _s_object(self, pred, gt):
        """
        score of object
        """
        fg = np.where(gt == 0, np.zeros_like(pred), pred)
        bg = np.where(gt == 1, np.zeros_like(pred), 1 - pred)
        o_fg = self._object(fg, gt)
        o_bg = self._object(bg, 1 - gt)
        u = gt.mean()
        q = u * o_fg + (1 - u) * o_bg
        return q

    @staticmethod
    def _object(pred, gt):
        """
        compute score of object
        """
        temp = pred[gt == 1]
        if temp.size == 0:
            return 0
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
        return score

    def _s_region(self, pred, gt):
        """
        compute score of region
        """
        x, y = self._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divide_gt(gt, x, y)
        p1, p2, p3, p4 = self._divide_prediction(pred, x, y)
        q1 = self._ssim(p1, gt1)
        q2 = self._ssim(p2, gt2)
        q3 = self._ssim(p3, gt3)
        q4 = self._ssim(p4, gt4)
        q = w1 * q1 + w2 * q2 + w3 * q3 + w4 * q4
        return q

    @staticmethod
    def _divide_gt(gt, x, y):
        """
        divide ground truth image
        """
        if not isinstance(x, np.int64):
            x = x[0][0]
        if not isinstance(y, np.int64):
            y = y[0][0]
        h, w = gt.shape[-2:]
        area = h * w
        gt = gt.reshape(h, w)
        lt = gt[:y, :x]
        rt = gt[:y, x:w]
        lb = gt[y:h, :x]
        rb = gt[y:h, x:w]
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        w1 = x * y / area
        w2 = (w - x) * y / area
        w3 = x * (h - y) / area
        w4 = 1 - w1 - w2 - w3
        return lt, rt, lb, rb, w1, w2, w3, w4

    @staticmethod
    def _divide_prediction(pred, x, y):
        """
        divide predict image
        """
        if not isinstance(x, np.int64):
            x = x[0][0]
        if not isinstance(y, np.int64):
            y = y[0][0]
        h, w = pred.shape[-2:]
        pred = pred.reshape(h, w)
        lt = pred[:y, :x]
        rt = pred[:y, x:w]
        lb = pred[y:h, :x]
        rb = pred[y:h, x:w]
        return lt, rt, lb, rb

    @staticmethod
    def _ssim(pred, gt):
        """
        structural similarity
        """
        gt = gt.astype(np.float32)
        h, w = pred.shape[-2:]
        n = h * w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x) * (pred - x)).sum() / (n - 1 + 1e-20)
        sigma_y2 = ((gt - y) * (gt - y)).sum() / (n - 1 + 1e-20)
        sigma_xy = ((pred - x) * (gt - y)).sum() / (n - 1 + 1e-20)

        alpha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

        if alpha != 0:
            q = alpha / (beta + 1e-20)
        elif alpha == 0 and beta == 0:
            q = 1.0
        else:
            q = 0
        return q

    @staticmethod
    def _centroid(gt):
        """
        compute center of ground truth image
        """
        rows, cols = gt.shape[-2:]
        gt = gt.reshape(rows, cols)
        if gt.sum() == 0:
            x = np.eye(1) * round(cols / 2)
            y = np.eye(1) * round(rows / 2)
        else:
            total = gt.sum()

            i = np.arange(0, cols).astype(np.float32)
            j = np.arange(0, rows).astype(np.float32)
            x = np.round((gt.sum(axis=0) * i).sum() / total)
            y = np.round((gt.sum(axis=1) * j).sum() / total)
        return x.astype(np.int64), y.astype(np.int64)

def load_bin_file(bin_file, shape=None, dtype="float32"):
    """Load data from bin file"""
    data = np.fromfile(bin_file, dtype=dtype)
    if shape:
        data = np.reshape(data, shape)
    return data

def save_bin_to_image(data, out_name):
    """Save bin file to image arrays"""
    pic = Image.fromarray(data)
    pic = pic.convert('RGB')
    pic.save(out_name)
    print("Successfully save image in " + out_name)

def scan_dir(bin_path):
    """Scan directory"""
    out = os.listdir(bin_path)
    return out

def sigmoid(z):
    """sigmoid"""
    return 1/(1 + np.exp(-z))

def postprocess(args):
    """Post process bin file"""
    file_list = scan_dir(args.bin_path)
    metric = Metric()
    has_label = False
    for file_path in file_list:
        data = load_bin_file(args.bin_path + file_path, shape=(200, 200), dtype="float32")
        sal_label = load_bin_file(args.mask_path + file_path, shape=(200, 200), dtype="float32")
        img = sigmoid(data).squeeze() * 255
        file_name = file_path.split(".")[0] + ".jpg"
        outfile = os.path.join(args.output_dir, file_name)
        cv2.imwrite(outfile, img)
        if sal_label is not None:
            has_label = True
            sal_label = sal_label.squeeze() * 255
            img = np.round(img).astype(np.uint8)
            metric.update(img, sal_label)
            metric.print_result()
        print("postprocess image index ", file_name, " done")
    if has_label:
        metric.print_result()

if __name__ == "__main__":
    argms = parse()
    postprocess(argms)
