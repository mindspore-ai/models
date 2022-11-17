# coding: utf-8
"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import json
import os

import numpy as np


def get_file_name(file_path):
    return os.path.splitext(os.path.basename(file_path.rstrip('/')))[0]


def load_gt(gt_file):
    gt = {}
    with open(gt_file, 'r') as fd:
        for line in fd.readlines():
            img_name, img_label_index = line.strip().split(" ", 1)
            gt[get_file_name(img_name)] = img_label_index
    return gt


def load_pred(pred_file):
    pred = {}
    with open(pred_file, 'r') as fd:
        for line in fd.readlines():
            ret = line.strip().split(" ", 1)
            if len(ret) < 2:
                print(f"Warning: load pred, no result, line:{line}")
                continue
            img_name, ids = ret
            img_name = get_file_name(img_name)
            pred[img_name] = [x.strip() for x in ids.split(',')]
    return pred


def calc_accuracy(gt_map, pred_map, top_k=5):
    hits = [0] * top_k
    miss_match = []
    total = 0
    for img, preds in pred_map.items():
        gt = gt_map.get(img)
        if gt is None:
            print(f"Warning: {img}'s gt is not exists.")
            continue
        try:
            index = preds.index(gt, 0, top_k)
            hits[index] += 1
        except ValueError:
            miss_match.append({'img': img, 'gt': gt, 'prediction': preds})
        finally:
            total += 1

    top_k_hit = np.cumsum(hits)
    accuracy = top_k_hit / total
    return {
        'total': total,
        'accuracy': [acc for acc in accuracy],
        'miss': miss_match,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('prediction', help='prediction result file')
    parser.add_argument('gt', help='ground true result file')
    parser.add_argument('result_json', help='metric result file')
    parser.add_argument('top_k', help='top k', type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    prediction_file = args.prediction
    gt_file = args.gt
    top_k = args.top_k
    result_json = args.result_json

    gt = load_gt(gt_file)
    prediction = load_pred(prediction_file)
    result = calc_accuracy(gt, prediction, top_k)
    result.update({
        'prediction_file': prediction_file,
        'gt_file': gt_file,
    })
    with open(result_json, 'w') as fd:
        json.dump(result, fd, indent=2)
        print(f"\nsuccess, result in {result_json}")


if __name__ == '__main__':
    main()
