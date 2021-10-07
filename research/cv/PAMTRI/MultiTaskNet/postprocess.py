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
"""postprocess"""
import os
import json
import argparse
from pathlib import Path
import numpy as np
from src.utils.evaluate import evaluate

parser = argparse.ArgumentParser(description='Eval MultiTaskNet')

parser.add_argument('--input_dir', type=str, default="")
parser.add_argument('--result_dir', type=str, default="")
parser.add_argument('--label_dir', type=str, default="")

args = parser.parse_args()

def get_cmc_map(_distmat, _q_vids, _g_vids, _q_camids, _g_camids):
    """get_cmc_map"""
    ranks = range(1, 51)
    print("Computing CMC and mAP")
    cmc, mAP = evaluate(_distmat, _q_vids, _g_vids, _q_camids, _g_camids)
    print("Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

def get_color_acc(_distmat, _vcolor2label, _q_vcolors, _pred_q_vcolors, _g_vcolors, _pred_g_vcolors):
    """get_color_acc"""
    print("Compute attribute classification accuracy")
    for q in range(_q_vcolors.size):
        _q_vcolors[q] = _vcolor2label[str(_q_vcolors[q])]
    for g in range(_g_vcolors.size):
        _g_vcolors[g] = _vcolor2label[str(_g_vcolors[g])]
    q_vcolor_errors = np.argmax(_pred_q_vcolors, axis=1) - _q_vcolors
    g_vcolor_errors = np.argmax(_pred_g_vcolors, axis=1) - _g_vcolors
    vcolor_error_num = np.count_nonzero(q_vcolor_errors) + np.count_nonzero(g_vcolor_errors)
    vcolor_accuracy = 1.0 - (float(vcolor_error_num) / float(_distmat.shape[0] + _distmat.shape[1]))
    print("Color classification accuracy: {:.2%}".format(vcolor_accuracy))

def get_type_acc(_distmat, _vcolor2label, _q_vtypes, _pred_q_vtypes, _g_vtypes, _pred_g_vtypes):
    """get_type_acc"""
    for q in range(_q_vtypes.size):
        _q_vtypes[q] = _vcolor2label[str(_q_vtypes[q])]
    for g in range(_g_vtypes.size):
        _g_vtypes[g] = _vcolor2label[str(_g_vtypes[g])]
    q_vtype_errors = np.argmax(_pred_q_vtypes, axis=1) - _q_vtypes
    g_vtype_errors = np.argmax(_pred_g_vtypes, axis=1) - _g_vtypes
    vtype_error_num = np.count_nonzero(q_vtype_errors) + np.count_nonzero(g_vtype_errors)
    vtype_accuracy = 1.0 - (float(vtype_error_num) / float(_distmat.shape[0] + _distmat.shape[1]))
    print("Type classification accuracy: {:.2%}".format(vtype_accuracy))

def acc(input_dir, result_dir, label_dir):
    """get acc"""
    label_path = Path(label_dir + "label.json")
    query_label_path = Path(label_dir + "query_label.json")
    gallery_label_path = Path(label_dir + "gallery_label.json")

    with label_path.open('r') as dst_file:
        label = json.load(dst_file)
    with query_label_path.open('r') as dst_file:
        query_label = json.load(dst_file)
    with gallery_label_path.open('r') as dst_file:
        gallery_label = json.load(dst_file)

    vcolor2label = label['vcolor2label']

    qf = []
    q_vids = []
    q_camids = []
    q_vcolors = []
    q_vtypes = []
    pred_q_vcolors = []
    pred_q_vtypes = []
    for i in range(len(os.listdir(input_dir + "input/query/img"))):
        single_label = query_label["{}".format(i)]
        vids = single_label['vid']
        camids = single_label['camid']
        vcolors = single_label['vcolor']
        vtypes = single_label['vtype']

        output_vcolors_path = result_dir + "veri_data_query_img_{}_1.bin".format(i)
        output_vtypes_path = result_dir + "veri_data_query_img_{}_2.bin".format(i)
        features_path = result_dir + "veri_data_query_img_{}_3.bin".format(i)

        output_vcolors = np.fromfile(output_vcolors_path, dtype=np.float32).reshape(1, 10)
        output_vtypes = np.fromfile(output_vtypes_path, dtype=np.float32).reshape(1, 9)
        features = np.fromfile(features_path, dtype=np.float32).reshape(1, 1024)

        qf.append(features)
        q_vids.extend(vids)
        q_camids.extend(camids)
        q_vcolors.extend(vcolors)
        q_vtypes.extend(vtypes)
        pred_q_vcolors.extend(output_vcolors.tolist())
        pred_q_vtypes.extend(output_vtypes.tolist())

    qf = np.concatenate(qf, axis=0)
    q_vids = np.asarray(q_vids)
    q_camids = np.asarray(q_camids)
    q_vcolors = np.asarray(q_vcolors)
    q_vtypes = np.asarray(q_vtypes)
    pred_q_vcolors = np.asarray(pred_q_vcolors)
    pred_q_vtypes = np.asarray(pred_q_vtypes)

    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.shape[0], qf.shape[1]))

    gf = []
    g_vids = []
    g_camids = []
    g_vcolors = []
    g_vtypes = []
    pred_g_vcolors = []
    pred_g_vtypes = []
    for i in range(len(os.listdir(input_dir + "input/gallery/img"))):
        single_label = gallery_label["{}".format(i)]
        vids = single_label['vid']
        camids = single_label['camid']
        vcolors = single_label['vcolor']
        vtypes = single_label['vtype']

        output_vcolors_path = result_dir + "veri_data_gallery_img_{}_1.bin".format(i)
        output_vtypes_path = result_dir + "veri_data_gallery_img_{}_2.bin".format(i)
        features_path = result_dir + "veri_data_gallery_img_{}_3.bin".format(i)

        output_vcolors = np.fromfile(output_vcolors_path, dtype=np.float32).reshape(1, 10)
        output_vtypes = np.fromfile(output_vtypes_path, dtype=np.float32).reshape(1, 9)
        features = np.fromfile(features_path, dtype=np.float32).reshape(1, 1024)

        gf.append(features)
        g_vids.extend(vids)
        g_camids.extend(camids)
        g_vcolors.extend(vcolors)
        g_vtypes.extend(vtypes)
        pred_g_vcolors.extend(output_vcolors.tolist())
        pred_g_vtypes.extend(output_vtypes.tolist())

    gf = np.concatenate(gf, axis=0)
    g_vids = np.asarray(g_vids)
    g_camids = np.asarray(g_camids)
    g_vcolors = np.asarray(g_vcolors)
    g_vtypes = np.asarray(g_vtypes)
    pred_g_vcolors = np.asarray(pred_g_vcolors)
    pred_g_vtypes = np.asarray(pred_g_vtypes)

    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.shape[0], gf.shape[1]))

    m, n = qf.shape[0], gf.shape[0]
    distmat = np.broadcast_to(np.power(qf, 2).sum(axis=1, keepdims=True), (m, n)) + \
              np.broadcast_to(np.power(gf, 2).sum(axis=1, keepdims=True), (n, m)).T

    distmat = distmat * 1 + (-2) * (np.matmul(qf, gf.T))

    get_cmc_map(distmat, q_vids, g_vids, q_camids, g_camids)
    get_color_acc(distmat, vcolor2label, q_vcolors, pred_q_vcolors, g_vcolors, pred_g_vcolors)
    get_type_acc(distmat, vcolor2label, q_vtypes, pred_q_vtypes, g_vtypes, pred_g_vtypes)

if __name__ == '__main__':
    acc(args.input_dir, args.result_dir, args.label_dir)
