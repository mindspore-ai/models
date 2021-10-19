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
''' eval MultiTaskNet '''
import csv
import ast
import argparse
import os.path as osp
import cv2
import numpy as np
from utils.inference import SdkInfer, infer_test
from utils.transforms import read_image_color, read_image_grayscale, segs, \
    Compose_Keypt, to_tensor, normalize_mt_input, Resize_Keypt


def eval_market1501(distmat, q_vids, g_vids, q_camids, g_camids, max_rank=50):
    """Evaluation with Market1501 metrics
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print(f"Note: number of gallery samples is quite small, got {num_g}")
    indices = np.argsort(distmat, axis=1)

    matches = (g_vids[indices] == q_vids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_ap = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query vid and camid
        q_vid = q_vids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same vid and camid with query
        order = indices[q_idx]

        remove = (g_vids[order] == q_vid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        ap_ = tmp_cmc.sum() / num_rel
        all_ap.append(ap_)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    all_cmc = np.array(all_cmc, dtype=np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    map_ = np.mean(all_ap)

    return all_cmc, map_


def get_mtn_dataset(label_path, imgs_path, desc='query', relabel=True):
    ''' get img and target value set '''
    dataset = []
    vid_container = set()
    vcolor_container = set()
    vtype_container = set()
    vcolor2label = {}
    vtype2label = {}
    with open(osp.join(label_path, f'label_{desc}.csv'), encoding='utf-8') as label_file:
        reader = csv.reader(label_file, delimiter=',')
        for row in reader:
            vid = int(row[1])
            vid_container.add(vid)
            vcolor = int(row[2])
            vcolor_container.add(vcolor)
            vtype = int(row[3])
            vtype_container.add(vtype)
            vkeypt = []
            for k in range(36):
                vkeypt.extend(
                    [float(row[4+3*k]), float(row[5+3*k]), float(row[6+3*k])])
            # synthetic data do not have camera ID
            camid = -1
            camidx = row[0].find('c')
            if camidx >= 0:
                camid = int(row[0][camidx+1:camidx+4])
            dataset.append([osp.join(imgs_path, f'image_{desc}', row[0]),
                            vid, camid, vcolor, vtype, vkeypt,
                            osp.join(
                                imgs_path, f'heatmap_{desc}', row[0][:-4]),
                            osp.join(imgs_path, f'segment_{desc}', row[0][:-4])])
    if relabel:
        vid2label = {vid: label for label, vid in enumerate(vid_container)}
        vcolor2label = {vcolor: label for label,
                        vcolor in enumerate(vcolor_container)}
        vtype2label = {vtype: label for label,
                       vtype in enumerate(vtype_container)}
        if desc == 'train':
            for v_ in dataset:
                v_[1] = vid2label[v_[1]]
                v_[3] = vcolor2label[v_[3]]
                v_[4] = vtype2label[v_[4]]

    return dataset, vcolor2label, vtype2label


transform = Compose_Keypt([
    Resize_Keypt((256, 256))
])


def get_mtn_dataitem(dataset, index, heatmapaware=True, segmentaware=True, keyptaware=True):
    ''' get input data '''
    img_chnls = []
    img_path, vid, camid, vcolor, vtype, vkeypt, heatmap_dir_path, segment_dir_path = dataset[
        index]

    img_orig = read_image_color(img_path)
    height_orig, width_orig, _ = img_orig.shape
    img_b, img_g, img_r = cv2.split(img_orig)
    img_chnls.extend([img_r, img_g, img_b])

    if heatmapaware:
        for h in range(36):
            heatmap_path = osp.join(heatmap_dir_path, "%02d.jpg" % h)
            heatmap = read_image_grayscale(heatmap_path)
            heatmap = cv2.resize(heatmap, dsize=(width_orig, height_orig))
            img_chnls.append(heatmap)

    if segmentaware:
        for s in range(len(segs)):
            segment_flag = True
            for k in segs[s]:
                # conf_thld = 0.5
                if vkeypt[k * 3+2] < 0.5:
                    segment_flag = False
                    break
            if segment_flag:
                segment_path = osp.join(segment_dir_path, "%02d.jpg" % s)
                segment = read_image_grayscale(segment_path)
                segment = cv2.resize(segment, dsize=(width_orig, height_orig))
            else:
                segment = np.zeros((height_orig, width_orig), np.uint8)
            img_chnls.append(segment)

    # assert transform is not None
    img = np.stack(img_chnls, axis=2)
    img = np.array(img, np.float32)
    img = transform(img, vkeypt)
    img = to_tensor(img)
    img = normalize_mt_input(img, heatmapaware, segmentaware)
    vkeypt = np.array(vkeypt, np.float32)

    # normalize keypt
    if keyptaware:
        for k in range(vkeypt.size):
            if k % 3 == 0:
                vkeypt[k] = (vkeypt[k] / float(256)) - 0.5
            elif k % 3 == 1:
                vkeypt[k] = (vkeypt[k] / float(256)) - 0.5
            elif k % 3 == 2:
                vkeypt[k] -= 0.5

    return img, vid, camid, vcolor, vtype, vkeypt


def eval_multitasknet(imgs_path, label_path, pipline_path, keyptaware=True, multitask=True,
                      return_distmat=True, batchsize=1, heatmapaware=True, segmentaware=True,
                      data_from_pn=True):
    ''' start eval '''
    stream = SdkInfer(pipline_path)
    stream.init_stream()
    mtn_query_dataset, vcolor2label, _ = get_mtn_dataset(
        label_path, imgs_path, 'query')
    mtn_gallary_dataset, _, _ = get_mtn_dataset(
        label_path, imgs_path, 'test', False)
    qf = []
    q_vids = []
    q_camids = []
    q_vcolors = []
    q_vtypes = []
    pred_q_vcolors = []
    pred_q_vtypes = []

    imgs, vids, camids, vcolors, vtypes, vkeypts = [], [], [], [], [], []
    for i in range(len(mtn_query_dataset)):
        if not data_from_pn:
            img, vid, camid, vcolor, vtype, vkeypt = get_mtn_dataitem(mtn_query_dataset, i,
                                                                      heatmapaware=heatmapaware,
                                                                      segmentaware=segmentaware,
                                                                      keyptaware=keyptaware)
            imgs.append(img)
        else:
            img_path, vid, camid, vcolor, vtype, vkeypt, _, _ = mtn_query_dataset[i]

        vids.append(vid)
        camids.append(camid)
        vcolors.append(vcolor)
        vtypes.append(vtype)
        vkeypts.append(vkeypt)
        if (i + 1) % batchsize != 0:
            continue

        if not data_from_pn:
            _ = stream.send_package_buf(b'MultiTaskNet0', np.array(imgs), 0)
            _ = stream.send_package_buf(b'MultiTaskNet0', np.array(vkeypts), 1)
            if keyptaware and multitask:
                _, output_vcolors, output_vtypes, features = stream.get_result(
                    b'MultiTaskNet0', 0)
        else:
            if keyptaware and multitask:
                _, output_vcolors, output_vtypes, features = infer_test(
                    stream, img_path, heatmapaware=heatmapaware, segmentaware=segmentaware)

        qf.append(features)
        q_vids.extend(vids)
        q_camids.extend(camids)
        if multitask:
            q_vcolors.extend(vcolors)
            q_vtypes.extend(vtypes)
            pred_q_vcolors.extend(output_vcolors)
            pred_q_vtypes.extend(output_vtypes)
        imgs, vids, camids, vcolors, vtypes, vkeypts = [], [], [], [], [], []
    qf = np.array(qf, dtype=np.float32)
    qf.shape = (len(qf), 1024)

    # qf = cat(qf) # (1664, 1024)

    q_vids = np.asarray(q_vids)
    q_camids = np.asarray(q_camids)
    if multitask:
        q_vcolors = np.asarray(q_vcolors)
        q_vtypes = np.asarray(q_vtypes)
        pred_q_vcolors = np.asarray(pred_q_vcolors)
        pred_q_vtypes = np.asarray(pred_q_vtypes)
    print(
        f"Extracted features for query set, obtained {qf.shape[0]}-by-{qf.shape[1]} matrix")

    gf = []
    g_vids = []
    g_camids = []
    g_vcolors = []
    g_vtypes = []
    pred_g_vcolors = []
    pred_g_vtypes = []

    for i in range(len(mtn_gallary_dataset)):
        if not data_from_pn:
            img, vid, camid, vcolor, vtype, vkeypt = get_mtn_dataitem(mtn_gallary_dataset, i,
                                                                      heatmapaware=heatmapaware,
                                                                      segmentaware=segmentaware,
                                                                      keyptaware=keyptaware)
            imgs.append(img)
        else:
            img_path, vid, camid, vcolor, vtype, vkeypt, _, _ = mtn_gallary_dataset[i]

        vids.append(vid)
        camids.append(camid)
        vcolors.append(vcolor)
        vtypes.append(vtype)
        vkeypts.append(vkeypt)
        if (i + 1) % batchsize != 0:
            continue

        if not data_from_pn:
            _ = stream.send_package_buf(b'MultiTaskNet0', np.array(imgs), 0)
            _ = stream.send_package_buf(b'MultiTaskNet0', np.array(vkeypts), 1)
            if keyptaware and multitask:
                _, output_vcolors, output_vtypes, features = stream.get_result(
                    b'MultiTaskNet0', 0)
        else:
            if keyptaware and multitask:
                _, output_vcolors, output_vtypes, features = infer_test(
                    stream, img_path, heatmapaware=heatmapaware, segmentaware=segmentaware)

        gf.append(features)  # (32, 1024)
        g_vids.extend(vids)
        g_camids.extend(camids)
        if multitask:
            g_vcolors.extend(vcolors)
            g_vtypes.extend(vtypes)
            pred_g_vcolors.extend(output_vcolors)
            pred_g_vtypes.extend(output_vtypes)
        _, vids, camids, vcolors, vtypes, vkeypts = [], [], [], [], [], []

    stream.destroy()

    gf = np.array(gf, dtype=np.float32)
    gf.shape = (len(gf), 1024)

    g_vids = np.asarray(g_vids)
    g_camids = np.asarray(g_camids)
    if multitask:
        g_vcolors = np.asarray(g_vcolors)
        g_vtypes = np.asarray(g_vtypes)
        pred_g_vcolors = np.asarray(pred_g_vcolors)
        pred_g_vtypes = np.asarray(pred_g_vtypes)

    print(
        f"Extracted features for gallery set, obtained {gf.shape[0]}-by-{gf.shape[1]} matrix")

    m, n = qf.shape[0], gf.shape[0]
    qf_distmat = qf**2
    qf_distmat = np.expand_dims(np.sum(qf_distmat, 1), axis=1)
    qf_distmat = np.broadcast_to(qf_distmat, (m, n))
    gf_distmat = gf**2
    gf_distmat = np.expand_dims(np.sum(gf_distmat, 1), axis=1)
    gf_distmat = np.broadcast_to(gf_distmat, (n, m))
    gf_distmat = gf_distmat.T
    distmat = qf_distmat + gf_distmat
    distmat = distmat*1 + np.matmul(qf, gf.T)*(-2)
    print("Computing CMC and mAP")
    cmc, map_ = eval_market1501(distmat, q_vids, g_vids, q_camids, g_camids, 50)
    print("Results ----------")
    print(f"mAP: {map_:8.2%}")
    print("CMC curve")
    for r in range(1, 51):
        print(f"Rank-{r:<3}: {cmc[r-1]:8.2%}")
    print("------------------")

    if multitask:
        print("Compute attribute classification accuracy")
        for q in range(q_vcolors.size):
            q_vcolors[q] = vcolor2label[q_vcolors[q]]
        for g in range(g_vcolors.size):
            g_vcolors[g] = vcolor2label[g_vcolors[g]]
        q_vcolor_errors = np.argmax(pred_q_vcolors, axis=1) - q_vcolors
        g_vcolor_errors = np.argmax(pred_g_vcolors, axis=1) - g_vcolors
        vcolor_error_num = np.count_nonzero(
            q_vcolor_errors) + np.count_nonzero(g_vcolor_errors)
        vcolor_accuracy = 1.0 - \
            (float(vcolor_error_num) /
             float(distmat.shape[0] + distmat.shape[1]))
        print(
            f"Color classification accuracy: {vcolor_accuracy:8.2%}".format())

        for q_ in range(q_vtypes.size):
            q_vtypes[q_] = vcolor2label[q_vtypes[q_]]
        for g_ in range(g_vtypes.size):
            g_vtypes[g_] = vcolor2label[g_vtypes[g_]]
        q_vtype_errors = np.argmax(pred_q_vtypes, axis=1) - q_vtypes
        g_vtype_errors = np.argmax(pred_g_vtypes, axis=1) - g_vtypes
        vtype_error_num = np.count_nonzero(
            q_vtype_errors) + np.count_nonzero(g_vtype_errors)
        vtype_accuracy = 1.0 - (float(vtype_error_num) /
                                float(distmat.shape[0] + distmat.shape[1]))
        print(f"Type classification accuracy: {vtype_accuracy:8.2%}")

        print("------------------")

    if return_distmat:
        return distmat
    return cmc[0]

parser = argparse.ArgumentParser(description='Eval MultiTaskNet')
parser.add_argument('--img_path', type=str, default='../data/MultiTaskNet/veri')
parser.add_argument('--label_path', type=str, default='../data/MultiTaskNet/veri')
parser.add_argument('--pipline_path', type=str, default='../pipline/pamtri.pipline')
parser.add_argument('--segmentaware', type=ast.literal_eval, default=True)
parser.add_argument('--heatmapaware', type=ast.literal_eval, default=False)
args = parser.parse_args()
if __name__ == '__main__':
    eval_multitasknet(args.img_path, args.label_path, args.pipline_path,
                      segmentaware=args.segmentaware, heatmapaware=args.heatmapaware)
                      