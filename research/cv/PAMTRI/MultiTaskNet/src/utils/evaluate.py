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
"""evaluate"""
import numpy as np

def eval_market1501(distmat, q_vids, g_vids, q_camids, g_camids, max_rank):
    """
    Evaluation with Market1501 metrics
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_vids[indices] == q_vids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query vid and camid
        q_vid = q_vids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same vid and camid with query
        order = indices[q_idx]
        remove = (g_vids[order] == q_vid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def evaluate(distmat, q_vids, g_vids, q_camids, g_camids, max_rank=50, use_cython=True):
    return eval_market1501(distmat, q_vids, g_vids, q_camids, g_camids, max_rank)

def test(model, keyptaware, multitask, queryloader, galleryloader,
         vcolor2label, vtype2label, ranks=range(1, 51), return_distmat=False):
    """function eval"""
    model.set_train(False)

    qf = []
    q_vids = []
    q_camids = []
    q_vcolors = []
    q_vtypes = []
    pred_q_vcolors = []
    pred_q_vtypes = []
    for _, data in enumerate(queryloader.create_dict_iterator()):
        imgs = data["img"]
        vids = data["vid"]
        camids = data["camid"]
        vcolors = data["vcolor"]
        vtypes = data["vtype"]
        vkeypts = data["vkeypt"]

        _, output_vcolors, output_vtypes, features = model(imgs, vkeypts)

        qf.append(features.asnumpy())
        q_vids.extend(vids.asnumpy())
        q_camids.extend(camids.asnumpy())
        q_vcolors.extend(vcolors.asnumpy())
        q_vtypes.extend(vtypes.asnumpy())
        pred_q_vcolors.extend(output_vcolors.asnumpy())
        pred_q_vtypes.extend(output_vtypes.asnumpy())

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
    for _, data in enumerate(galleryloader.create_dict_iterator()):
        imgs = data["img"]
        vids = data["vid"]
        camids = data["camid"]
        vcolors = data["vcolor"]
        vtypes = data["vtype"]
        vkeypts = data["vkeypt"]

        _, output_vcolors, output_vtypes, features = model(imgs, vkeypts)

        gf.append(features.asnumpy())
        g_vids.extend(vids.asnumpy())
        g_camids.extend(camids.asnumpy())
        g_vcolors.extend(vcolors.asnumpy())
        g_vtypes.extend(vtypes.asnumpy())
        pred_g_vcolors.extend(output_vcolors.asnumpy())
        pred_g_vtypes.extend(output_vtypes.asnumpy())

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

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_vids, g_vids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    print("Compute attribute classification accuracy")

    for q in range(q_vcolors.size):
        q_vcolors[q] = vcolor2label[q_vcolors[q]]
    for g in range(g_vcolors.size):
        g_vcolors[g] = vcolor2label[g_vcolors[g]]
    q_vcolor_errors = np.argmax(pred_q_vcolors, axis=1) - q_vcolors
    g_vcolor_errors = np.argmax(pred_g_vcolors, axis=1) - g_vcolors
    vcolor_error_num = np.count_nonzero(q_vcolor_errors) + np.count_nonzero(g_vcolor_errors)
    vcolor_accuracy = 1.0 - (float(vcolor_error_num) / float(distmat.shape[0] + distmat.shape[1]))
    print("Color classification accuracy: {:.2%}".format(vcolor_accuracy))

    for q in range(q_vtypes.size):
        q_vtypes[q] = vcolor2label[q_vtypes[q]]
    for g in range(g_vtypes.size):
        g_vtypes[g] = vcolor2label[g_vtypes[g]]
    q_vtype_errors = np.argmax(pred_q_vtypes, axis=1) - q_vtypes
    g_vtype_errors = np.argmax(pred_g_vtypes, axis=1) - g_vtypes
    vtype_error_num = np.count_nonzero(q_vtype_errors) + np.count_nonzero(g_vtype_errors)
    vtype_accuracy = 1.0 - (float(vtype_error_num) / float(distmat.shape[0] + distmat.shape[1]))
    print("Type classification accuracy: {:.2%}".format(vtype_accuracy))

    if return_distmat:
        return distmat
    return cmc[0]

def onnx_test(InferenceSession, input_name, keyptaware, multitask, queryloader, galleryloader,
              vcolor2label, vtype2label, ranks=range(1, 51), return_distmat=False):
    """function eval"""

    qf = []
    q_vids = []
    q_camids = []
    q_vcolors = []
    q_vtypes = []
    pred_q_vcolors = []
    pred_q_vtypes = []
    for _, data in enumerate(queryloader.create_dict_iterator()):
        imgs = data["img"]
        vids = data["vid"]
        camids = data["camid"]
        vcolors = data["vcolor"]
        vtypes = data["vtype"]
        vkeypts = data["vkeypt"]

        _, output_vcolors, output_vtypes, features = InferenceSession.run(None, {input_name[0]: imgs.asnumpy(),
                                                                                 input_name[1]: vkeypts.asnumpy()})

        qf.append(features)
        q_vids.extend(vids.asnumpy())
        q_camids.extend(camids.asnumpy())
        q_vcolors.extend(vcolors.asnumpy())
        q_vtypes.extend(vtypes.asnumpy())
        pred_q_vcolors.extend(output_vcolors)
        pred_q_vtypes.extend(output_vtypes)

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
    for _, data in enumerate(galleryloader.create_dict_iterator()):
        imgs = data["img"]
        vids = data["vid"]
        camids = data["camid"]
        vcolors = data["vcolor"]
        vtypes = data["vtype"]
        vkeypts = data["vkeypt"]

        _, output_vcolors, output_vtypes, features = InferenceSession.run(None, {input_name[0]: imgs.asnumpy(),
                                                                                 input_name[1]: vkeypts.asnumpy()})
        gf.append(features)
        g_vids.extend(vids.asnumpy())
        g_camids.extend(camids.asnumpy())
        g_vcolors.extend(vcolors.asnumpy())
        g_vtypes.extend(vtypes.asnumpy())
        pred_g_vcolors.extend(output_vcolors)
        pred_g_vtypes.extend(output_vtypes)

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

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_vids, g_vids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    print("Compute attribute classification accuracy")

    for q in range(q_vcolors.size):
        q_vcolors[q] = vcolor2label[q_vcolors[q]]
    for g in range(g_vcolors.size):
        g_vcolors[g] = vcolor2label[g_vcolors[g]]
    q_vcolor_errors = np.argmax(pred_q_vcolors, axis=1) - q_vcolors
    g_vcolor_errors = np.argmax(pred_g_vcolors, axis=1) - g_vcolors
    vcolor_error_num = np.count_nonzero(q_vcolor_errors) + np.count_nonzero(g_vcolor_errors)
    vcolor_accuracy = 1.0 - (float(vcolor_error_num) / float(distmat.shape[0] + distmat.shape[1]))
    print("Color classification accuracy: {:.2%}".format(vcolor_accuracy))

    for q in range(q_vtypes.size):
        q_vtypes[q] = vcolor2label[q_vtypes[q]]
    for g in range(g_vtypes.size):
        g_vtypes[g] = vcolor2label[g_vtypes[g]]
    q_vtype_errors = np.argmax(pred_q_vtypes, axis=1) - q_vtypes
    g_vtype_errors = np.argmax(pred_g_vtypes, axis=1) - g_vtypes
    vtype_error_num = np.count_nonzero(q_vtype_errors) + np.count_nonzero(g_vtype_errors)
    vtype_accuracy = 1.0 - (float(vtype_error_num) / float(distmat.shape[0] + distmat.shape[1]))
    print("Type classification accuracy: {:.2%}".format(vtype_accuracy))

    if return_distmat:
        return distmat
    return cmc[0]
