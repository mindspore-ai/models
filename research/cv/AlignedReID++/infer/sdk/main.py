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

import argparse
import re
import os
import numpy as np
from StreamManagerApi import MxDataInput
from StreamManagerApi import StreamManagerApi
from StreamManagerApi import  StringVector
import MxpiDataType_pb2 as MxpiDataType


# The supported image suffixes are the four suffixes respectively
SUPPORT_IMG_SUFFIX = (".jpg", ".JPG", ".jpeg", ".JPEG")

# os.path.dirname(__file__) gets the full path of the current script, os.path.abspath() gets the full path of the current script
current_path = os.path.abspath(os.path.dirname(__file__))

# argparse is a parser. argparse blocks make it easy to write user-friendly command-line interfaces.
parser = argparse.ArgumentParser(
    description="AlignedReID infer " "example.",
    fromfile_prefix_chars="@",
)

# name or flags, a list of command or option strings
# str      Cast data to a string.  Each data type can be cast to a string
# help     A brief description of what this option does
# default  The value used when the parameter does not appear on the command line.
parser.add_argument(
    "--mode",
    type=str,
    help="Infer mode",
    default="infer",
    required=False,
)

parser.add_argument(
    "--pipeline_path",
    type=str,
    help="mxManufacture pipeline file path",
    default=os.path.join(current_path, "../data/config/AlignedReID.pipeline"),
)
parser.add_argument(
    "--stream_name",
    type=str,
    help="Infer stream name in the pipeline config file",
    default="detection",
)
parser.add_argument(
    "--img_path",
    type=str,
    help="Image pathname, can be a image file or image directory",
    default=os.path.join(current_path, "../data/infer/query"),
)
# The purpose is to store the result of reasoning
parser.add_argument(
    "--res_path",
    type=str,
    help="Directory to store the inferred result",
    default="./",
    required=False,
)
# gallery_path
parser.add_argument(
    "--gallery_path",
    type=str,
    help="Gallery path",
    default="../data/test/",
    required=False,
)

# Assign the value and then parse the parameters
args = parser.parse_args()

# Infer images
def infer():
    # Encode the stream name in utF-8 format
    stream_name = args.stream_name.encode()
    gallery_path = os.path.abspath(args.gallery_path)
    img_path = os.path.abspath(args.img_path)

    # StreamManagerApi() is used for the basic management of the process: loading the process configuration,
    # creating the process, sending data to the process, and getting execution results
    stream_manager_api = StreamManagerApi()
    # InitManager initializes a StreamManagerApi
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    # Reading pipeline files
    with open(args.pipeline_path, "rb") as f:
        pipeline_str = f.read()

    # CreateMultipleStreams, Creates a Stream based on the specified pipeline configuration
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # The id of the plug-in
    # Construct the input of the stream
    # Construct the stream input, MxDataInput for the data structure definition received by the stream.
    data_input = MxDataInput()

    # Os.path.isfile () is used to check whether an object (with an absolute path) is a file
    # Endswith specifies the end of the string for the image
    if os.path.isfile(img_path) and img_path.endswith(SUPPORT_IMG_SUFFIX):
        query_list = [os.path.abspath(img_path)]
        gallery_list = [os.path.abspath(gallery_path)]
    else:
        # os.path.isdir()is used to determine whether an object is a directory
        query_list = os.listdir(img_path)
        query_list = [os.path.join(img_path, img) for img in query_list if img.endswith(SUPPORT_IMG_SUFFIX)]
        gallery_list = os.listdir(gallery_path)
        gallery_list = [os.path.join(gallery_path, img) for img in gallery_list \
            if img.endswith(SUPPORT_IMG_SUFFIX)]

    qf, q_pids, q_camids = [], [], []
    # Start traversing the query List
    cnt = 0
    cnt1 = int(len(query_list) * 0.50)

    print("Getting query features......")
    for file_name in query_list:
        # Read each photo in turn
        with open(file_name, "rb") as f:
            img_data = f.read()
            # Data Input The data element of this object has the value IMG Data
            data_input.data = img_data
            # SendDataWithUniqueId sends data to the specified component. Type in_plugin_id for the plugin ID, data_input.
            # According to the official API,stream_name should not be used as input
            _ = stream_manager_api.SendData(stream_name, 0, data_input)
            key = b"mxpi_tensorinfer0"
            key_vec = StringVector()
            key_vec.push_back(key)
            infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
            result = MxpiDataType.MxpiTensorPackageList()
            result.ParseFromString(infer_result[0].messageBuf)

            features = np.frombuffer(result.tensorPackageVec[0].tensorVec[1].dataStr,
                                     dtype='<f4').reshape(2048)
            cnt = cnt + 1
            target, pid, camid = process_img(file_name)
            if target:
                qf.append(features)
                q_pids.append(pid)
                q_camids.append(camid)
        if cnt == cnt1:
            print("  50% have done")

    print(f"  100% have done \n -- inferred successfully!")
    qf = np.asarray(qf)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print("-- Extracted features for query set, obtained {}-by-{} matrix".format(qf.shape[0], qf.shape[1]))

    # Save the results into files
    np.save("../data/input/qf.npy", qf)
    np.save("../data/input/q_pids.npy", q_pids)
    np.save("../data/input/q_camids.npy", q_camids)

    gf, g_pids, g_camids = [], [], []
    cnt = 0
    cnt1 = int(len(gallery_list) * 0.50)

    for file_name in gallery_list:
        # Read each photo in turn
        with open(file_name, "rb") as f:
            img_data = f.read()
            data_input.data = img_data
            _ = stream_manager_api.SendData(
                stream_name, 0, data_input)
            key = b"mxpi_tensorinfer0"
            key_vec = StringVector()
            key_vec.push_back(key)
            infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)
            result = MxpiDataType.MxpiTensorPackageList()
            result.ParseFromString(infer_result[0].messageBuf)
            features = np.frombuffer(result.tensorPackageVec[0].tensorVec[1].dataStr,
                                     dtype='<f4').reshape(2048)
            target, pid, camid = process_img(file_name)
            if target:
                gf.append(features)
                g_pids.append(pid)
                g_camids.append(camid)
            cnt = cnt + 1
        if cnt == cnt1:
            print("  50% have done")

    print("  100% have done \n All gallery features are gotten successfully!")
    print("np.array(gf).shape = ", np.array(gf).shape)  # (y, 2048)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    gf = np.asarray(gf)
    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.shape[0], gf.shape[1]))
    stream_manager_api.DestroyAllStreams()

    # Save the results into files
    np.save("../data/input/gf.npy", gf)
    np.save("../data/input/g_pids.npy", g_pids)
    np.save("../data/input/g_camids.npy", g_camids)
    return qf, q_pids, q_camids, gf, g_pids, g_camids


def save_result(query_pids, inferred_pids):
    res_path = args.res_path
    os.makedirs(res_path, exist_ok=True)
    for i in range(query_pids.shape[0]):
        query_pid, inferred_pid = query_pids[i], inferred_pids[i]
        with open(os.path.join(res_path, "./result.json"), "a") as f:
            f.writelines(
                ["query_pid: {0:<6}".format(str(query_pid)), "   ", "inferred_pid: {0:<6}".format(str(inferred_pid)),
                 "\n"])


def process_img(img_path):
    pattern = re.compile(r'([-\d]+)_c(\d)')  # regular expression
    target = True
    pid, camid = map(int, pattern.search(img_path).groups())

    if not 0 <= pid <= 1501:
        target = False
    assert 1 <= camid <= 6
    camid -= 1  # index starts from 0
    return target, pid, camid


def re_ranking(qf, gf, k1, k2, lambda_value, local_distmat=None, only_local=False):
    query_num = np.size(qf, 0)
    all_num = np.size(qf, 0) + np.size(gf, 0)

    if only_local:
        original_dist = local_distmat
    else:
        feat = np.concatenate([qf, gf])

        x = np.broadcast_to(np.power(feat, 2).sum(axis=1, keepdims=True), (all_num, all_num))
        y = np.broadcast_to(np.power(feat, 2).sum(axis=1, keepdims=True), (all_num, all_num)).T
        dist = x + y - 2 * np.matmul(feat, feat.T)
        original_dist = dist

        del feat
        if not local_distmat is None:
            original_dist = original_dist + local_distmat

    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)
    print("==================================")
    print('-- Doing re_ranking......')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[ \
                candidate_forward_k_neigh_index, :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len( \
                        candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum( \
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


def evaluate(qf, q_pids, q_camids, gf, g_pids, g_camids, max_rank=50):
    # normalization
    qf = qf / (np.linalg.norm(qf, 2, axis=1, keepdims=True) + 1)
    gf = gf / (np.linalg.norm(gf, 2, axis=1, keepdims=True) + 1)
    _, _ = np.size(qf, 0), np.size(gf, 0)

    distmat = np.power(qf, 2).sum(axis=1, keepdims=True) + np.power(gf, 2).sum(axis=1, keepdims=True).T
    distmat = distmat + (-2) * np.matmul(qf, gf.T)

    print("==================================")
    print("Doing precision evaluation......")
    print("-- Computing CMC and mAP......")
    cmc, mAP, inferred_pids = eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
    print("-- Results ----------")
    print("-- mAP: {:.1%}".format(mAP))
    print("-- CMC curve")
    for r in [1, 5, 10, 20]:
        print("   Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("==================================")

    # Save results, for inferred pids before re_ranking and after re_ranking
    save_result(q_pids, inferred_pids)

    print("Using global branch for re_ranking......")
    distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.01)
    print("-- Re_ranking has been done successfully")
    print("==================================")
    print("Computing CMC and mAP for re_ranking......")
    cmc, mAP, inferred_pids = eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)

    print("-- Results ----------")
    print("-- mAP(RK): {:.1%}".format(mAP))  # mean Average Precision
    print("-- CMC curve(RK)")  # Cumulative Match Characteristic
    for r in [1, 5, 10, 20]:
        print("   Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("==================================")


# Computing CMC and mAP
def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    print("distmat = ", distmat)
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    print("Waiting for a moment, almost 1s for one query image. q_pids.shape[0] = ", q_pids.shape[0])

    inferred_pids = []
    indices = np.argsort(distmat, axis=1)

    for i in range(q_pids.shape[0]):
        inferred_pids.append(g_pids[indices][i][0])
        print("  query_pid: {0:<4}".format(q_pids[i]), " -> inferred_pid: {0:<6}".format(
            g_pids[indices][i][0]), "   img_index = {0:<6}".format(indices[i][0]))

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP, inferred_pids


if __name__ == "__main__":
    mode = args.mode
    # If you don't have any data before, do this
    if mode == "infer":
        qf_1, q_pids_1, q_camids_1, gf_1, g_pids_1, g_camids_1 = infer()
    if mode == "load_eval":
        qf_1 = np.load("../data/input/qf.npy")
        q_pids_1 = np.load("../data/input/q_pids.npy")
        q_camids_1 = np.load("../data/input/q_camids.npy")
        gf_1 = np.load("../data/input/gf.npy")
        g_pids_1 = np.load("../data/input/g_pids.npy")
        g_camids_1 = np.load("../data/input/g_camids.npy")
        print("All data has been loaded successfully.")
        evaluate(qf_1, q_pids_1, q_camids_1, gf_1, g_pids_1, g_camids_1)
    if mode == "infer_eval":
        qf_1, q_pids_1, q_camids_1, gf_1, g_pids_1, g_camids_1 = infer()
        evaluate(qf_1, q_pids_1, q_camids_1, gf_1, g_pids_1, g_camids_1)
