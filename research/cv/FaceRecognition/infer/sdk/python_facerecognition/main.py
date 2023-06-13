# coding = utf-8
"""
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import math
import os
import time
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi
from StreamManagerApi import MxDataInput, StringVector
from config import config

zj_test_result_name = "zj_result.txt"
jk_test_result_name = "jk_result.txt"


def l2normalize(features):
    epsilon = 1e-12
    l2norm = np.sum(np.abs(features) ** 2, axis=1, keepdims=True) ** (1.0 / 2)
    l2norm[np.logical_and(l2norm < 0, l2norm > -epsilon)] = -epsilon
    l2norm[np.logical_and(l2norm >= 0, l2norm < epsilon)] = epsilon
    return features / l2norm


def check_minmax(data, min_value=0.99, max_value=1.01):
    min_data = data.min()
    max_data = data.max()
    if np.isnan(min_data) or np.isnan(max_data):
        print("ERROR, nan happened, please check if used fp16 or other error")
        raise Exception
    if min_data < min_value or max_data > max_value:
        print(
            "ERROR, min or max is out if range, range=[{}, {}], minmax=[{}, {}]".format(
                min_value, max_value, min_data, max_data
            )
        )
        raise Exception


def generate_test_pair(jk_list, zj_list):
    """generate_test_pair"""
    file_paths = [jk_list, zj_list]
    jk_dict = {}
    zj_dict = {}
    jk_zj_dict_list = [jk_dict, zj_dict]
    # The following code is to put the pictures corresponding to the tags (ids) in each list into a list,
    # for example, people has n pictures, so x_dict [people] can take out the paths of the n pictures.
    for path, x_dict in zip(file_paths, jk_zj_dict_list):
        with open(path, "r") as fr:
            for line in fr:
                label = line.strip().split(" ")[1]
                tmp = x_dict.get(label, [])
                tmp.append(line.strip())
                x_dict[label] = tmp
    zj2jk_pairs = []
    # Here is zj and jk two lists are stored inside the path corresponding to the id,
    # so they are directly combined with each other
    for key in jk_dict:
        jk_file_list = jk_dict[key]
        zj_file_list = zj_dict[key]
        for zj_file in zj_file_list:
            zj2jk_pairs.append([zj_file, jk_file_list])
    return zj2jk_pairs


class DistributedSampler:
    """DistributedSampler"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.num_replicas = 1
        self.rank = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))

    def __iter__(self):
        # Here len(self.dataset) means the number of people in the test image
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank :: self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples


def topk(matrix, k, axis=1):
    """topk"""
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, k, axis=axis)[0:k, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[topk_index_sort, row_index]
        topk_index_sort = topk_index[0:k, :][topk_index_sort, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, k, axis=axis)[:, 0:k]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:, 0:k][column_index, topk_index_sort]
    return topk_data_sort, topk_index_sort, topk_index


def cal_topk(idx, zj2jk_pairs, test_embedding_tot, dis_embedding_tot, dis_labels):
    """cal_topk"""
    correct = np.array([0] * 2)
    tot = np.array([0])

    # Here we get zj's label and all the images corresponding to this person
    zj, jk_all = zj2jk_pairs[idx]
    # Get the feature vector of the id
    zj_embedding = test_embedding_tot[zj]
    # Here is to take out all the feature vectors of all the images corresponding to zj in jk
    jk_all_embedding = np.concatenate([np.expand_dims(test_embedding_tot[jk], axis=0) for jk in jk_all], axis=0)

    test_time = time.time()
    # mm is the vector in zj multiplied by all the vectors in dis_embedding zj(1,256) dis(257,256)
    mm = np.matmul(np.expand_dims(zj_embedding, axis=0), dis_embedding_tot)
    # Here the dimension (1, N) is turned into (N, )
    _, _, jk2zj_sort_1 = topk(mm, 1)
    top100_jk2zj = np.squeeze(topk(mm, 1)[0], axis=0)
    top100_zj2jk, _, zj2jk_sort_1 = topk(np.matmul(jk_all_embedding, dis_embedding_tot), 1)
    test_time_used = time.time() - test_time
    print("INFO, calculate top1 acc index:{}, np.matmul().top(100) time used:{:.2f}s".format(idx, test_time_used))
    tot[0] = len(jk_all)

    for i, jk in enumerate(jk_all):
        jk_embedding = test_embedding_tot[jk]
        similarity = np.dot(jk_embedding, zj_embedding)
        # write the groundtruth to the txt
        writeresult(1, zj + " ")
        writeresult(0, jk + " ")
        if similarity > top100_jk2zj[0]:
            writeresult(1, zj + "\n")
            correct[0] += 1
        else:
            writeresult(1, dis_labels[jk2zj_sort_1[0, 0]] + "\n")
        if similarity > top100_zj2jk[i, 0]:
            writeresult(0, jk + "\n")
            correct[1] += 1
        else:
            writeresult(0, dis_labels[zj2jk_sort_1[i, 0]] + "\n")
    return correct, tot


def writeresult(flag=1, string="test"):
    # write the result to zj
    if flag == 1:
        with open(zj_test_result_name, "a") as f:
            f.write(string)
    # write the result to jk
    else:
        with open(jk_test_result_name, "a") as f:
            f.write(string)


stream_manager_api = StreamManagerApi()
ret = stream_manager_api.InitManager()
if ret != 0:
    print("Failed to init Stream manager, ret=%s" % str(ret))
    exit()
print("create streams by pipeline config file")

with open("../pipeline/facerecognition.pipeline", "rb") as pipeline_file:
    pipelineStr = pipeline_file.read()
ret = stream_manager_api.CreateMultipleStreams(pipelineStr)

if ret != 0:
    print("Failed to create Stream, ret=%s" % str(ret))
    exit()
print("Construct the input of the stream")

data_input = MxDataInput()


def calculate(file_path):
    """calculate the output"""

    with open(file_path, "rb") as f:
        print("processing img ", file_path)
        data_input.data = f.read()

    stream_name = b"im_resnetface"
    in_plugin_id = 0
    unique_id = stream_manager_api.SendData(stream_name, in_plugin_id, data_input)
    if unique_id < 0:
        print("Failed to send data to stream.")
        exit()

    keys = [b"mxpi_tensorinfer0"]
    keyVec = StringVector()
    for key in keys:
        keyVec.push_back(key)

    infer_result = stream_manager_api.GetProtobuf(stream_name, unique_id, keyVec)

    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)

    result = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype="float32")

    out = np.expand_dims(result, axis=0)
    out = out.astype(np.float32)

    embeddings = l2normalize(out)
    return embeddings


def run_eval(test_img_predix, test_img_list, dis_img_predix, dis_img_list):
    """init stream manager"""

    print(" read zj_list and jk_list ")

    zj_jk_labels = []
    zj_jk_imgs = []
    print(" jk's txt first  zj's txt second ")

    for file in test_img_list:
        with open(file, "r") as ft:
            lines = ft.readlines()
            for line in lines:
                imgpath = line.strip().split(" ")[0]
                zj_jk_imgs.append(imgpath)
                zj_jk_labels.append(line.strip())
    print(" test img total number ")

    img_tot = len(zj_jk_labels)
    print("*" * 20, "total img number is {}".format(img_tot))
    print("This is the feature vector used to store the images in jk and zj ")

    test_embedding_tot_np = np.zeros((img_tot, config.emb_size))
    test_img_labels = zj_jk_labels
    print(" Read the images in turn and get the corresponding feature vectors ")

    for index in range(img_tot):
        file_path = zj_jk_imgs[index]
        embeddings = calculate(file_path)
        test_embedding_tot_np[index] = embeddings[0]
    # there aim is to check value
    try:
        check_minmax(np.linalg.norm(test_embedding_tot_np, ord=2, axis=1))
    except ValueError:
        print("-" * 20, "error occur!!")

    # Construct the feature vectors of the images in the test set as key-value pairs ,Use each line of the txt as a key
    test_embedding_tot = {}
    for index in range(img_tot):
        test_embedding_tot[test_img_labels[index]] = test_embedding_tot_np[index]

    # for dis images
    dis_labels = []
    dis_img = []
    with open(dis_img_list[0], "r") as ft:
        lines = ft.readlines()
        for line in lines:
            imgpath = line.strip().split(" ")[0]
            # 得到dis_embedding当中对应的人的id
            dis_labels.append(imgpath)
            dis_img.append(line.strip())
    dis_img_tot = len(dis_labels)
    dis_embedding_tot_np = np.zeros((dis_img_tot, config.emb_size))

    print("dis_label is ", dis_labels)
    for index in range(dis_img_tot):
        file_path = dis_img[index]
        embeddings = calculate(file_path)
        dis_embedding_tot_np[index] = embeddings[0]
        # there aim is to check value
    try:
        check_minmax(np.linalg.norm(dis_embedding_tot_np, ord=2, axis=1))
    except ValueError:
        print("-" * 20, "error occur!!")

    # convert the dis_embedding_tot_np shape to (emb_size , total dis img number)
    dis_embedding_tot_np = np.transpose(dis_embedding_tot_np, (1, 0))

    # find best match
    assert len(test_img_list) % 2 == 0
    task_num = int(len(test_img_list) / 2)
    correct = np.array([0] * (2 * task_num))
    tot = np.array([0] * task_num)

    # calculate the accuracy
    for i in range(int(len(test_img_list) / 2)):
        jk_list = test_img_list[2 * i]
        zj_list = test_img_list[2 * i + 1]

        # merge the data (zj and jk)
        zj2jk_pairs = sorted(generate_test_pair(jk_list, zj_list))
        print("-" * 20, "数据融合完成!")

        # Here you only need the number of samplers, that is, the number of test objects in zj_list.txt
        sampler = DistributedSampler(zj2jk_pairs)
        print("INFO, calculate top1 acc sampler len:{}".format(len(sampler)))
        for idx in sampler:
            out1, out2 = cal_topk(idx, zj2jk_pairs, test_embedding_tot, dis_embedding_tot_np, dis_labels)
            correct[2 * i] += out1[0]
            correct[2 * i + 1] += out1[1]
            tot[i] += out2[0]

    print("tot={},correct={}".format(tot, correct))

    for i in range(int(len(test_img_list) / 2)):
        test_set_name = "test_dataset"
        zj2jk_acc = correct[2 * i] / tot[i]
        jk2zj_acc = correct[2 * i + 1] / tot[i]
        avg_acc = (zj2jk_acc + jk2zj_acc) / 2
        results = "[{}]: zj2jk={:.4f}, jk2zj={:.4f}, avg={:.4f}".format(test_set_name, zj2jk_acc, jk2zj_acc, avg_acc)
        print(results)
    # destroy streams
    stream_manager_api.DestroyAllStreams()


if __name__ == "__main__":
    test_dir = config.test_dir
    test_img_predix_fixversion = [
        os.path.join(test_dir, config.test_img_predix),
        os.path.join(test_dir, config.test_img_predix),
    ]
    test_img_list_fixversion = [
        os.path.join(test_dir, config.test_img_list_jk),
        os.path.join(test_dir, config.test_img_list_zj),
    ]
    dis_img_predix_fixversion = [os.path.join(test_dir, config.dis_img_predix)]
    dis_img_list_fixversion = [os.path.join(test_dir, config.dis_img_list)]

    run_eval(test_img_predix_fixversion, test_img_list_fixversion, dis_img_predix_fixversion, dis_img_list_fixversion)
