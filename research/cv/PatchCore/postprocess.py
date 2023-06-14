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
import argparse
import json
import os
from pathlib import Path

import cv2
import faiss
import numpy as np
from mindspore.common import set_seed
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from sklearn.random_projection import SparseRandomProjection

from src.config import cfg
from src.operator import embedding_concat, prep_dirs, reshape_embedding, save_anomaly_map
from src.sampling_methods.kcenter_greedy import kCenterGreedy

set_seed(1)

parser = argparse.ArgumentParser(description="postprocess")

parser.add_argument("--result_dir", type=str, default="")
parser.add_argument("--img_dir", type=str, default="")
parser.add_argument("--label_dir", type=str, default="")
parser.add_argument("--category", type=str, default="screw")
parser.add_argument("--coreset_sampling_ratio", type=float, default=0.01)

args = parser.parse_args()


def normalize(input_, mean_, std_):
    mean_ = np.array(mean_).reshape((-1, 1, 1))
    std_ = np.array(std_).reshape((-1, 1, 1))

    out = np.divide(np.subtract(input_, mean_), std_).astype(np.float32)

    return out


if __name__ == "__main__":
    train_label_path = Path(os.path.join(args.label_dir, "pre_label.json"))
    test_label_path = Path(os.path.join(args.label_dir, "infer_label.json"))
    train_result_path = os.path.join(args.result_dir, "pre")
    test_result_path = os.path.join(args.result_dir, "infer")

    with train_label_path.open("r") as dst_file:
        train_label = json.load(dst_file)
    with test_label_path.open("r") as dst_file:
        test_label = json.load(dst_file)

    test_json_path = test_label["infer_json_path"]

    # dataset
    embedding_dir_path, sample_path = prep_dirs("./", args.category)

    mean = cfg.mean
    std = cfg.std

    json_path = Path(test_json_path)
    with json_path.open("r") as label_file:
        test_label_string = json.load(label_file)

    # train
    embedding_list = []
    for i in range(int(len(os.listdir(train_result_path)) / 2)):
        features_one_path = os.path.join(train_result_path, "data_img_{}_0.bin".format(i))
        features_two_path = os.path.join(train_result_path, "data_img_{}_1.bin".format(i))

        features_one = np.fromfile(features_one_path, dtype=np.float32).reshape(1, 512, 28, 28)
        features_two = np.fromfile(features_two_path, dtype=np.float32).reshape(1, 1024, 14, 14)

        embedding = embedding_concat(features_one, features_two)
        embedding_list.extend(reshape_embedding(embedding))

    total_embeddings = np.array(embedding_list, dtype=np.float32)

    # Random projection
    randomprojector = SparseRandomProjection(n_components="auto", eps=0.9)
    randomprojector.fit(total_embeddings)

    # Coreset Subsampling
    selector = kCenterGreedy(total_embeddings, 0, 0)
    selected_idx = selector.select_batch(
        model=randomprojector, already_selected=[], N=int(total_embeddings.shape[0] * args.coreset_sampling_ratio)
    )
    embedding_coreset = total_embeddings[selected_idx]

    print("initial embedding size : {}".format(total_embeddings.shape))
    print("final embedding size : {}".format(embedding_coreset.shape))

    # faiss
    index = faiss.IndexFlatL2(embedding_coreset.shape[1])
    index.add(embedding_coreset)
    faiss.write_index(index, os.path.join(embedding_dir_path, "index.faiss"))

    # eval
    gt_list_px_lvl = []
    pred_list_px_lvl = []
    gt_list_img_lvl = []
    pred_list_img_lvl = []
    img_path_list = []
    index = faiss.read_index(os.path.join(embedding_dir_path, "index.faiss"))
    for i in range(int(len(os.listdir(test_result_path)) / 2)):
        test_single_label = test_label["{}".format(i)]
        gt = test_single_label["gt"]
        label = test_single_label["label"]
        idx = test_single_label["idx"]
        test_single_label_string = test_label_string["{}".format(idx[0])]
        file_name = test_single_label_string["name"]
        x_type = test_single_label_string["img_type"]

        img_path = os.path.join(args.img_dir, "data_img_{}.bin".format(i))
        features_one_path = os.path.join(test_result_path, "data_img_{}_0.bin".format(i))
        features_two_path = os.path.join(test_result_path, "data_img_{}_1.bin".format(i))

        img = np.fromfile(img_path, dtype=np.float32).reshape(1, 3, 224, 224)
        features_one = np.fromfile(features_one_path, dtype=np.float32).reshape(1, 512, 28, 28)
        features_two = np.fromfile(features_two_path, dtype=np.float32).reshape(1, 1024, 14, 14)

        embedding = embedding_concat(features_one, features_two)
        embedding_test = reshape_embedding(embedding)

        embedding_test = np.array(embedding_test, dtype=np.float32)
        score_patches, _ = index.search(embedding_test, k=9)

        anomaly_map = score_patches[:, 0].reshape((28, 28))
        N_b = score_patches[np.argmax(score_patches[:, 0])]
        w = 1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b)))
        score = w * max(score_patches[:, 0])
        gt_np = np.array(gt)[0, 0].astype(int)
        anomaly_map_resized = cv2.resize(anomaly_map, (224, 224))
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)

        gt_list_px_lvl.extend(gt_np.ravel())
        pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
        gt_list_img_lvl.append(label[0])
        pred_list_img_lvl.append(score)
        img_path_list.extend(file_name)
        img = normalize(img, mean, std)
        input_img = cv2.cvtColor(np.transpose(img, (0, 2, 3, 1))[0] * 255, cv2.COLOR_BGR2RGB)
        save_anomaly_map(sample_path, anomaly_map_resized_blur, input_img, gt_np * 255, file_name, x_type)

    pixel_acc = roc_auc_score(gt_list_px_lvl, pred_list_px_lvl)
    img_acc = roc_auc_score(gt_list_img_lvl, pred_list_img_lvl)

    print("\n310 acc is")
    print("category is {}".format(args.category))
    print("img_acc: {}, pixel_acc: {}".format(img_acc, pixel_acc))
