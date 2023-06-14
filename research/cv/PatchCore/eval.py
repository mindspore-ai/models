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
"""eval"""
import sys
import json
import os
from pathlib import Path

import cv2
import faiss
import numpy as np
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

from src.config import cfg, merge_from_cli_list
from src.dataset import createDataset
from src.model import wide_resnet50_2
from src.oneStep import OneStepCell
from src.operator import embedding_concat, normalize, prep_dirs, reshape_embedding, save_anomaly_map

opts = sys.argv[1:]
merge_from_cli_list(opts)
cfg.freeze()
print(cfg)

if __name__ == "__main__":
    current_path = os.path.abspath(os.path.dirname(__file__))
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.platform, device_id=cfg.device_id)

    # dataset
    mean = cfg.mean
    std = cfg.std
    _, test_dataset, _, test_json_path = createDataset(cfg.dataset_path, cfg.category)
    json_path = Path(test_json_path)
    with json_path.open("r") as label_file:
        label = json.load(label_file)
    data_iter = test_dataset.create_dict_iterator()
    step_size = test_dataset.get_dataset_size()

    embedding_dir_path, sample_path = prep_dirs(current_path, cfg.category)
    index = faiss.read_index(os.path.join(embedding_dir_path, "index.faiss"))

    # network
    network = wide_resnet50_2()
    param_dict = load_checkpoint(cfg.pre_ckpt_path)
    load_param_into_net(network, param_dict)

    for p in network.trainable_params():
        p.requires_grad = False

    model = OneStepCell(network)

    print("***************start eval***************")
    gt_list_px_lvl = []
    pred_list_px_lvl = []
    gt_list_img_lvl = []
    pred_list_img_lvl = []
    img_path_list = []
    for step, data in enumerate(data_iter):
        step_label = label["{}".format(data["idx"][0])]
        file_name = step_label["name"]
        x_type = step_label["img_type"]

        features = model(data["img"])
        embedding = embedding_concat(features[0].asnumpy(), features[1].asnumpy())
        embedding_test = reshape_embedding(embedding)

        embedding_test = np.array(embedding_test, dtype=np.float32)
        score_patches, _ = index.search(embedding_test, k=9)

        anomaly_map = score_patches[:, 0].reshape((28, 28))
        N_b = score_patches[np.argmax(score_patches[:, 0])]
        w = 1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b)))
        score = w * max(score_patches[:, 0])
        gt_np = data["gt"].asnumpy()[0, 0].astype(int)
        anomaly_map_resized = cv2.resize(anomaly_map, (224, 224))
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)

        gt_list_px_lvl.extend(gt_np.ravel())
        pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
        gt_list_img_lvl.append(data["label"].asnumpy()[0])
        pred_list_img_lvl.append(score)
        img_path_list.extend(file_name)
        img = normalize(data["img"], mean, std)
        input_img = cv2.cvtColor(np.transpose(img, (0, 2, 3, 1))[0] * 255, cv2.COLOR_BGR2RGB)
        save_anomaly_map(sample_path, anomaly_map_resized_blur, input_img, gt_np * 255, file_name, x_type)

    pixel_auc = roc_auc_score(gt_list_px_lvl, pred_list_px_lvl)
    img_auc = roc_auc_score(gt_list_img_lvl, pred_list_img_lvl)

    print("\ntest_epoch_end")
    print("category is {}".format(cfg.category))
    print("img_auc: {}, pixel_auc: {}".format(img_auc, pixel_auc))
