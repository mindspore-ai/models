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
"""train"""

import sys
import datetime
import os
import time
import faiss
import numpy as np
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from sklearn.random_projection import SparseRandomProjection

from src.dataset import createDataset
from src.model import wide_resnet50_2
from src.oneStep import OneStepCell
from src.operator import embedding_concat, prep_dirs, reshape_embedding
from src.sampling_methods.kcenter_greedy import kCenterGreedy

from src.config import cfg, merge_from_cli_list

set_seed(1)

opts = sys.argv[1:]
merge_from_cli_list(opts)
cfg.freeze()
print(cfg)

if cfg.isModelArts:
    import moxing as mox

if __name__ == "__main__":
    current_path = os.path.abspath(os.path.dirname(__file__))
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.platform, save_graphs=False)
    if cfg.isModelArts:
        device_id = int(os.getenv("DEVICE_ID"))
        context.set_context(device_id=device_id)
    else:
        context.set_context(device_id=cfg.device_id)

    # dataset
    if cfg.isModelArts:
        mox.file.copy_parallel(src_url=cfg.data_url, dst_url="/cache/dataset/device_" + os.getenv("DEVICE_ID"))
        train_dataset_path = "/cache/dataset/device_" + os.getenv("DEVICE_ID")
        prep_path = "/cache/train_output/device_" + os.getenv("DEVICE_ID")

        train_dataset, _, _, _ = createDataset(train_dataset_path, cfg.category)
        embedding_dir_path, _ = prep_dirs(prep_path, cfg.category)
    else:
        train_dataset, _, _, _ = createDataset(cfg.dataset_path, cfg.category)
        embedding_dir_path, _ = prep_dirs(current_path, cfg.category)

    # network
    network = wide_resnet50_2()
    param_dict = load_checkpoint(cfg.pre_ckpt_path)
    load_param_into_net(network, param_dict)

    for p in network.trainable_params():
        p.requires_grad = False

    model = OneStepCell(network)

    # train
    embedding_list = []
    print("***************start train***************")
    for epoch in range(cfg.num_epochs):
        data_iter = train_dataset.create_dict_iterator()
        step_size = train_dataset.get_dataset_size()

        for step, data in enumerate(data_iter):
            # time
            start = datetime.datetime.fromtimestamp(time.time())
            features = model(data["img"])
            end = datetime.datetime.fromtimestamp(time.time())
            step_time = (end - start).microseconds / 1000.0
            print("step: {}, time: {}ms".format(step, step_time))

            embedding = embedding_concat(features[0].asnumpy(), features[1].asnumpy())
            embedding_list.extend(reshape_embedding(embedding))

        total_embeddings = np.array(embedding_list, dtype=np.float32)

        # Random projection
        randomprojector = SparseRandomProjection(n_components="auto", eps=0.9)
        randomprojector.fit(total_embeddings)

        # Coreset Subsampling
        selector = kCenterGreedy(total_embeddings, 0, 0)
        selected_idx = selector.select_batch(
            model=randomprojector, already_selected=[], N=int(total_embeddings.shape[0] * cfg.coreset_sampling_ratio)
        )
        embedding_coreset = total_embeddings[selected_idx]

        print("initial embedding size : {}".format(total_embeddings.shape))
        print("final embedding size : {}".format(embedding_coreset.shape))

        # faiss
        index = faiss.IndexFlatL2(embedding_coreset.shape[1])
        index.add(embedding_coreset)
        faiss.write_index(index, os.path.join(embedding_dir_path, "index.faiss"))

    if cfg.isModelArts:
        mox.file.copy_parallel(src_url="/cache/train_output", dst_url=cfg.train_url)

    print("***************train end***************")
