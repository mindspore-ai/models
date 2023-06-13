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

import os
import os.path

import numpy as np
import mindspore.dataset as ds


def distribution_creator(value):
    gt = np.zeros(256)
    gt[int(value)] = 1.0
    return gt


def normalization(data):
    data[:, 0] = data[:, 0] / 15
    data[:, 1] = data[:, 1] / 15
    data[:, 2] = data[:, 2] / 15
    data[:, 3] = (data[:, 3] - 4) / 8
    data[:, 4] = (data[:, 4] - 6) / 12
    data[:, 5] = (data[:, 5] - 128) / 256
    return data


class NodeInfoFolder:
    def __init__(self, root, multi=None, node_size=0):
        frames = []
        for filename in os.listdir(root):
            if filename.endswith(".npy"):
                frames.append("{}".format(filename))

        self.root = root
        self.frames = frames
        self.multi = multi
        self.node_size = node_size

    def __getitem__(self, index):
        filename = self.frames[int(index)]
        nodes = np.load(os.path.join(self.root, filename), allow_pickle=True)

        gt = nodes.item().get("gt").astype(np.int32)
        cur_node = normalization(nodes.item().get("cur_node").astype(np.float16))
        parent1 = normalization(nodes.item().get("parent1").astype(np.float16))
        parent2 = normalization(nodes.item().get("parent2").astype(np.float16))
        parent3 = normalization(nodes.item().get("parent3").astype(np.float16))
        feature = np.concatenate([cur_node, parent1, parent2, parent3], axis=1)

        if self.node_size != 0:
            node_num = feature.shape[0]
            random_index = np.random.randint(0, node_num - self.node_size)
            gt_batch = gt[random_index : random_index + self.node_size]
            feature_batch = feature[random_index : random_index + self.node_size, :]
        else:
            gt_batch = gt
            feature_batch = feature

        return feature_batch, gt_batch

    def __len__(self):
        return len(self.frames)


if __name__ == "__main__":
    # For test, print elements' shape
    data_generator = NodeInfoFolder(root="training dataset/Feature/branches/")
    dataset = ds.GeneratorDataset(data_generator, ["feature", "gt"], shuffle=False)

    for ds_data in dataset.create_dict_iterator():
        print(np.shape(ds_data["gt"]))
