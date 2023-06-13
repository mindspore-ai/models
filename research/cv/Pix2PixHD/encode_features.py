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
# ===========================================================================

import os
import numpy as np
from sklearn.cluster import KMeans
import mindspore as ms
from src.models.pix2pixHD import Pix2PixHD
from src.dataset.pix2pixHD_dataset import Pix2PixHDDataset, create_train_dataset
from src.utils.config import config
from src.utils.local_adapter import get_device_id

name = "features"
save_path = os.path.join(config.save_ckpt_dir, config.name)

"==========Initialize=========="

dataset = Pix2PixHDDataset(root_dir=config.data_root)
ds = create_train_dataset(dataset, config.batch_size)
data_loader = ds.create_dict_iterator(output_numpy=True, num_epochs=1)
steps_per_epoch = ds.get_dataset_size()
if config.device_target == "Ascend":
    ms.set_context(device_id=get_device_id(), device_target="Ascend")
elif config.device_target == "GPU":
    ms.set_context(device_id=get_device_id(), device_target="GPU")

pix2pixhd = Pix2PixHD(is_train=False)

"==========Encode features=========="
re_encode = True
if re_encode:
    features = {}
    for label in range(config.label_nc):
        features[label] = np.zeros((0, config.feat_num + 1))
    for i, data in enumerate(data_loader):
        feat = pix2pixhd.encode_features(data["image"], data["inst"])
        for label in range(config.label_nc):
            features[label] = np.append(features[label], feat[label], axis=0)
        print("%d / %d images" % (i + 1, steps_per_epoch))
    save_name = os.path.join(save_path, name + ".npy")
    np.save(save_name, features)

n_clusters = config.n_clusters
load_name = os.path.join(save_path, name + ".npy")
features = np.load(load_name, allow_pickle=True).item()
centers = {}
for label in range(config.label_nc):
    feat = features[label]
    feat = feat[feat[:, -1] > 0.5, :-1]
    if feat.shape[0]:
        n_clusters = min(feat.shape[0], config.n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(feat)
        centers[label] = kmeans.cluster_centers_
save_name = os.path.join(save_path, name + "_clustered_%03d.npy" % config.n_clusters)
np.save(save_name, centers)
print("saving to %s" % save_name)
