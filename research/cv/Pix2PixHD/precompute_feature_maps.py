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
import mindspore as ms
import mindspore.nn as nn
from src.models.pix2pixHD import Pix2PixHD
from src.dataset.pix2pixHD_dataset import Pix2PixHDDataset, create_train_dataset
from src.utils.config import config
from src.utils.local_adapter import get_device_id
from src.utils.tools import save_image

name = "features"
save_path = os.path.join(config.save_ckpt_dir, config.name)

"======Initialize======"
dataset = Pix2PixHDDataset(root_dir=config.data_root)
ds = create_train_dataset(dataset, config.batch_size)
data_loader = ds.create_dict_iterator(output_numpy=True, num_epochs=1)
steps_per_epoch = ds.get_dataset_size()
ms.set_context(device_id=get_device_id(), device_target=config.device_target)

pix2pixHD = Pix2PixHD()
feat_img_path = os.path.join(config.data_root, config.phase + "_feat")
if not os.path.exists(feat_img_path):
    os.makedirs(feat_img_path)

netE = pix2pixHD.netE
resizeBilinear = nn.ResizeBilinear()

"======Save precomputed feature maps for 1024p training======"
for i, data in enumerate(data_loader):
    print("%d/%d images" % (i + 1, steps_per_epoch))
    image = ms.Tensor(data["image"])
    inst = ms.Tensor(data["inst"])
    feat_map = netE(image, inst)
    feat_map = resizeBilinear(feat_map, scale_factor=2)
    save_path = data["path"][0].replace("/train_label", "/train_feat")
    save_path = os.path.splitext(save_path)[0]
    save_image(feat_map, save_path, format_name=".png")
