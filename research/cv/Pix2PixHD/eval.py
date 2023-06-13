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

"""
    Train pix2pixHD model for eval
"""

import os
import mindspore as ms
from src.models.pix2pixHD import Pix2PixHD
from src.dataset.pix2pixHD_dataset import Pix2PixHDDataset, create_eval_dataset
from src.utils.tools import save_image
from src.utils.config import config
from src.utils.local_adapter import get_device_id


def eval_process():
    # Preprocess the data for eval
    dataset = Pix2PixHDDataset(root_dir=config.data_root, is_train=False)
    ds = create_eval_dataset(dataset)

    if config.device_target == "Ascend":
        ms.set_context(device_id=get_device_id(), device_target="Ascend")
    elif config.device_target == "GPU":
        ms.set_context(device_id=get_device_id(), device_target="GPU")

    pix2pixHD = Pix2PixHD(is_train=False)
    predict_image_path = os.path.join(config.predict_dir, config.name)
    if not os.path.exists(predict_image_path):
        os.makedirs(predict_image_path)

    data_loader = ds.create_dict_iterator(output_numpy=True, num_epochs=1)
    print("======Starting evaluating Loop======")
    for i, data in enumerate(data_loader):
        label = data["label"]
        inst = data["inst"]
        image = data["image"]
        feat = data["feat"]
        label, inst, image, feat = pix2pixHD.encode_input(label, inst, image, feat)
        # sample clusters from precomputed features
        if pix2pixHD.use_features and not pix2pixHD.use_encoded_image:
            feat = pix2pixHD.sample_features(inst)
        fake_image = pix2pixHD(label, inst, image, feat)
        save_image(fake_image, predict_image_path + "/" + str(i + 1))
        print("======image", i + 1, "saved success======")


if __name__ == "__main__":
    eval_process()
