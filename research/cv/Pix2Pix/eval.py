# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
    Evaluate Pix2Pix Model.
"""

import os
import mindspore as ms
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint
from src.dataset.pix2pix_dataset import pix2pixDataset_val, create_val_dataset
from src.models.pix2pix import get_generator
from src.utils.tools import save_image
from src.utils.config import config
from src.utils.moxing_adapter import moxing_wrapper
from src.utils.device_adapter import get_device_id

@moxing_wrapper()
def pix2pix_eval():

    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())

    # Preprocess the data for evaluating
    dataset_val = pix2pixDataset_val(root_dir=config.val_data_dir)
    ds_val = create_val_dataset(dataset_val)
    print("ds:", ds_val.get_dataset_size())
    print("ds:", ds_val.get_col_names())
    print("ds.shape:", ds_val.output_shapes())

    netG = get_generator()
    netG.set_train()
    print("CKPT:", config.ckpt)
    load_checkpoint(config.ckpt, netG)

    if not os.path.isdir(config.predict_dir):
        os.makedirs(config.predict_dir)

    data_loader_val = ds_val.create_dict_iterator(output_numpy=True, num_epochs=config.epoch_num)
    print("=======Starting evaluating Loop=======")
    for i, data in enumerate(data_loader_val):
        input_image = Tensor(data["input_images"])
        fake_image = netG(input_image)
        save_image(fake_image, config.predict_dir + str(i + 1))
        print("=======image", i + 1, "saved success=======")

if __name__ == '__main__':
    pix2pix_eval()
