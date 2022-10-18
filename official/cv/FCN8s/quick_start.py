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
"""quick start"""

from mindspore import load_checkpoint, load_param_into_net
from src.nets.FCN8s import FCN8s
from src.model_utils.config import config
from PIL import Image
from eval import eval_batch_scales, cal_hist
import numpy as np
import matplotlib.pyplot as plt

network = FCN8s(n_class=config.num_classes)
# load model
param_dict = load_checkpoint(config.ckpt_file)
# load parameters to the network
load_param_into_net(network, param_dict)

with open(config.data_lst) as f:
    img_lst = f.readlines()
hist = np.zeros((config.num_classes, config.num_classes))
batch_img_lst = []
batch_msk_lst = []
bi = 0
# eval
for i, line in enumerate(img_lst):
    img_name = line.strip('\n')
    data_root = config.data_root
    img_path = data_root + '/JPEGImages/' + str(img_name) + '.jpg'
    msk_path = data_root + '/SegmentationClass/' + str(img_name) + '.png'

    img_ = np.array(Image.open(img_path), dtype=np.uint8)
    msk_ = np.array(Image.open(msk_path), dtype=np.uint8)

    batch_img_lst.append(img_)
    batch_msk_lst.append(msk_)
    bi += 1
    if bi == config.eval_batch_size:
        batch_res = eval_batch_scales(config, network, batch_img_lst, scales=config.scales,
                                      base_crop_size=config.crop_size, flip=config.flip)
        for mi in range(config.eval_batch_size):
            hist += cal_hist(batch_msk_lst[mi].flatten(), batch_res[mi].flatten(), config.num_classes)


# calculate mean IoU
iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
print('per-class IoU', iu)
print('mean IoU', np.nanmean(iu))
# the number of the visual images
quick_num = 2

quick_num_lst = np.random.randint(0, config.eval_batch_size, quick_num)
# visualization and save
for i in range(0, quick_num):
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(batch_img_lst[quick_num_lst[i]], interpolation="None")#index: i + quick_num
    plt.subplot(1, 3, 2)
    plt.imshow(batch_msk_lst[quick_num_lst[i]], interpolation="None", cmap="gray")
    plt.subplot(1, 3, 3)
    plt.imshow(batch_res[quick_num_lst[i]], interpolation="None", cmap="gray")
    plt.savefig('quick{}.png'.format(quick_num_lst[i]))

plt.show()
